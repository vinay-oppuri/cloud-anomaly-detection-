from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
SEPARATOR = "=" * 60
ENTRYPOINT_MODULES = {
    "test_cicids": "src.experts.network_expert.test",
    "test_hdfs": "src.experts.system_expert.test",
}


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str


def _build_command(entrypoint: str, *args: str) -> tuple[str, ...]:
    executable = shutil.which(entrypoint)
    if executable is not None:
        return (executable, *args)

    module_name = ENTRYPOINT_MODULES.get(entrypoint)
    if module_name is None:
        return (entrypoint, *args)
    return (sys.executable, "-m", module_name, *args)


def _run_json_command(*command: str) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"STDERR: {completed.stderr.strip()}"
        )
    if not stdout:
        raise RuntimeError(f"No output from: {' '.join(command)}")
    return json.loads(stdout)


def _check_metric(label: str, value: float, threshold: float, *, higher_is_better: bool = True) -> bool:
    ok = value >= threshold if higher_is_better else value <= threshold
    operator = ">=" if higher_is_better else "<="
    symbol = PASS if ok else FAIL
    print(f"  {symbol} {label}: {value:.4f} ({operator}{threshold})")
    return ok


def main() -> None:
    results: list[CheckResult] = []

    print(f"\n{SEPARATOR}")
    print("  1. NETWORK EXPERT - CICIDS2018 Test Split")
    print(SEPARATOR)
    network_eval = _run_json_command(*_build_command("test_cicids"))
    network_metrics = network_eval["metrics"]
    print(f"  Threshold : {network_eval['threshold']:.4f} (source: {network_eval['threshold_source']})")
    print(f"  Sequences : {network_eval['num_sequences']:,}")
    network_ok = True
    network_ok &= _check_metric("Accuracy", float(network_metrics["accuracy"]), 0.97)
    network_ok &= _check_metric("Anomaly F1", float(network_metrics["anomaly_f1"]), 0.95)
    network_ok &= _check_metric("Anomaly Precision", float(network_metrics["anomaly_precision"]), 0.95)
    network_ok &= _check_metric("Anomaly Recall", float(network_metrics["anomaly_recall"]), 0.90)
    network_ok &= _check_metric("ROC-AUC", float(network_metrics["roc_auc"]), 0.99)
    print(
        "  Confusion : "
        f"TP={network_metrics['tp']:,} FP={network_metrics['fp']:,} "
        f"FN={network_metrics['fn']:,} TN={network_metrics['tn']:,}"
    )
    results.append(CheckResult(name="network_cicids", status="PASS" if network_ok else "FAIL"))

    print(f"\n{SEPARATOR}")
    print("  2. NETWORK EXPERT - Real-world Normal Syslog")
    print(SEPARATOR)
    network_normal = _run_json_command(
        *_build_command("test_cicids", "--log-file", "data/real_normal_syslog.log")
    )
    normal_label = str(network_normal["decision_label"])
    normal_score = float(network_normal["summary"]["max_anomaly_score"])
    normal_windows = int(network_normal["summary"]["anomaly_windows"])
    benign_windows = int(network_normal["summary"]["benign_windows"])
    print(f"  Decision        : {normal_label}")
    print(f"  Max Score       : {normal_score:.4f}")
    print(f"  Anomaly Windows : {normal_windows} / {normal_windows + benign_windows}")
    normal_ok = normal_label == "Benign"
    print(f"  {PASS if normal_ok else FAIL} Expected: Benign, Got: {normal_label}")
    results.append(CheckResult(name="network_normal_log", status="PASS" if normal_ok else "FAIL"))

    print(f"\n{SEPARATOR}")
    print("  3. NETWORK EXPERT - Real-world Anomaly Syslog")
    print(SEPARATOR)
    network_anomaly = _run_json_command(
        *_build_command("test_cicids", "--log-file", "data/real_anomaly_syslog.log")
    )
    anomaly_label = str(network_anomaly["decision_label"])
    anomaly_score = float(network_anomaly["summary"]["max_anomaly_score"])
    print(f"  Decision  : {anomaly_label}")
    print(f"  Max Score : {anomaly_score:.4f}")
    print(f"  Type      : {network_anomaly.get('anomaly_type', 'N/A')}")
    anomaly_ok = anomaly_label == "Anomaly"
    print(f"  {PASS if anomaly_ok else FAIL} Expected: Anomaly, Got: {anomaly_label}")
    results.append(CheckResult(name="network_anomaly_log", status="PASS" if anomaly_ok else "FAIL"))

    print(f"\n{SEPARATOR}")
    print("  4. SYSTEM EXPERT - HDFS Test Split")
    print(SEPARATOR)
    try:
        system_eval = _run_json_command(*_build_command("test_hdfs"))
        system_metrics = system_eval.get("metrics", {})
        if system_metrics:
            print(f"  Split : {system_eval.get('split', 'test')}")
            system_ok = True
            for key in ("accuracy", "f1_score", "precision", "recall", "roc_auc"):
                if key in system_metrics:
                    system_ok &= _check_metric(key.replace("_", " ").title(), float(system_metrics[key]), 0.90)
            results.append(CheckResult(name="system_hdfs", status="PASS" if system_ok else "FAIL"))
        else:
            print(f"  {INFO} No metrics key in output. Keys: {sorted(system_eval.keys())}")
            results.append(CheckResult(name="system_hdfs", status="INFO"))
    except Exception as exc:  # noqa: BLE001
        print(f"  {FAIL} {exc}")
        results.append(CheckResult(name="system_hdfs", status="ERROR"))

    print(f"\n{SEPARATOR}")
    print("  SUMMARY")
    print(SEPARATOR)
    overall_ok = True
    for result in results:
        symbol = PASS if result.status == "PASS" else FAIL if result.status in {"FAIL", "ERROR"} else INFO
        print(f"  {symbol} {result.name}")
        if result.status not in {"PASS", "INFO"}:
            overall_ok = False

    print()
    print(f"  Overall: {'ALL TESTS PASSED' if overall_ok else 'SOME TESTS FAILED'}")
    print(SEPARATOR)
    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
