"""
Full end-to-end validation of both experts.
Run: uv run python data/run_model_tests.py
"""
import subprocess, json, sys

PASS = "[PASS]"
FAIL = "[FAIL]"
SEP  = "=" * 60

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    out = r.stdout.strip()
    if not out:
        raise RuntimeError(f"No output from: {cmd}\nSTDERR: {r.stderr[:500]}")
    return json.loads(out)

def check(label, value, threshold, higher_is_better=True):
    ok = value >= threshold if higher_is_better else value <= threshold
    sym = PASS if ok else FAIL
    print(f"  {sym}  {label}: {value:.4f}  (threshold={'>=' if higher_is_better else '<='}{threshold})")
    return ok

results = {}

# ── 1. NETWORK EXPERT — CICIDS test split ────────────────────────
print(f"\n{SEP}")
print("  1. NETWORK EXPERT — CICIDS2018 Test Split")
print(SEP)
d = run("uv run test_cicids")
m = d["metrics"]
print(f"  Threshold : {d['threshold']:.4f}  (source: {d['threshold_source']})")
print(f"  Sequences : {d['num_sequences']:,}")
ok = True
ok &= check("Accuracy",         m["accuracy"],          0.97)
ok &= check("Anomaly F1",       m["anomaly_f1"],        0.95)
ok &= check("Anomaly Precision",m["anomaly_precision"],  0.95)
ok &= check("Anomaly Recall",   m["anomaly_recall"],    0.90)
ok &= check("ROC-AUC",          m["roc_auc"],           0.99)
print(f"  Confusion  TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TN={m['tn']:,}")
results["network_cicids"] = "PASS" if ok else "FAIL"

# ── 2. NETWORK EXPERT — Real-world normal syslog ─────────────────
print(f"\n{SEP}")
print("  2. NETWORK EXPERT — Real-world Normal Syslog")
print(SEP)
d = run("uv run test_cicids --log-file data/real_normal_syslog.log")
label   = d["decision_label"]
score   = d["summary"]["max_anomaly_score"]
windows = d["summary"]["anomaly_windows"]
print(f"  Decision        : {label}")
print(f"  Max Score       : {score:.4f}")
print(f"  Anomaly Windows : {windows} / {d['summary']['anomaly_windows'] + d['summary']['benign_windows']}")
ok = label == "Benign"
print(f"  {PASS if ok else FAIL}  Expected: Benign, Got: {label}")
results["network_normal_log"] = "PASS" if ok else "FAIL"

# ── 3. NETWORK EXPERT — Real-world anomaly syslog ────────────────
print(f"\n{SEP}")
print("  3. NETWORK EXPERT — Real-world Anomaly Syslog")
print(SEP)
d = run("uv run test_cicids --log-file data/real_anomaly_syslog.log")
label = d["decision_label"]
score = d["summary"]["max_anomaly_score"]
print(f"  Decision  : {label}")
print(f"  Max Score : {score:.4f}")
print(f"  Type      : {d.get('anomaly_type','N/A')}")
ok = label == "Anomaly"
print(f"  {PASS if ok else FAIL}  Expected: Anomaly, Got: {label}")
results["network_anomaly_log"] = "PASS" if ok else "FAIL"

# ── 4. SYSTEM EXPERT — HDFS test split ───────────────────────────
print(f"\n{SEP}")
print("  4. SYSTEM EXPERT — HDFS Test Split")
print(SEP)
try:
    d = run("uv run test_hdfs")
    m = d.get("metrics", {})
    if m:
        print(f"  Split : {d.get('split','test')}")
        ok = True
        for key in ["accuracy","f1_score","precision","recall","roc_auc"]:
            if key in m:
                ok &= check(key.replace("_", " ").title(), float(m[key]), 0.90)
        results["system_hdfs"] = "PASS" if ok else "FAIL"
    else:
        print("  (No metrics key in output — printing raw keys)")
        print(" ", list(d.keys()))
        results["system_hdfs"] = "INFO"
except Exception as e:
    print(f"  ERROR: {e}")
    results["system_hdfs"] = "ERROR"

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  SUMMARY")
print(SEP)
all_pass = True
for name, status in results.items():
    sym = PASS if status == "PASS" else (FAIL if status in ("FAIL","ERROR") else "[INFO]")
    print(f"  {sym}  {name}")
    if status not in ("PASS", "INFO"):
        all_pass = False
print()
print("  Overall:", "ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
print(SEP)
