from __future__ import annotations

from typing import Iterable


MITIGATION_MAP: dict[str, list[str]] = {
    "DDoS": [
        "Enable rate limiting or upstream scrubbing on the affected service edge.",
        "Block top offending sources and tighten exposure on public-facing ports.",
        "Monitor backend saturation and autoscaling signals for spillover impact.",
    ],
    "DoS": [
        "Throttle repeated abusive flows and protect the targeted application endpoint.",
        "Inspect service-side resource contention and recover unhealthy instances.",
    ],
    "BruteForce": [
        "Block the attacking sources and enforce MFA or temporary account lockouts.",
        "Audit recent authentication activity for successful unauthorized access.",
    ],
    "WebAttack": [
        "Inspect the application and database for unauthorized modification attempts.",
        "Enable stricter WAF rules and rotate any exposed secrets or session tokens.",
    ],
    "Botnet": [
        "Isolate suspected hosts and block suspicious outbound command-and-control destinations.",
        "Run malware and persistence checks on impacted hosts.",
    ],
    "OtherAttack": [
        "Hold the related assets in heightened monitoring and review adjacent telemetry.",
    ],
    "Network_Connection_Error": [
        "Inspect DataNode network paths and packet loss between the impacted hosts.",
        "Check for NIC saturation, retransmissions, or upstream filtering during the attack window.",
    ],
    "Pipeline_Failure": [
        "Inspect HDFS write-pipeline health and restart unstable DataNode services if needed.",
        "Rebalance or reroute affected HDFS data flows away from unhealthy nodes.",
    ],
    "Cascading_Failure": [
        "Escalate immediately and review cluster-wide health to stop ongoing propagation.",
        "Stabilize networking and storage dependencies before replaying failed workloads.",
    ],
    "Node_Failure": [
        "Isolate the unhealthy DataNode and rebalance or re-replicate its blocks.",
    ],
    "Data_Corruption": [
        "Trigger block verification and force replication from known-good replicas.",
        "Freeze suspect downstream consumers until integrity checks complete.",
    ],
    "Replication_Failure": [
        "Force re-replication and inspect NameNode/DataNode coordination logs.",
    ],
    "Storage_Write_Failure": [
        "Inspect local disk health and write path errors on the impacted node.",
    ],
    "PacketResponder_Crash": [
        "Check responder thread failures and stabilize the associated DataNode process.",
    ],
    "Unknown_System_Anomaly": [
        "Retain the logs for deeper triage and monitor the same workload under tighter thresholds.",
    ],
}


def suggest_mitigations(labels: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for label in labels:
        for action in MITIGATION_MAP.get(str(label), []):
            if action in seen:
                continue
            seen.add(action)
            ordered.append(action)
    if not ordered:
        ordered.append("Continue monitoring while collecting additional correlated telemetry.")
    return ordered
