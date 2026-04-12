from __future__ import annotations

CANONICAL_CICIDS_15_CLASSES: tuple[str, ...] = (
    "Benign",
    "Botnet",
    "Brute Force-FTP",
    "Brute Force-SSH",
    "DDoS-HOIC",
    "DDoS-LOIC-HTTP",
    "DDoS-LOIC-UDP",
    "DoS-GoldenEye",
    "DoS-Hulk",
    "DoS-SlowHTTPTest",
    "DoS-Slowloris",
    "Infiltration",
    "Web Attack-Brute Force",
    "Web Attack-SQL Injection",
    "Web Attack-XSS",
)

ATTACK_FAMILY_CLASSES: tuple[str, ...] = (
    "Benign",
    "DDoS",
    "DoS",
    "BruteForce",
    "WebAttack",
    "Botnet",
    "Infiltration",
    "OtherAttack",
)

