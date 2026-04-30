from __future__ import annotations

import sys

from src.validation.model_checks import _build_command


def test_build_command_uses_discovered_entrypoint(monkeypatch) -> None:
    monkeypatch.setattr("src.validation.model_checks.shutil.which", lambda name: f"C:/tools/{name}.exe")

    command = _build_command("test_cicids", "--log-file", "data/sample.log")

    assert command == ("C:/tools/test_cicids.exe", "--log-file", "data/sample.log")


def test_build_command_falls_back_to_module_execution(monkeypatch) -> None:
    monkeypatch.setattr("src.validation.model_checks.shutil.which", lambda name: None)

    command = _build_command("test_hdfs")

    assert command == (sys.executable, "-m", "src.experts.system_expert.test")
