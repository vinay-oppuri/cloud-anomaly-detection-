from __future__ import annotations

from src.experts.network_expert.process import main as _preprocess_main


def main() -> None:
    """Network parser entrypoint aligned with system_expert/parser.py."""
    _preprocess_main()


if __name__ == "__main__":
    main()
