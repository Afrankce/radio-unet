from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize sparse same-frequency results")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--results-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise RuntimeError("sparse summary requires completed evaluation result directories")


if __name__ == "__main__":
    raise SystemExit(main())
