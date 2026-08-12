from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from training.sparse_config import FORMAL_RUN_VARIANT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the locked sparse 6.7GHz 5% single-beam RadioFlow run"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("select-cfg", "test"):
        child = subparsers.add_parser(command)
        child.add_argument("--dataset-root", type=Path, required=True)
        child.add_argument("--manifest-path", type=Path, required=True)
        child.add_argument("--height-stats-path", type=Path, required=True)
        child.add_argument("--run-root", type=Path, required=True)
        child.add_argument("--results-root", type=Path, required=True)
        child.add_argument("--array-size", choices=("8x8", "16x16", "32x32"), required=True)
        child.add_argument("--variant", choices=(FORMAL_RUN_VARIANT,), required=True)
        child.add_argument("--device", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    raise RuntimeError(
        f"sparse evaluation command {arguments.command!r} requires completed checkpoints and CFG selection"
    )


if __name__ == "__main__":
    raise SystemExit(main())
