from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.cross_frequency import build_same_frequency_manifest_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a locked same-frequency zero-degree JSONL manifest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-manifest", help="build the same-frequency JSONL")
    build.add_argument("--dataset-root", required=True, type=Path)
    build.add_argument("--split-path", required=True, type=Path)
    build.add_argument("--array-size", required=True, choices=("8x8", "16x16", "32x32"))
    build.add_argument("--frequency-hz", required=True, type=int)
    build.add_argument("--steering-deg", required=True, type=float)
    build.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-manifest":
        summary = build_same_frequency_manifest_artifact(
            dataset_root=args.dataset_root,
            split_path=args.split_path,
            array_size=args.array_size,
            frequency_hz=args.frequency_hz,
            steering_deg=args.steering_deg,
            output_path=args.output,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
