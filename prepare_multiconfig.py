from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.multiconfig_download import (
    EXTRACTION_RECEIPT_NAME,
    OFFICIAL_SOURCE,
    download_archive,
    extract_official_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned Multi-config Radiomap dataset"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "extract"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--dataset-root",
            type=Path,
            required=True,
            help="Workspace root on the data volume",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    dataset_root: Path = arguments.dataset_root.resolve()
    if arguments.command == "download":
        archive, receipt = download_archive(
            OFFICIAL_SOURCE,
            dataset_root / "downloads",
        )
        print(json.dumps({"archive": str(archive), **receipt}, sort_keys=True))
        return 0
    if arguments.command == "extract":
        destination = extract_official_dataset(dataset_root)
        receipt = dataset_root / EXTRACTION_RECEIPT_NAME
        print(
            json.dumps(
                {"destination": str(destination), "receipt": str(receipt)},
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
