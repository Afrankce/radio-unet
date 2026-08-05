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
from experiments.multiconfig_manifest import (
    freeze_schema_lock,
    verify_schema_lock,
    write_audit_report,
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
    audit = subparsers.add_parser("audit-schema")
    audit.add_argument("--dataset-root", type=Path, required=True)
    audit.add_argument("--report", type=Path, required=True)
    freeze = subparsers.add_parser("freeze-schema")
    freeze.add_argument("--dataset-root", type=Path, required=True)
    freeze.add_argument("--audit-report", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-schema")
    verify.add_argument("--dataset-root", type=Path, required=True)
    verify.add_argument("--schema", type=Path, required=True)
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
    if arguments.command == "audit-schema":
        report = write_audit_report(dataset_root, arguments.report)
        print(
            json.dumps(
                {
                    "report": str(arguments.report.resolve()),
                    "text_files": len(report.text_files),
                    "configurations": len(report.configurations),
                    "reference_scripts": len(report.reference_scripts),
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "freeze-schema":
        lock = freeze_schema_lock(
            dataset_root,
            arguments.audit_report,
            arguments.output,
            progress=True,
        )
        print(
            json.dumps(
                {
                    "schema": str(arguments.output.resolve()),
                    "arrays": len(lock.arrays),
                    "configurations": len(lock.configurations),
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "verify-schema":
        summary = verify_schema_lock(
            dataset_root,
            arguments.schema,
            progress=True,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
