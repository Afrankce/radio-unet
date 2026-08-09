from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.cross_frequency import (
    build_cross_frequency_records,
    cross_frequency_spec,
    select_zero_degree_configurations,
    validate_cross_frequency_records,
)
from experiments.multiconfig_manifest import (
    SceneSplit,
    inventory_samples,
    load_manifest_jsonl,
    load_schema_lock,
    write_manifest_jsonl,
)
from experiments.provenance import sha256_file


DEFAULT_MANIFEST_NAME = "manifest_cross_frequency_8x8.jsonl"


def build_cross_frequency_manifest_artifact(
    *,
    dataset_root: Path,
    schema_path: Path,
    manifest_dir: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    schema_path = Path(schema_path).resolve()
    manifest_dir = Path(manifest_dir).resolve()
    output = Path(output_path).resolve() if output_path is not None else manifest_dir / DEFAULT_MANIFEST_NAME
    schema = load_schema_lock(schema_path)
    inventory = inventory_samples(dataset_root, schema)
    split_path = manifest_dir / "scene_split_seed42.json"
    try:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read fixed scene split {split_path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("fixed scene split must be a JSON object")
    split = SceneSplit.from_dict(payload)
    spec = cross_frequency_spec()
    selected = select_zero_degree_configurations(schema)
    records = build_cross_frequency_records(inventory, split, selected, spec)
    validate_cross_frequency_records(
        records,
        split,
        selected,
        spec,
        inventory=inventory,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_jsonl(output, records)
    written = load_manifest_jsonl(output)
    validate_cross_frequency_records(
        written,
        split,
        selected,
        spec,
        inventory=inventory,
    )
    return {
        "schema_version": 1,
        "manifest": str(output),
        "manifest_sha256": sha256_file(output),
        "scene_split": str(split_path),
        "scene_split_sha256": sha256_file(split_path),
        "dataset_revision": schema.identities.get("dataset_revision"),
        "records": len(written),
        "split_counts": {
            name: sum(record.split == name for record in written)
            for name in ("train", "val", "test")
        },
        "selected": {
            str(frequency): {
                "config_id": selection.config_id,
                "beam_id": selection.beam_id,
                "steering_deg": selection.steering_deg,
            }
            for frequency, selection in sorted(selected.items())
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the locked 8x8 4.9GHz-to-6.7GHz manifest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-manifest", help="build the cross-frequency JSONL")
    build.add_argument("--dataset-root", required=True, type=Path)
    build.add_argument("--schema", required=True, type=Path)
    build.add_argument("--manifest-dir", required=True, type=Path)
    build.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-manifest":
        summary = build_cross_frequency_manifest_artifact(
            dataset_root=args.dataset_root,
            schema_path=args.schema,
            manifest_dir=args.manifest_dir,
            output_path=args.output,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
