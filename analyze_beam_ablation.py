from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Sequence

from data_loaders.same_frequency import (
    SameFrequencyRadiomapDataset,
    load_same_frequency_height_max,
)
from evaluation.beam_ablation_regions import (
    ARRAY_SIZES,
    BUILDING_RADIUS,
    MINIMUM_PRACTICAL_DB,
    classify_ablation,
    compare_array_predictions,
)
from experiments.multiconfig_manifest import (
    canonical_json_bytes,
    load_manifest_jsonl,
    load_schema_lock,
)


REPO_ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = REPO_ROOT / "experiments" / "multiconfig_schema.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Full and Beam-zero predictions on frozen building regions"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--full-results-root", type=Path, required=True)
    parser.add_argument("--beam-zero-results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--building-radius",
        type=int,
        choices=(BUILDING_RADIUS,),
        default=BUILDING_RADIUS,
        help="frozen five-pixel building-neighborhood radius",
    )
    return parser


def build_test_dataset(
    dataset_root: str | Path,
    manifest_dir: str | Path,
    array_size: str,
) -> SameFrequencyRadiomapDataset:
    if array_size not in ARRAY_SIZES:
        raise ValueError(f"unsupported array size: {array_size}")
    root = Path(dataset_root).resolve()
    manifests = Path(manifest_dir).resolve()
    manifest = manifests / f"manifest_samefreq_6.7ghz_{array_size}_0deg.jsonl"
    split_path = manifests / "scene_split_seed42.json"
    height_stats = manifests / "height_stats_train.json"
    records = load_manifest_jsonl(manifest)
    beam_ids = {record.beam_id for record in records if record.array_name == array_size}
    if len(beam_ids) != 1:
        raise ValueError(f"manifest must contain exactly one beam for {array_size}")
    schema = load_schema_lock(SCHEMA_PATH)
    source_metadata = schema.raw.get("source_metadata")
    if not isinstance(source_metadata, dict):
        raise ValueError("schema source_metadata must be an object")
    return SameFrequencyRadiomapDataset(
        dataset_root=root,
        manifest_path=manifest,
        split="test",
        array_size=array_size,
        height_max=load_same_frequency_height_max(
            height_stats,
            split_path=split_path,
        ),
        expected_frequency_hz=6_700_000_000,
        expected_beam_id=next(iter(beam_ids)),
        expected_counts={"train": 560, "val": 80, "test": 160},
        source_metadata=source_metadata,
        condition_variant="full",
    )


def _write_once(path: Path, payload: dict[str, object]) -> None:
    destination = path.resolve()
    if destination.exists():
        raise FileExistsError(f"comparison output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(canonical_json_bytes(payload))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    full_root = arguments.full_results_root.resolve()
    zero_root = arguments.beam_zero_results_root.resolve()
    rows = []
    for array_size in ARRAY_SIZES:
        dataset = build_test_dataset(
            arguments.dataset_root,
            arguments.manifest_dir,
            array_size,
        )
        rows.append(
            compare_array_predictions(
                array_size,
                dataset,
                full_root / array_size / "predictions",
                zero_root / array_size / "predictions",
                radius=arguments.building_radius,
            )
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": "beam_zero_shortcut_ablation_6.7ghz_0deg",
        "building_radius_pixels": BUILDING_RADIUS,
        "minimum_practical_difference_db": MINIMUM_PRACTICAL_DB,
        "arrays": rows,
        "decision": classify_ablation(rows),
        "decision_scope": "fixed_single_beam_array_specific_models",
    }
    _write_once(arguments.output, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "build_test_dataset", "main"]
