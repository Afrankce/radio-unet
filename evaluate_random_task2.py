from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from data_loaders.random_task2 import random_task2_collate
from evaluation.random_task2_sampling import random_task2_euler_cfg_sample
from evaluation.sparse_task2_sampling import make_task2_sample_noise
from evaluation.sparse_task2_metrics import (
    SparseTask2MetricAccumulator,
    sparse_task2_metrics_for_json,
)
from training.multiconfig_trainer import resolve_device, seed_worker
from training.random_task2_config import RandomTask2TrainConfig
from training.random_task2_trainer import (
    RandomTask2TrainerError,
    build_random_task2_model,
)


_PINNED_FM_CFG_SCALE = 1.0
_PINNED_FM_EULER_STEPS = 2


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a random-instance Task 2 run")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--height-stats-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--array-size", choices=("8x8", "16x16", "32x32"), required=True)
    parser.add_argument("--variant", choices=("feature4", "feature5_mask"), default="feature4")
    parser.add_argument("--mode", choices=("regression", "pinned_fm"), default="regression")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", required=True)
    return parser


@torch.inference_mode()
def evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    cfg = RandomTask2TrainConfig(
        dataset_root=arguments.dataset_root,
        manifest_path=arguments.manifest_path,
        height_stats_path=arguments.height_stats_path,
        run_root=arguments.run_root,
        array_size=arguments.array_size,
        variant=arguments.variant,
        mode=arguments.mode,
    )
    device = resolve_device(arguments.device)
    checkpoint = Path(arguments.checkpoint) if arguments.checkpoint else cfg.run_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("schema_version") != 1:
        raise RandomTask2TrainerError("unsupported checkpoint schema")
    if payload["config"]["config_sha256"] != cfg.config_sha256:
        raise RandomTask2TrainerError("checkpoint config hash does not match requested config")
    model = build_random_task2_model(cfg).to(device)
    model.load_state_dict(payload["ema"])
    model.eval()
    from data_loaders.cross_frequency import load_cross_frequency_height_max
    from data_loaders.random_task2 import RandomTask2RadiomapDataset

    height_max = load_cross_frequency_height_max(cfg.height_stats_path)
    dataset = RandomTask2RadiomapDataset(
        dataset_root=cfg.dataset_root,
        manifest_path=cfg.manifest_path,
        split="test",
        array_size=cfg.array_size,
        height_max=height_max,
        variant=cfg.variant,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=random_task2_collate,
        worker_init_fn=seed_worker,
        pin_memory=False,
        persistent_workers=False,
    )
    accumulator = SparseTask2MetricAccumulator()
    per_scene: list[dict[str, Any]] = []
    started = time.time()
    for batch in loader:
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        observation_mask = batch["observation_mask"].to(device)
        sparse_map = batch["sparse_map"].to(device)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda" and cfg.use_amp,
        ):
            if cfg.mode == "regression":
                prediction = model(condition)
                prediction = prediction.float().clamp(0.0, 1.0)
                prediction = torch.where(observation_mask, sparse_map, prediction)
            else:
                noise = torch.stack(
                    [
                        make_task2_sample_noise(
                            protocol=cfg.canonical_payload()["protocol"],
                            array_size=cfg.array_size,
                            split="test",
                            sample_key=str(metadata["sample_key"]),
                            shape=tuple(target.shape[1:]),
                            base_seed=cfg.seed,
                            dtype=target.dtype,
                        )
                        for metadata in batch["metadata"]
                    ]
                ).to(device, dtype=target.dtype)
                prediction = random_task2_euler_cfg_sample(
                    model,
                    condition=condition,
                    x0=noise,
                    sparse_map=sparse_map,
                    observation_mask=observation_mask,
                    cfg_scale=_PINNED_FM_CFG_SCALE,
                    steps=_PINNED_FM_EULER_STEPS,
                    use_amp=cfg.use_amp,
                )
        accumulator.update(
            prediction,
            target,
            valid_mask,
            observation_mask,
            batch["metadata"],
        )
        missing = valid_mask & ~observation_mask
        error = prediction - target
        for sample_index, metadata in enumerate(batch["metadata"]):
            per_scene.append({
                "scene_id": str(metadata["scene_id"]),
                "sample_key": str(metadata["sample_key"]),
                "beam_id": int(metadata["beam_id"]),
                "steering_deg": float(metadata["steering_deg"]),
                "array_size": cfg.array_size,
                "missing_pixel_count": int(missing[sample_index].sum().item()),
                "missing_sq_sum": float(error[sample_index].square().masked_select(missing[sample_index]).sum().item()),
                "missing_abs_sum": float(error[sample_index].abs().masked_select(missing[sample_index]).sum().item()),
            })
    metrics = sparse_task2_metrics_for_json(accumulator.compute())
    output = {
        "schema_version": 1,
        "protocol": cfg.canonical_payload()["protocol"],
        "array_size": cfg.array_size,
        "variant": cfg.variant,
        "mode": cfg.mode,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_completed_epochs": payload["state"]["completed_epochs"],
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(cfg.run_dir / "test_metrics.json", output)
    _write_json(cfg.run_dir / "per_scene_metrics.json", per_scene)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output = evaluate(arguments)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
