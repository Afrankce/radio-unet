from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from evaluation.sparse_consistent_sampling import (
    make_sparse_consistent_sample_noise,
    sparse_consistent_euler_cfg_sample,
)
from evaluation.sparse_task2_metrics import sparse_task2_metrics_for_json
from data_loaders.sparse_consistent import sparse_consistent_collate
from training.checkpointing import load_ema_for_evaluation
from training.sparse_consistent_config import SPARSE_CONSISTENT_ARMS, SparseConsistentTrainConfig
from training.sparse_consistent_trainer import (
    build_sparse_consistent_checkpoint_identity,
    preflight_sparse_consistent,
    resolve_device,
)
from training.sparse_consistent_model import build_sparse_consistent_model
from training.multiconfig_trainer import seed_worker


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one sparse-consistent A/B/C/D arm")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--height-stats-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--array-size", choices=("8x8", "16x16", "32x32"), required=True)
    parser.add_argument("--arm", choices=SPARSE_CONSISTENT_ARMS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--case-index", type=int)
    return parser


@torch.inference_mode()
def evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    cfg = SparseConsistentTrainConfig(
        dataset_root=arguments.dataset_root,
        manifest_path=arguments.manifest_path,
        height_stats_path=arguments.height_stats_path,
        run_root=arguments.run_root,
        array_size=arguments.array_size,
        arm=arguments.arm,
    )
    device = resolve_device(arguments.device)
    context = preflight_sparse_consistent(cfg)
    model = build_sparse_consistent_model(cfg.arm).to(device)
    identity = build_sparse_consistent_checkpoint_identity(cfg, context, model)
    checkpoint = Path(arguments.checkpoint) if arguments.checkpoint else cfg.run_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    state = load_ema_for_evaluation(
        checkpoint,
        model=model,
        expected_identity=identity,
    )
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        context.test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=sparse_consistent_collate,
        worker_init_fn=seed_worker,
        pin_memory=False,
        persistent_workers=False,
    )
    from evaluation.sparse_task2_metrics import SparseTask2MetricAccumulator

    accumulator = SparseTask2MetricAccumulator()
    per_scene: list[dict[str, Any]] = []
    case_payload: dict[str, Any] | None = None
    started = time.time()
    for index, batch in enumerate(test_loader):
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        observation_mask = batch["observation_mask"].to(device)
        sparse_map = batch["sparse_map"].to(device)
        noise = torch.stack([
            make_sparse_consistent_sample_noise(
                array_size=cfg.array_size,
                split="test",
                sample_key=str(metadata["sample_key"]),
                shape=tuple(target.shape[1:]),
                base_seed=cfg.seed,
            )
            for metadata in batch["metadata"]
        ]).to(device)
        prediction = sparse_consistent_euler_cfg_sample(
            model,
            arm=cfg.arm,
            condition=condition,
            x0=noise,
            sparse_map=sparse_map,
            observation_mask=observation_mask,
            cfg_scale=cfg.cfg_scale,
            steps=cfg.euler_steps,
            use_amp=cfg.use_amp,
        )
        accumulator.update(
            prediction,
            target,
            valid_mask,
            observation_mask,
            batch["metadata"],
        )
        clipped = prediction.float().clamp(0.0, 1.0)
        for sample_index, metadata in enumerate(batch["metadata"]):
            missing = valid_mask[sample_index] & ~observation_mask[sample_index]
            observed = observation_mask[sample_index]
            missing_error = clipped[sample_index] - target[sample_index]
            per_scene.append({
                "scene_id": str(metadata["scene_id"]),
                "sample_key": str(metadata["sample_key"]),
                "array_size": cfg.array_size,
                "missing_pixel_count": int(missing.sum().item()),
                "missing_sq_sum": float(missing_error.square().masked_select(missing).sum().item()),
                "missing_abs_sum": float(missing_error.abs().masked_select(missing).sum().item()),
                "observed_pixel_count": int(observed.sum().item()),
                "observed_max_abs": float(missing_error.abs().masked_select(observed).max().item()),
                "observed_mean_abs": float(missing_error.abs().masked_select(observed).mean().item()),
            })
        if arguments.case_index is not None and index == arguments.case_index:
            case_payload = {
                "condition": condition[0].detach().cpu().numpy().tolist(),
                "sparse_map": sparse_map[0].detach().cpu().numpy().tolist(),
                "observation_mask": observation_mask[0].detach().cpu().numpy().astype("uint8").tolist(),
                "valid_mask": valid_mask[0].detach().cpu().numpy().astype("uint8").tolist(),
                "target": target[0].detach().cpu().numpy().tolist(),
                "prediction": clipped[0].detach().cpu().numpy().tolist(),
                "metadata": dict(batch["metadata"][0]),
            }
    metrics = sparse_task2_metrics_for_json(accumulator.compute())
    output = {
        "schema_version": 1,
        "protocol": cfg.canonical_payload()["protocol"],
        "array_size": cfg.array_size,
        "arm": cfg.arm,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_optimizer_step": state.optimizer_step,
        "checkpoint_completed_epochs": state.completed_epochs,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(cfg.run_dir / "test_metrics.json", output)
    _write_json(cfg.run_dir / "per_scene_metrics.json", per_scene)
    if case_payload is not None:
        _write_json(cfg.run_dir / "visualization_case.json", case_payload)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output = evaluate(arguments)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
