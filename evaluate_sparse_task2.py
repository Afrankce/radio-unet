from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch.utils.data import DataLoader

from data_loaders.sparse_task2 import sparse_task2_collate
from evaluation.sparse_task2_metrics import sparse_task2_metrics_for_json
from evaluation.sparse_task2_sampling import (
    make_task2_sample_noise,
    sparse_task2_euler_cfg_sample,
)
from experiments.sparse_task2_manifest import MANDATORY_SINGLEBEAM_PROTOCOL
from training.checkpointing import load_ema_for_evaluation
from training.config import InvocationControls
from training.model_factory import build_task2_sparse_radioflow
from training.sparse_task2_config import (
    SINGLEBEAM_TASK2_SAMPLE_COUNT,
    SparseTask2TrainConfig,
)
from training.sparse_task2_trainer import (
    build_sparse_task2_checkpoint_identity,
    build_sparse_task2_loaders,
    preflight_sparse_task2,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a completed single-beam Task 2 EMA checkpoint"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--height-stats-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--array-size", choices=("8x8", "16x16", "32x32"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--projected-consistency", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.steps <= 0:
        raise ValueError("steps must be positive")
    cfg = SparseTask2TrainConfig(
        dataset_root=arguments.dataset_root,
        manifest_path=arguments.manifest_path,
        height_stats_path=arguments.height_stats_path,
        run_root=arguments.run_root,
        array_size=arguments.array_size,
    )
    device = resolve_device(arguments.device)
    context = preflight_sparse_task2(cfg)
    model = build_task2_sparse_radioflow(
        condition_variant=cfg.condition_variant,
        model_size=cfg.model_size,
    ).to(device)
    identity = build_sparse_task2_checkpoint_identity(cfg, context, model)
    state = load_ema_for_evaluation(
        arguments.checkpoint,
        model=model,
        expected_identity=identity,
    )
    _, val_loader, _ = build_sparse_task2_loaders(cfg, context)
    loader = val_loader
    if arguments.split == "test":
        loader = DataLoader(
            context.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=sparse_task2_collate,
        )

    from evaluation.sparse_task2_metrics import SparseTask2MetricAccumulator

    model.eval()
    accumulator = SparseTask2MetricAccumulator()
    with torch.inference_mode():
        for batch in loader:
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            observation_mask = batch["observation_mask"].to(device)
            sparse_map = batch["sparse_map"].to(device)
            noise = torch.stack([
                make_task2_sample_noise(
                    protocol=MANDATORY_SINGLEBEAM_PROTOCOL,
                    array_size=cfg.array_size,
                    split=arguments.split,
                    sample_key=str(metadata["sample_key"]),
                    shape=tuple(target.shape[1:]),
                    base_seed=cfg.seed,
                )
                for metadata in batch["metadata"]
            ]).to(device)
            prediction = sparse_task2_euler_cfg_sample(
                model,
                condition,
                noise,
                cfg_scale=arguments.cfg_scale,
                steps=arguments.steps,
                observation_mask=observation_mask,
                sparse_map=sparse_map,
                projected_consistency=arguments.projected_consistency,
                use_amp=cfg.use_amp,
            )
            accumulator.update(
                prediction,
                target,
                valid_mask,
                observation_mask,
                batch["metadata"],
            )
    payload = {
        "schema_version": 1,
        "protocol": MANDATORY_SINGLEBEAM_PROTOCOL,
        "array_size": cfg.array_size,
        "split": arguments.split,
        "observation_count": SINGLEBEAM_TASK2_SAMPLE_COUNT,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "cfg_scale": float(arguments.cfg_scale),
        "steps": arguments.steps,
        "sampling_mode": (
            "projected_consistency"
            if arguments.projected_consistency
            else "source_equivalent"
        ),
        "metrics": sparse_task2_metrics_for_json(accumulator.compute()),
        "trainer_state": state.to_dict(),
    }
    arguments.results_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.results_path.write_bytes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2).encode("utf-8")
    )
    print(json.dumps({
        "status": "evaluation_complete",
        "results": str(arguments.results_path.resolve()),
        "split": arguments.split,
        "sampling_mode": payload["sampling_mode"],
        "overall": payload["metrics"]["overall"],
        "missing": payload["metrics"]["missing"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
