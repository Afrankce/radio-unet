from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evaluation.same_frequency_evaluator import (
    SameFrequencyEvaluationError,
    run_test_evaluation,
)
from training.same_frequency_config import SameFrequencyTrainConfig


def test_test_evaluation_rejects_existing_result_directory(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    cfg = SameFrequencyTrainConfig(
        dataset_root=tmp_path / "dataset",
        manifest_path=tmp_path / "manifest.jsonl",
        height_stats_path=tmp_path / "height_stats.json",
        run_root=tmp_path / "run",
        array_size="8x8",
        beam_id=4,
    )

    with pytest.raises(SameFrequencyEvaluationError, match="already exists"):
        run_test_evaluation(cfg, torch.device("cpu"), results_root)
