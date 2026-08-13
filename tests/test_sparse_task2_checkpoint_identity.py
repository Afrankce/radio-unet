from __future__ import annotations

from dataclasses import replace

import pytest
import torch


def _identity():
    from training.checkpointing import CheckpointIdentity

    return CheckpointIdentity(
        protocol="singlebeam_feature5_samples819",
        array_size="8x8",
        variant="feature5_mask",
        model_size="lite",
        condition_channels=5,
        parameter_count=3_996_011,
        config_sha256="1" * 64,
        manifest_sha256="2" * 64,
        split_sha256="3" * 64,
        mask_protocol_sha256="4" * 64,
        observation_count=819,
        split_type="scene_disjoint_single_beam",
    )


def test_task2_identity_has_distinct_key_schema_and_round_trips() -> None:
    from training.checkpointing import CheckpointIdentity

    identity = _identity()
    payload = identity.to_dict()
    assert set(payload) == set(CheckpointIdentity.TASK2_KEYS)
    assert CheckpointIdentity.from_dict(payload) == identity


@pytest.mark.parametrize(
    "field",
    ("protocol", "variant", "condition_channels", "observation_count", "split_type"),
)
def test_task2_identity_rejects_mutated_locked_fields(field: str) -> None:
    from training.checkpointing import CheckpointIdentityError

    identity = _identity()
    bad = {
        "protocol": "wrong",
        "variant": "feature4",
        "condition_channels": 4,
        "observation_count": 818,
        "split_type": "random_instance",
    }[field]
    with pytest.raises(CheckpointIdentityError):
        replace(identity, **{field: bad}).validate()


def test_task2_identity_is_not_old_sparse_identity() -> None:
    identity = _identity()
    assert identity._mode() == "task2"
    assert identity.experiment is None
    assert identity.protocol == "singlebeam_feature5_samples819"
