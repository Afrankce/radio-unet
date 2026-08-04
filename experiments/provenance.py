from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


RADIOFLOW_ORIGIN_URL = "https://github.com/Hxxxz0/RadioFlow.git"
RADIOFLOW_UPSTREAM_BASE = "8944e3160f6a7a85b5451ae58e337186a4d98771"

DATASET_REPO_ID = "lxj321/Multi-config-Radiomap-Dataset"
DATASET_REPO_TYPE = "dataset"
DATASET_REVISION = "49ca1dcebe2caa2b2112e6c862132243a992b00a"
DATASET_FILENAME = "Dataset_20260306164917.zip"

REFERENCE_CODE_URL = "https://github.com/Lxj321/MulticonfigRadiomapDataset.git"
REFERENCE_CODE_REVISION = "f64e22a578933aa0ba57850ab2c7cf0695063c90"


@dataclass(frozen=True)
class RadioFlowCheckout:
    origin_url: str
    upstream_base: str
    head_commit: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=Path(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _normalized_origin(url: str) -> str:
    return url.strip().replace("\\", "/").rstrip("/")


def assert_radioflow_checkout(repo_root: Path) -> RadioFlowCheckout:
    repo_root = Path(repo_root).resolve()
    actual_origin = git_output(repo_root, "remote", "get-url", "origin")
    if _normalized_origin(actual_origin) != _normalized_origin(
        RADIOFLOW_ORIGIN_URL
    ):
        raise RuntimeError(
            "RadioFlow origin mismatch: "
            f"expected {RADIOFLOW_ORIGIN_URL}, got {actual_origin}"
        )

    try:
        subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                RADIOFLOW_UPSTREAM_BASE,
                "HEAD",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "checkout does not descend from the locked RadioFlow baseline "
            f"{RADIOFLOW_UPSTREAM_BASE}"
        ) from error

    return RadioFlowCheckout(
        origin_url=actual_origin,
        upstream_base=RADIOFLOW_UPSTREAM_BASE,
        head_commit=git_output(repo_root, "rev-parse", "HEAD"),
    )
