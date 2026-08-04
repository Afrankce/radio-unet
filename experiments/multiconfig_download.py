from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import shutil
import stat
import ssl
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from huggingface_hub import hf_hub_url

from experiments.provenance import (
    DATASET_FILENAME,
    DATASET_REPO_ID,
    DATASET_REPO_TYPE,
    DATASET_REVISION,
    sha256_file,
)


DOWNLOAD_RECEIPT_NAME = "download_receipt.json"
EXTRACTION_RECEIPT_NAME = "extraction_receipt.json"


class ArchiveVerificationError(RuntimeError):
    """The archive is absent, unreadable, or internally corrupt."""


class ArchiveHashMismatchError(ArchiveVerificationError):
    """The archive digest does not match its locked receipt."""


class UnsafeArchiveMemberError(ArchiveVerificationError):
    """A ZIP member could escape or alter extraction semantics."""


class InsufficientDiskSpaceError(RuntimeError):
    """The destination volume cannot hold the extracted archive."""


class ExistingExtractionMismatchError(RuntimeError):
    """An existing extraction does not match its immutable receipt."""


class DownloadReceiptMismatchError(RuntimeError):
    """The local archive and download receipt disagree."""


class DownloadInterruptedError(RuntimeError):
    """A partial download can be resumed by invoking the command again."""


@dataclass(frozen=True)
class DatasetSource:
    repo_id: str
    revision: str
    filename: str


@dataclass(frozen=True)
class ArchiveVerification:
    filename: str
    size_bytes: int
    sha256: str
    zip_members: int
    uncompressed_bytes: int


@dataclass(frozen=True)
class InventorySummary:
    sha256: str
    files: int
    bytes: int


OFFICIAL_SOURCE = DatasetSource(
    repo_id=DATASET_REPO_ID,
    revision=DATASET_REVISION,
    filename=DATASET_FILENAME,
)
OFFICIAL_ABSOLUTE_ROOT = "Dataset"


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as destination:
            destination.write(_canonical_json_bytes(payload))
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DownloadReceiptMismatchError(
            f"cannot read receipt {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise DownloadReceiptMismatchError(f"receipt is not an object: {path}")
    return value


def verify_zip(
    archive_path: Path,
    expected_sha256: str | None = None,
) -> ArchiveVerification:
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise ArchiveVerificationError(f"archive is not a file: {archive_path}")

    archive_sha256 = sha256_file(archive_path)
    if expected_sha256 is not None and archive_sha256 != expected_sha256:
        raise ArchiveHashMismatchError(
            "archive SHA-256 mismatch: "
            f"expected {expected_sha256}, got {archive_sha256}"
        )

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise ArchiveVerificationError(
                    f"ZIP member failed CRC verification: {corrupt_member}"
                )
            members = archive.infolist()
    except ArchiveVerificationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ArchiveVerificationError(
            f"archive is not a valid ZIP: {archive_path}"
        ) from error

    return ArchiveVerification(
        filename=archive_path.name,
        size_bytes=archive_path.stat().st_size,
        sha256=archive_sha256,
        zip_members=len(members),
        uncompressed_bytes=sum(member.file_size for member in members),
    )


def _validate_allowed_absolute_root(value: str | None) -> str | None:
    if value is None:
        return None
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or value in (".", "..")
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or len(posix_path.parts) != 1
        or "\\" in value
    ):
        raise ValueError(
            "allowed_absolute_root must be one relative path component"
        )
    return value


def _safe_member_parts(
    member: zipfile.ZipInfo,
    allowed_absolute_root: str | None = None,
) -> tuple[str, ...]:
    name = member.filename
    unix_mode = member.external_attr >> 16
    if stat.S_ISLNK(unix_mode):
        raise UnsafeArchiveMemberError(
            f"symbolic link ZIP member is forbidden: {name}"
        )

    normalized = name.replace("\\", "/")
    if name.startswith("/"):
        allowed_prefix = (
            f"/{allowed_absolute_root}/"
            if allowed_absolute_root is not None
            else None
        )
        if allowed_prefix is None or not normalized.startswith(allowed_prefix):
            raise UnsafeArchiveMemberError(f"unsafe ZIP member path: {name}")
        normalized = normalized[1:]
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(normalized)
    if (
        not name
        or "\x00" in name
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
    ):
        raise UnsafeArchiveMemberError(f"unsafe ZIP member path: {name}")

    parts = tuple(part for part in posix_path.parts if part not in ("", "."))
    if not parts:
        raise UnsafeArchiveMemberError(f"empty ZIP member path: {name}")
    return parts


def _validate_archive_members(
    archive_path: Path,
    allowed_absolute_root: str | None,
) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            _safe_member_parts(member, allowed_absolute_root)


def _assert_free_space(destination_parent: Path, uncompressed_bytes: int) -> None:
    required = math.ceil(uncompressed_bytes * 1.10)
    free = shutil.disk_usage(destination_parent).free
    if free < required:
        raise InsufficientDiskSpaceError(
            f"required free bytes: {required}; available free bytes: {free}"
        )


def _inventory_summary(root: Path) -> InventorySummary:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in paths:
        if path.is_symlink():
            raise ExistingExtractionMismatchError(
                f"inventory contains a symbolic link: {path}"
            )
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_digest = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        byte_count += size
    return InventorySummary(
        sha256=digest.hexdigest(),
        files=file_count,
        bytes=byte_count,
    )


def _extraction_receipt_payload(
    archive: ArchiveVerification,
    destination: Path,
    inventory: InventorySummary,
    allowed_absolute_root: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "archive_filename": archive.filename,
        "archive_sha256": archive.sha256,
        "archive_size_bytes": archive.size_bytes,
        "zip_members": archive.zip_members,
        "uncompressed_bytes": archive.uncompressed_bytes,
        "destination": destination.name,
        "inventory_sha256": inventory.sha256,
        "inventory_files": inventory.files,
        "inventory_bytes": inventory.bytes,
        "path_policy": {
            "allowed_absolute_root": allowed_absolute_root,
        },
    }


def _validate_existing_extraction(
    destination: Path,
    archive: ArchiveVerification,
    allowed_absolute_root: str | None,
) -> None:
    receipt_path = destination.parent / EXTRACTION_RECEIPT_NAME
    if not destination.is_dir() or not receipt_path.is_file():
        raise ExistingExtractionMismatchError(
            "existing extraction has no matching receipt"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExistingExtractionMismatchError(
            f"cannot read extraction receipt: {error}"
        ) from error
    if not isinstance(receipt, dict):
        raise ExistingExtractionMismatchError("extraction receipt is not an object")

    expected_values = {
        "archive_sha256": archive.sha256,
        "archive_size_bytes": archive.size_bytes,
        "zip_members": archive.zip_members,
        "uncompressed_bytes": archive.uncompressed_bytes,
        "destination": destination.name,
        "path_policy": {
            "allowed_absolute_root": allowed_absolute_root,
        },
    }
    for key, expected in expected_values.items():
        if receipt.get(key) != expected:
            raise ExistingExtractionMismatchError(
                f"existing extraction receipt mismatch for {key}"
            )

    inventory = _inventory_summary(destination)
    if (
        receipt.get("inventory_sha256") != inventory.sha256
        or receipt.get("inventory_files") != inventory.files
        or receipt.get("inventory_bytes") != inventory.bytes
    ):
        raise ExistingExtractionMismatchError(
            "existing extraction inventory does not match its receipt"
        )


def _extract_to_temporary_directory(
    archive_path: Path,
    destination_parent: Path,
    destination_name: str,
    allowed_absolute_root: str | None,
) -> Path:
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_name}.extract-",
            dir=destination_parent,
        )
    )
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.infolist():
                parts = _safe_member_parts(member, allowed_absolute_root)
                target = temporary.joinpath(*parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
        return temporary
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
    *,
    allowed_absolute_root: str | None = None,
) -> Path:
    archive_path = Path(archive_path)
    destination = Path(destination)
    allowed_absolute_root = _validate_allowed_absolute_root(
        allowed_absolute_root
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    verification = verify_zip(archive_path)

    if destination.exists():
        _validate_existing_extraction(
            destination,
            verification,
            allowed_absolute_root,
        )
        return destination

    _validate_archive_members(archive_path, allowed_absolute_root)
    _assert_free_space(destination.parent, verification.uncompressed_bytes)
    temporary = _extract_to_temporary_directory(
        archive_path,
        destination.parent,
        destination.name,
        allowed_absolute_root,
    )
    try:
        inventory = _inventory_summary(temporary)
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    receipt = _extraction_receipt_payload(
        verification,
        destination,
        inventory,
        allowed_absolute_root,
    )
    _write_json_atomic(destination.parent / EXTRACTION_RECEIPT_NAME, receipt)
    return destination


def _stream_url_to_file(url: str, destination: Path) -> None:
    existing_bytes = destination.stat().st_size if destination.is_file() else 0
    headers = {"User-Agent": "RadioFlow-MultiConfig/1.0"}
    if existing_bytes:
        headers["Range"] = f"bytes={existing_bytes}-"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            status = int(getattr(response, "status", 200))
            if existing_bytes and status == 206:
                content_range = response.headers.get("Content-Range", "")
                expected_prefix = f"bytes {existing_bytes}-"
                if not content_range.startswith(expected_prefix):
                    raise DownloadInterruptedError(
                        "download interrupted: server returned an invalid "
                        f"Content-Range {content_range!r}"
                    )
                mode = "ab"
            elif status == 200:
                mode = "wb"
            else:
                raise DownloadInterruptedError(
                    f"download interrupted: unexpected HTTP status {status}"
                )

            with destination.open(mode) as output:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
    except DownloadInterruptedError:
        raise
    except (
        TimeoutError,
        ConnectionError,
        URLError,
        ssl.SSLError,
        http.client.IncompleteRead,
    ) as error:
        raise DownloadInterruptedError(
            f"download interrupted; rerun the command to resume: {error}"
        ) from error


def _download_receipt_payload(
    source: DatasetSource,
    verification: ArchiveVerification,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "repo_id": source.repo_id,
        "repo_type": DATASET_REPO_TYPE,
        "revision": source.revision,
        **asdict(verification),
    }
    payload["filename"] = source.filename
    return payload


def _try_reuse_download(
    source: DatasetSource,
    destination: Path,
    receipt_path: Path,
) -> tuple[Path, dict[str, Any]] | None:
    if not destination.is_file() or not receipt_path.is_file():
        return None
    try:
        receipt = _read_json_object(receipt_path)
        if (
            receipt.get("repo_id") != source.repo_id
            or receipt.get("repo_type") != DATASET_REPO_TYPE
            or receipt.get("revision") != source.revision
            or receipt.get("filename") != source.filename
        ):
            return None
        verification = verify_zip(
            destination,
            expected_sha256=str(receipt.get("sha256", "")),
        )
        if _download_receipt_payload(source, verification) != receipt:
            return None
    except (ArchiveVerificationError, DownloadReceiptMismatchError):
        return None
    return destination, receipt


def download_archive(
    source: DatasetSource,
    downloads_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    downloads_dir = Path(downloads_dir)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    destination = downloads_dir / source.filename
    receipt_path = downloads_dir.parent / DOWNLOAD_RECEIPT_NAME

    reused = _try_reuse_download(source, destination, receipt_path)
    if reused is not None:
        return reused

    temporary = destination.with_suffix(destination.suffix + ".part")
    url = hf_hub_url(
        repo_id=source.repo_id,
        repo_type=DATASET_REPO_TYPE,
        filename=source.filename,
        revision=source.revision,
    )
    _stream_url_to_file(url, temporary)
    try:
        verification = verify_zip(temporary)
    except ArchiveVerificationError:
        if temporary.exists():
            temporary.unlink()
        raise
    receipt = _download_receipt_payload(source, verification)
    os.replace(temporary, destination)
    _write_json_atomic(receipt_path, receipt)
    return destination, receipt


def validate_download_receipt(dataset_root: Path) -> tuple[Path, dict[str, Any]]:
    dataset_root = Path(dataset_root)
    archive = dataset_root / "downloads" / OFFICIAL_SOURCE.filename
    receipt_path = dataset_root / DOWNLOAD_RECEIPT_NAME
    if not receipt_path.is_file():
        raise DownloadReceiptMismatchError(
            f"download receipt is missing: {receipt_path}"
        )
    receipt = _read_json_object(receipt_path)
    required_identity = {
        "repo_id": OFFICIAL_SOURCE.repo_id,
        "repo_type": DATASET_REPO_TYPE,
        "revision": OFFICIAL_SOURCE.revision,
        "filename": OFFICIAL_SOURCE.filename,
    }
    for key, expected in required_identity.items():
        if receipt.get(key) != expected:
            raise DownloadReceiptMismatchError(
                f"download receipt mismatch for {key}"
            )
    verification = verify_zip(
        archive,
        expected_sha256=str(receipt.get("sha256", "")),
    )
    if _download_receipt_payload(OFFICIAL_SOURCE, verification) != receipt:
        raise DownloadReceiptMismatchError(
            "download receipt metadata does not match the archive"
        )
    return archive, receipt


def extract_official_dataset(dataset_root: Path) -> Path:
    dataset_root = Path(dataset_root)
    archive, _receipt = validate_download_receipt(dataset_root)
    return safe_extract_zip(
        archive,
        dataset_root / "raw",
        allowed_absolute_root=OFFICIAL_ABSOLUTE_ROOT,
    )
