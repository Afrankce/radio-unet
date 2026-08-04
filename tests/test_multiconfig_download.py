from __future__ import annotations

import base64
import io
import json
import stat
import zipfile
from collections import namedtuple
from pathlib import Path

import pytest


VALID_ZIP_BYTES = base64.b64decode(
    "UEsDBBQAAAAAAAAAIQBMgPkcCQAAAAkAAAAPAAAAZm9sZGVyL2RhdGEudHh0"
    "cmFkaW9mbG93UEsBAhQAFAAAAAAAAAAhAEyA+RwJAAAACQAAAA8AAAAAAAAA"
    "AAAAAIABAAAAAGZvbGRlci9kYXRhLnR4dFBLBQYAAAAAAQABAD0AAAA2AAAAAAA="
)
VALID_ZIP_SHA256 = (
    "033bc8183c7cca6337a3fa38fea27fff"
    "85a0ba73b095f83143285ce258091e52"
)


def _download_module():
    from experiments import multiconfig_download

    return multiconfig_download


def _write_zip(path: Path, member_name: str, payload: bytes = b"escape") -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member_name, payload)


class _MemoryResponse(io.BytesIO):
    def __init__(self, payload: bytes, status: int = 200) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}


class _InterruptedResponse:
    status = 200
    headers = {"Content-Length": str(len(VALID_ZIP_BYTES))}

    def __init__(self) -> None:
        self._reads = 0

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        return None

    def read(self, _size: int) -> bytes:
        self._reads += 1
        if self._reads == 1:
            return VALID_ZIP_BYTES[:50]
        raise TimeoutError("simulated stalled connection")


def test_official_source_is_pinned_to_dataset_revision() -> None:
    download = _download_module()

    assert download.OFFICIAL_SOURCE.repo_id == (
        "lxj321/Multi-config-Radiomap-Dataset"
    )
    assert download.OFFICIAL_SOURCE.revision == (
        "49ca1dcebe2caa2b2112e6c862132243a992b00a"
    )
    assert download.OFFICIAL_SOURCE.filename == "Dataset_20260306164917.zip"


def test_verify_zip_reports_literal_archive_metadata(tmp_path: Path) -> None:
    download = _download_module()
    archive = tmp_path / "fixture.zip"
    archive.write_bytes(VALID_ZIP_BYTES)

    verification = download.verify_zip(archive)

    assert verification.filename == "fixture.zip"
    assert verification.size_bytes == 137
    assert verification.sha256 == VALID_ZIP_SHA256
    assert verification.zip_members == 1
    assert verification.uncompressed_bytes == 9


def test_verify_zip_rejects_corrupt_bytes(tmp_path: Path) -> None:
    download = _download_module()
    archive = tmp_path / "corrupt.zip"
    archive.write_bytes(b"not a zip archive")

    with pytest.raises(download.ArchiveVerificationError, match="valid ZIP"):
        download.verify_zip(archive)


def test_verify_zip_rejects_wrong_expected_sha256(tmp_path: Path) -> None:
    download = _download_module()
    archive = tmp_path / "fixture.zip"
    archive.write_bytes(VALID_ZIP_BYTES)

    with pytest.raises(download.ArchiveHashMismatchError, match="SHA-256"):
        download.verify_zip(archive, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    "member_name",
    ["../escape.txt", "/absolute.txt", "C:/drive.txt", "..\\escape.txt"],
)
def test_safe_extract_rejects_paths_outside_destination(
    tmp_path: Path,
    member_name: str,
) -> None:
    download = _download_module()
    archive = tmp_path / "unsafe.zip"
    destination = tmp_path / "raw"
    _write_zip(archive, member_name)

    with pytest.raises(download.UnsafeArchiveMemberError):
        download.safe_extract_zip(archive, destination)

    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_allows_only_an_explicit_absolute_archive_root(
    tmp_path: Path,
) -> None:
    download = _download_module()
    archive = tmp_path / "official-layout.zip"
    destination = tmp_path / "raw"
    _write_zip(archive, "/Dataset/folder/data.txt", payload=b"radioflow")

    assert download.safe_extract_zip(
        archive,
        destination,
        allowed_absolute_root="Dataset",
    ) == destination
    assert (destination / "Dataset" / "folder" / "data.txt").read_bytes() == (
        b"radioflow"
    )
    receipt = json.loads(
        (tmp_path / "extraction_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["path_policy"] == {
        "allowed_absolute_root": "Dataset",
    }


def test_safe_extract_rejects_absolute_member_outside_explicit_root(
    tmp_path: Path,
) -> None:
    download = _download_module()
    archive = tmp_path / "wrong-root.zip"
    destination = tmp_path / "raw"
    _write_zip(archive, "/Elsewhere/data.txt")

    with pytest.raises(download.UnsafeArchiveMemberError):
        download.safe_extract_zip(
            archive,
            destination,
            allowed_absolute_root="Dataset",
        )

    assert not destination.exists()


def test_safe_extract_rejects_symbolic_link_members(tmp_path: Path) -> None:
    download = _download_module()
    archive = tmp_path / "symlink.zip"
    destination = tmp_path / "raw"
    with zipfile.ZipFile(archive, "w") as zip_file:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zip_file.writestr(info, "outside")

    with pytest.raises(download.UnsafeArchiveMemberError, match="symbolic"):
        download.safe_extract_zip(archive, destination)

    assert not destination.exists()


def test_safe_extract_rejects_insufficient_free_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = _download_module()
    archive = tmp_path / "fixture.zip"
    archive.write_bytes(VALID_ZIP_BYTES)
    DiskUsage = namedtuple("DiskUsage", "total used free")
    monkeypatch.setattr(
        download.shutil,
        "disk_usage",
        lambda _path: DiskUsage(total=100, used=100, free=0),
    )

    with pytest.raises(download.InsufficientDiskSpaceError, match="required"):
        download.safe_extract_zip(archive, tmp_path / "raw")


def test_safe_extract_is_idempotent_only_for_matching_inventory(
    tmp_path: Path,
) -> None:
    download = _download_module()
    archive = tmp_path / "fixture.zip"
    archive.write_bytes(VALID_ZIP_BYTES)
    destination = tmp_path / "raw"

    assert download.safe_extract_zip(archive, destination) == destination
    assert (destination / "folder" / "data.txt").read_bytes() == b"radioflow"
    receipt_path = tmp_path / "extraction_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["archive_filename"] == "fixture.zip"
    assert receipt["archive_sha256"] == VALID_ZIP_SHA256
    assert receipt["zip_members"] == 1
    assert receipt["uncompressed_bytes"] == 9
    assert len(receipt["inventory_sha256"]) == 64

    assert download.safe_extract_zip(archive, destination) == destination

    (destination / "unexpected.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(
        download.ExistingExtractionMismatchError,
        match="inventory",
    ):
        download.safe_extract_zip(archive, destination)


def test_download_uses_dataset_url_and_publishes_only_verified_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = _download_module()
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout=None):
        requested_urls.append(request.full_url)
        assert timeout == 60
        return _MemoryResponse(VALID_ZIP_BYTES)

    monkeypatch.setattr(download, "urlopen", fake_urlopen)
    downloads_dir = tmp_path / "downloads"

    archive, receipt = download.download_archive(
        download.OFFICIAL_SOURCE,
        downloads_dir,
    )

    assert archive == downloads_dir / "Dataset_20260306164917.zip"
    assert archive.read_bytes() == VALID_ZIP_BYTES
    assert not archive.with_suffix(".zip.part").exists()
    assert receipt["repo_type"] == "dataset"
    assert receipt["sha256"] == VALID_ZIP_SHA256
    assert "/datasets/lxj321/Multi-config-Radiomap-Dataset/resolve/" in (
        requested_urls[0]
    )
    assert download.OFFICIAL_SOURCE.revision in requested_urls[0]
    saved_receipt = json.loads(
        (tmp_path / "download_receipt.json").read_text(encoding="utf-8")
    )
    assert saved_receipt == receipt


def test_failed_download_does_not_replace_existing_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = _download_module()
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()
    destination = downloads_dir / download.OFFICIAL_SOURCE.filename
    destination.write_bytes(b"existing archive bytes")
    monkeypatch.setattr(
        download,
        "urlopen",
        lambda _request, timeout=None: _MemoryResponse(b"corrupt replacement"),
    )

    with pytest.raises(download.ArchiveVerificationError):
        download.download_archive(download.OFFICIAL_SOURCE, downloads_dir)

    assert destination.read_bytes() == b"existing archive bytes"
    assert not destination.with_suffix(".zip.part").exists()


def test_download_resumes_existing_part_with_http_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = _download_module()
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()
    temporary = downloads_dir / (download.OFFICIAL_SOURCE.filename + ".part")
    temporary.write_bytes(VALID_ZIP_BYTES[:50])
    observed_range: list[str | None] = []

    def fake_urlopen(request, timeout=None):
        assert timeout == 60
        observed_range.append(request.get_header("Range"))
        response = _MemoryResponse(VALID_ZIP_BYTES[50:], status=206)
        response.headers["Content-Range"] = "bytes 50-136/137"
        return response

    monkeypatch.setattr(download, "urlopen", fake_urlopen)

    archive, receipt = download.download_archive(
        download.OFFICIAL_SOURCE,
        downloads_dir,
    )

    assert observed_range == ["bytes=50-"]
    assert archive.read_bytes() == VALID_ZIP_BYTES
    assert receipt["sha256"] == VALID_ZIP_SHA256


def test_interrupted_download_keeps_part_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = _download_module()
    monkeypatch.setattr(
        download,
        "urlopen",
        lambda _request, timeout=None: _InterruptedResponse(),
    )

    with pytest.raises(download.DownloadInterruptedError, match="interrupted"):
        download.download_archive(
            download.OFFICIAL_SOURCE,
            tmp_path / "downloads",
        )

    temporary = (
        tmp_path
        / "downloads"
        / (download.OFFICIAL_SOURCE.filename + ".part")
    )
    assert temporary.read_bytes() == VALID_ZIP_BYTES[:50]


def test_prepare_cli_download_and_extract_use_pinned_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = _download_module()
    prepare = __import__("prepare_multiconfig")
    monkeypatch.setattr(
        download,
        "urlopen",
        lambda _request, timeout=None: _MemoryResponse(VALID_ZIP_BYTES),
    )

    assert prepare.main(["download", "--dataset-root", str(tmp_path)]) == 0
    assert prepare.main(["extract", "--dataset-root", str(tmp_path)]) == 0

    assert (tmp_path / "download_receipt.json").is_file()
    assert (tmp_path / "extraction_receipt.json").is_file()
    assert (tmp_path / "raw" / "folder" / "data.txt").read_bytes() == (
        b"radioflow"
    )
