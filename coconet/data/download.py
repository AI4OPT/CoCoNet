"""Download and unzip helpers for the raw dataset files (idempotent: cached files are reused)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from coconet.utils import get_logger

logger = get_logger(__name__)

_TIMEOUT_SECONDS = 60
_CHUNK_BYTES = 1 << 16


class DownloadError(RuntimeError):
    pass


def download_file(url: str, dest_path: Path, force: bool = False) -> Path:
    """Download `url` to `dest_path` unless it already exists (written via a temp file)."""
    dest_path = Path(dest_path)
    if dest_path.exists() and not force:
        logger.info("Using cached file %s", dest_path)
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    try:
        logger.info("Downloading %s -> %s", url, dest_path)
        with requests.get(url, stream=True, timeout=_TIMEOUT_SECONDS) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=_CHUNK_BYTES):
                    if chunk:
                        f.write(chunk)
        tmp_path.rename(dest_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise DownloadError(f"Failed to download {url}: {exc}") from exc

    return dest_path


def download_and_extract_zip(url: str, cache_dir: Path, inner_file: str, force: bool = False) -> Path:
    """Download a zip to `cache_dir` and extract one member; returns the extracted file."""
    cache_dir = Path(cache_dir)
    zip_path = cache_dir / Path(url).name
    extracted_path = cache_dir / Path(inner_file).name

    if extracted_path.exists() and not force:
        logger.info("Using cached extracted file %s", extracted_path)
        return extracted_path

    download_file(url, zip_path, force=force)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            match = _find_zip_member(names, inner_file)
            if match is None:
                raise DownloadError(
                    f"Inner file '{inner_file}' not found in archive {url}. "
                    f"Available members: {names}"
                )
            with zf.open(match) as src, open(extracted_path, "wb") as dst:
                dst.write(src.read())
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"Corrupt zip archive at {zip_path}: {exc}") from exc

    return extracted_path


def _find_zip_member(names: list[str], inner_file: str) -> str | None:
    if inner_file in names:
        return inner_file
    target_name = Path(inner_file).name
    candidates = [n for n in names if Path(n).name == target_name]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        parent = Path(inner_file).parent.name
        narrowed = [c for c in candidates if parent and parent in c]
        if len(narrowed) == 1:
            return narrowed[0]
    return None
