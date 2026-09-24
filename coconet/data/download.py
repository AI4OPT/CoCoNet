"""Download and unzip helpers; existing files are reused."""

import zipfile
from pathlib import Path

import requests


def download_file(url, dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def zip_member(zip_path, inner_name):
    """Bytes of the member of `zip_path` whose file name is `inner_name`."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX")]
        match = [n for n in names if n == inner_name or Path(n).name == inner_name]
        if not match:
            raise FileNotFoundError(f"{inner_name} not in {zip_path}: {names}")
        return zf.read(match[0])


def download_and_extract(url, cache_dir, inner_name, zip_name=None):
    """Download the zip at `url` into `cache_dir` and extract one member next to it."""
    cache_dir = Path(cache_dir)
    extracted = cache_dir / Path(inner_name).name
    if extracted.exists():
        return extracted
    zip_path = download_file(url, cache_dir / (zip_name or Path(url).name))
    extracted.write_bytes(zip_member(zip_path, inner_name))
    return extracted
