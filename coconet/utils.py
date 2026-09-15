"""Small shared helpers: YAML loading, seeding, logging, hashing, atomic writes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                                               datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def sha256_file(path, n_hex: int | None = None) -> str:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return h if n_hex is None else h[:n_hex]


def residual_sha(r_test: np.ndarray) -> str:
    """16-hex hash of the float64 test residuals; proves two runs saw the same predictor."""
    rt = np.ascontiguousarray(np.asarray(r_test, dtype=np.float64))
    return hashlib.sha256(rt.tobytes()).hexdigest()[:16]


def atomic_write_bytes(path: Path, write_fn) -> None:
    """Write through a temporary file and rename, so a killed job never leaves a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    write_fn(tmp)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, lambda p: p.write_text(json.dumps(obj, indent=1, default=str)))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def cuda_device_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
