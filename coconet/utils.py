import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residual_sha(r):
    """Short hash of the float64 residual matrix; equal hashes mean the same predictor and data."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(r, dtype=np.float64)).tobytes()).hexdigest()[:16]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())
