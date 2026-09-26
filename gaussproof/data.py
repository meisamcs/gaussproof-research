"""Strict MNIST ingestion and persisted, disjoint sample roles."""
from pathlib import Path
import hashlib
import numpy as np
import torch


def load_mnist(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MNIST CSV missing: {path}. Supply --data /path/to/mnist_train.csv")
    with path.open() as f:
        first = f.readline().strip().split(',')
    try:
        [float(v) for v in first]
        skip = 0
    except ValueError:
        skip = 1
    a = np.loadtxt(path, delimiter=',', skiprows=skip, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 785 or not np.isfinite(a).all():
        raise ValueError("Expected finite label + 784 pixel columns")
    labels, pixels = a[:, 0], a[:, 1:]
    if not np.all((labels == np.floor(labels)) & (labels >= 0) & (labels <= 9)):
        raise ValueError("Labels must be integer digits 0..9")
    if pixels.min() < 0 or pixels.max() > 255:
        raise ValueError("Pixels must be in [0,255]")
    if pixels.max() <= 1:
        raise ValueError("Expected original 0..255 MNIST pixels, not pre-normalized data")
    return torch.from_numpy(pixels.reshape(-1, 1, 28, 28) / 255), torch.from_numpy(labels.astype(np.int64))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def make_splits(n, cfg, seed):
    sizes = {k: cfg[k] for k in ['train_size', 'eval_size', 'calibration_size', 'shadow_pool_size', 'population_size']}
    if cfg['eval_size'] + cfg['calibration_size'] > cfg['train_size']:
        raise ValueError("Evaluation and calibration members must fit inside target training set")
    if cfg['shadow_pool_size'] < 2 * cfg['train_size']:
        raise ValueError("Shadow pool must hold disjoint shadow train and nonmember sets")
    if sum(sizes.values()) > n:
        raise ValueError(f"Need {sum(sizes.values())} rows; dataset has {n}")
    perm = np.random.default_rng(seed).permutation(n)
    result, cursor = {}, 0
    for key, size in sizes.items():
        result[key] = perm[cursor:cursor + size]
        cursor += size
    train = result.pop('train_size')
    return dict(train=train, eval_member=train[:cfg['eval_size']],
                cal_member=train[cfg['eval_size']:cfg['eval_size'] + cfg['calibration_size']],
                eval_nonmember=result['eval_size'], cal_nonmember=result['calibration_size'],
                shadow_pool=result['shadow_pool_size'], population=result['population_size'])
