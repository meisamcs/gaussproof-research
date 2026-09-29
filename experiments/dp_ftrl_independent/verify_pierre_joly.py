"""Compare our Pierre-Joly score variants with the pinned external code.

Download commit 9182ed809d9fa3d5141d50816b7e83a06590371b separately and
pass its extracted root with --source. This script uses synthetic models and
never runs the external repository's network-submission entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from endpoint_baselines import (offline_rmia_record_scores,
                                pierre_offline_lira_cdf)


EXPECTED_HASHES = {
    "attacks/offline_rmia.py":
        "1ba814f106273ee21fba2b982602f75e1c7cea827b5b93e6e2c41ce302f7afcb",
    "attacks/offline_lira.py":
        "4f0c58be04facda6b27c1f0a474af71fda69af045e80876cc04c40d74ddaae91",
}


class ToyData:
    def __init__(self, values, labels):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.labels = list(labels)
        self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index, self.values[index], self.labels[index], 0


def model(scale):
    instance = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        instance.weight.copy_(torch.tensor([[scale, 0.], [0., scale]]))
    return instance


def probabilities_and_logits(model_, data):
    with torch.no_grad():
        logits = model_(data.values)
        rows = torch.arange(len(data))
        labels = torch.tensor(data.labels)
        confidence = torch.softmax(logits, dim=1)[rows, labels]
        raw = logits[rows, labels]
    return confidence.numpy(), raw.numpy()


def run(source: Path):
    for name, expected in EXPECTED_HASHES.items():
        path = source / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Unexpected source contents: {path}")
    sys.path.insert(0, str(source))
    import attacks.offline_lira as lira
    import attacks.offline_rmia as rmia

    candidates = ToyData(
        [[2, 0], [0, 2], [2, 0], [0, 2], [1, -1], [-1, 1]],
        [0, 1, 1, 0, 0, 1])
    population = ToyData(
        [[2, 0], [0, 2], [2, 0], [0, 2], [1, -1], [-1, 1]],
        [1, 0, 0, 1, 1, 0])
    target = model(3.)
    references = [model(s) for s in (.5, .7, 1., 1.2)]

    def loader(data, batch_size, shuffle):
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=lambda batch: (
                              torch.stack([row[1] for row in batch]),
                              torch.tensor([row[2] for row in batch])))

    for module in (lira, rmia):
        module.get_out_dataset = lambda _: population
        module.get_off_shadow_models = lambda _, n: tuple(references[:n])
        module.get_data_loader = loader
        module.get_device = lambda: torch.device("cpu")
    rmia.transform_test = lambda: None

    external_rmia = rmia.OfflineRMIA(
        num_shadow_models=4, batch_size=2,
        reference_data="synthetic", a_param=.5, gamma=1.).run_attack(
            target, candidates)
    external_lira = lira.OfflineLiRA(
        num_shadow_models=4, batch_size=2,
        reference_data="synthetic").run_attack(target, candidates)

    x, xl = probabilities_and_logits(target, candidates)
    z, _ = probabilities_and_logits(target, population)
    xr, xrl = zip(*(probabilities_and_logits(m, candidates)
                    for m in references))
    zr, _ = zip(*(probabilities_and_logits(m, population)
                  for m in references))
    ours_rmia = offline_rmia_record_scores(
        x, np.stack(xr), z, np.stack(zr), a=.5, gamma=1.,
        population_correction=False)
    ours_lira = pierre_offline_lira_cdf(xl, np.stack(xrl))
    np.testing.assert_allclose(ours_rmia, external_rmia, atol=1e-6)
    np.testing.assert_allclose(ours_lira, external_lira, atol=1e-5)
    if len(np.unique(external_rmia)) < 2:
        raise AssertionError("Synthetic RMIA fixture failed to exercise rankings")
    print("Pinned Pierre-Joly RMIA/LiRA score parity passed")
    print("RMIA:", external_rmia.tolist())
    print("LiRA:", external_lira.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    run(parser.parse_args().source)
