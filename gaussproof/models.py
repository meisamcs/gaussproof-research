"""Exact per-record gradients, global clipping, and Gaussian noisy SGD."""
import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.Tanh(), nn.AvgPool2d(2),
                                  nn.Conv2d(4, 8, 3, padding=1), nn.Tanh(), nn.AvgPool2d(2),
                                  nn.Flatten(), nn.Linear(8*7*7, 32), nn.Tanh())
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        return self.head(self.body(x))


def initialize(seed):
    torch.manual_seed(seed)
    return SmallCNN()


def per_record_gradients(model, x, y, microbatch=64):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    def loss(p, b, record, label):
        pred = functional_call(model, (p, b), (record.unsqueeze(0),))
        return nn.functional.cross_entropy(pred, label.unsqueeze(0))
    fn = vmap(grad(loss), in_dims=(None, None, 0, 0))
    chunks = []
    for start in range(0, len(x), microbatch):
        g = fn(params, buffers, x[start:start+microbatch], y[start:start+microbatch])
        chunks.append(torch.cat([g[k].reshape(len(g[k]), -1) for k in params], dim=1).detach())
    return torch.cat(chunks)


def clip_gradients(g, clip):
    return g * (clip/g.norm(dim=1).clamp_min(1e-12)).clamp(max=1)[:, None]


def projection_indices(model, dimensions, seed):
    offset, indices = 0, []
    for name, p in model.named_parameters():
        if name.startswith('head.'):
            indices.extend(range(offset, offset+p.numel()))
        offset += p.numel()
    if not 1 <= dimensions <= len(indices):
        raise ValueError(f'Projection dimensions must be in [1,{len(indices)}]')
    return np.sort(np.random.default_rng(seed).choice(indices, dimensions, replace=False))


def train(model, x, y, train_ids, steps, batch_size, clip, sigma, lr, seed,
          candidate_ids=None, coordinates=None, progress=None):
    if clip <= 0 or sigma < 0 or batch_size > len(train_ids):
        raise ValueError('Invalid noise, clipping or batch size')
    rng = np.random.default_rng(seed)
    noise_rng = torch.Generator().manual_seed(seed+1)
    dictionaries, releases, losses = [], [], []
    model.train()
    for step in range(steps):
        # Independent fixed-size samples without replacement at every step.
        ids = rng.choice(train_ids, batch_size, replace=False)
        if candidate_ids is not None:
            candidate_g = clip_gradients(per_record_gradients(model, x[candidate_ids], y[candidate_ids]), clip)
            dictionaries.append(candidate_g[:, coordinates].numpy())
        g = clip_gradients(per_record_gradients(model, x[ids], y[ids]), clip)
        noise = torch.randn(g.shape[1], generator=noise_rng)*(sigma*clip)
        released_sum = g.sum(0)+noise
        if candidate_ids is not None:
            releases.append(released_sum[coordinates].numpy())
        with torch.no_grad():
            losses.append(float(nn.functional.cross_entropy(model(x[ids]), y[ids])))
            start = 0
            for p in model.parameters():
                size = p.numel()
                p.add_(released_sum[start:start+size].reshape(p.shape), alpha=-lr/batch_size)
                start += size
        if progress and (step == 0 or (step+1) % 10 == 0 or step+1 == steps):
            progress(f'step {step+1}/{steps}, batch loss {losses[-1]:.4f}')
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite target parameters')
    return dict(dictionary=np.asarray(dictionaries), releases=np.asarray(releases), losses=np.asarray(losses))


def predict(model, x, ids):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[part]).softmax(1) for part in np.array_split(ids, max(1, int(np.ceil(len(ids)/256))))]).numpy()


def conservative_epsilon(steps, sigma, delta):
    # Replace-one sensitivity <=2C for the sum; no sampling amplification claimed.
    # Each Gaussian mechanism is rho=2/sigma² zCDP; compose over all steps.
    if sigma == 0:
        return None
    rho = 2*steps/sigma**2
    return float(rho+2*np.sqrt(rho*np.log(1/delta)))
