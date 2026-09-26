"""Cross-fitted LSTM and diffusion detectors for unseen canary trajectories.

The learned models see compact per-round statistics derived from exactly the
same observations, checkpoint fingerprints, and public background estimates as
the analytic likelihood detector. Training uses calibration canaries only.
"""
import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import torch
from torch import nn
from diffusers import DDPMScheduler, DDIMScheduler

from .canary_audit import run_sequence, scores as analytic_scores
from .data import digest, load_mnist
from .metrics import auc, rates, threshold_at_fpr
from .models import initialize


FEATURE_NAMES = (
    "round_llr",
    "matched_z",
    "fingerprint_snr",
    "observation_rms",
    "cosine",
    "round_fraction",
    "log_sigma",
)


def write_csv(path, rows):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def trajectory_features(run, sigma, clip, batch_size):
    """Compact, dimensionless features available to every detector."""
    residual = run["fingerprints"] - run["backgrounds"]
    delta = run["observations"] - run["backgrounds"]
    noise_sd = sigma * clip / batch_size
    residual_norm = np.linalg.norm(residual, axis=1).clip(min=1e-12)
    delta_norm = np.linalg.norm(delta, axis=1).clip(min=1e-12)
    matched_z = np.sum(delta * residual, axis=1) / (noise_sd * residual_norm)
    fingerprint_snr = residual_norm / (batch_size * noise_sd)
    observation_rms = delta_norm / (np.sqrt(delta.shape[1]) * noise_sd)
    cosine = np.sum(delta * residual, axis=1) / (delta_norm * residual_norm)
    analytic = analytic_scores(run, sigma * clip, batch_size)
    rounds = np.arange(1, len(delta) + 1, dtype=np.float32) / len(delta)
    features = np.stack([
        analytic["round_llr"], matched_z, fingerprint_snr, observation_rms,
        cosine, rounds, np.full(len(delta), np.log(sigma), dtype=np.float32)
    ], axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite learned-detector feature")
    return features, analytic


def mixture_llr(round_llr, probability):
    """Log likelihood ratio for Bernoulli canary inclusion per round."""
    values = np.asarray(round_llr, dtype=np.float64)
    if not 0 < probability <= 1:
        raise ValueError("Inclusion probability must be in (0, 1]")
    return float(np.logaddexp(np.log1p(-probability), np.log(probability) + values).sum()) \
        if probability < 1 else float(values.sum())


class CanaryBiLSTM(nn.Module):
    def __init__(self, features, hidden=64):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(features, hidden), nn.Tanh())
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        encoded = self.lstm(self.embed(x))[0]
        pooled = torch.cat([encoded.mean(1), encoded.amax(1)], dim=1)
        return self.head(pooled).squeeze(1)


class InclusionDiffusion(nn.Module):
    """Conditional temporal diffusion model for the hidden inclusion schedule."""
    def __init__(self, features, steps, train_timesteps=100, hidden=64):
        super().__init__()
        self.input = nn.Linear(features + 1, hidden)
        self.time = nn.Embedding(train_timesteps, hidden)
        self.position = nn.Parameter(torch.randn(1, steps, hidden) * .01)
        layer = nn.TransformerEncoderLayer(
            hidden, 4, 2 * hidden, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output = nn.Linear(hidden, 1)

    def forward(self, noisy_schedule, timestep, context):
        hidden = self.input(torch.cat([noisy_schedule, context], dim=2))
        hidden = hidden + self.position + self.time(timestep)[:, None]
        return self.output(self.temporal(hidden))


def standardize(train, *others):
    mean = train.mean(axis=(0, 1), keepdims=True)
    scale = train.std(axis=(0, 1), keepdims=True).clip(min=1e-5)
    return (mean, scale, *((item - mean) / scale for item in (train,) + others))


@torch.no_grad()
def lstm_score(model, features):
    model.eval()
    return torch.sigmoid(model(torch.as_tensor(features, dtype=torch.float32))).numpy()


def fit_lstm(train_x, train_y, validation_x, validation_y, cfg, seed):
    torch.manual_seed(seed)
    model = CanaryBiLSTM(train_x.shape[-1], cfg["hidden_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lstm_learning_rate"], weight_decay=1e-4)
    x = torch.as_tensor(train_x, dtype=torch.float32)
    y = torch.as_tensor(train_y, dtype=torch.float32)
    validation_y_t = torch.as_tensor(validation_y, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed + 1)
    best_loss, best_state, logs = float("inf"), None, []
    for step in range(cfg["lstm_steps"]):
        model.train()
        index = torch.randint(len(x), (cfg["training_batch_size"],), generator=generator)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x[index]), y[index])
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if (step + 1) % cfg["validation_interval"] == 0:
            model.eval()
            with torch.no_grad():
                logits = model(torch.as_tensor(validation_x, dtype=torch.float32))
                validation_loss = float(nn.functional.binary_cross_entropy_with_logits(logits, validation_y_t))
                validation_auc = auc(torch.sigmoid(logits).numpy(), validation_y)
            if validation_loss < best_loss:
                best_loss, best_state = validation_loss, copy.deepcopy(model.state_dict())
            logs.append(dict(step=step + 1, training_loss=float(loss.detach()),
                             validation_loss=validation_loss, validation_auc=validation_auc))
    if best_state is None:
        raise RuntimeError("No LSTM checkpoint was selected")
    model.load_state_dict(best_state); model.eval()
    return model, logs


def diffusion_schedule(cfg):
    return DDPMScheduler(
        num_train_timesteps=cfg["diffusion_train_timesteps"],
        beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction",
        clip_sample=False)


@torch.no_grad()
def diffusion_reconstruct(model, schedule, features, cfg, seed, draws=None):
    model.eval()
    sampler = DDIMScheduler.from_config(schedule.config)
    sampler.set_timesteps(cfg["diffusion_sampling_steps"])
    context = torch.as_tensor(features, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    recovered = []
    for _ in range(draws or cfg["posterior_draws"]):
        sample = torch.randn((len(context), context.shape[1], 1), generator=generator)
        for timestep in sampler.timesteps:
            t = torch.full((len(context),), int(timestep), dtype=torch.long)
            prediction = model(sample, t, context)
            sample = sampler.step(prediction, int(timestep), sample, eta=0).prev_sample
        recovered.append(sample)
    return torch.stack(recovered)


def diffusion_score(model, schedule, features, cfg, seed, draws=None):
    reconstructed = diffusion_reconstruct(model, schedule, features, cfg, seed, draws)
    probabilities = ((reconstructed + 1) / 2).clamp(0, 1)
    return probabilities.mean(dim=(0, 2, 3)).numpy(), reconstructed.mean(0).squeeze(2).numpy()


def fit_diffusion(train_x, train_schedule, validation_x, validation_schedule,
                  validation_labels, cfg, seed):
    torch.manual_seed(seed)
    model = InclusionDiffusion(train_x.shape[-1], train_x.shape[1],
                               cfg["diffusion_train_timesteps"], cfg["hidden_size"])
    schedule = diffusion_schedule(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["diffusion_learning_rate"], weight_decay=1e-4)
    x = torch.as_tensor(train_x, dtype=torch.float32)
    target = torch.as_tensor(2 * train_schedule[..., None] - 1, dtype=torch.float32)
    validation_target = torch.as_tensor(2 * validation_schedule - 1, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed + 1)
    best_loss, best_state, logs = float("inf"), None, []
    for step in range(cfg["diffusion_steps"]):
        model.train()
        index = torch.randint(len(x), (cfg["training_batch_size"],), generator=generator)
        clean = target[index]
        noise = torch.randn(clean.shape, generator=generator)
        timestep = torch.randint(cfg["diffusion_train_timesteps"], (len(index),), generator=generator)
        noisy = schedule.add_noise(clean, noise, timestep)
        prediction = model(noisy, timestep, x[index])
        loss = (prediction - schedule.get_velocity(clean, noise, timestep)).square().mean()
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if (step + 1) % cfg["validation_interval"] == 0:
            reconstructed = diffusion_reconstruct(
                model, schedule, validation_x, cfg, seed + 9000, draws=2).mean(0).squeeze(2)
            validation_loss = float((reconstructed - validation_target).square().mean())
            validation_scores = ((reconstructed + 1) / 2).clamp(0, 1).mean(1).numpy()
            validation_auc = auc(validation_scores, validation_labels)
            if validation_loss < best_loss:
                best_loss, best_state = validation_loss, copy.deepcopy(model.state_dict())
            logs.append(dict(step=step + 1, training_loss=float(loss.detach()),
                             validation_schedule_mse=validation_loss,
                             validation_membership_auc=validation_auc))
    if best_state is None:
        raise RuntimeError("No diffusion checkpoint was selected")
    model.load_state_dict(best_state); model.eval()
    return model, schedule, logs


def build_dataset(source, data_path, output):
    cfg = json.loads((source / "config.json").read_text())
    split = json.loads((source / "splits.json").read_text())
    completion = json.loads((source / "completion.json").read_text())
    if completion["dataset_sha256"] != digest(data_path):
        raise ValueError("Dataset differs from the completed canary holdout run")
    x, labels = load_mnist(data_path)
    initial = initialize(cfg["seed"])
    initial.load_state_dict(torch.load(source / "public_initial_cnn.pt", weights_only=True))
    initial.eval()
    population = np.asarray(split["population"])
    background = np.asarray(split["background"])
    records, features, schedules = [], [], []
    split_canaries = {
        "calibration": split["calibration_canaries"],
        "holdout": split["test_canaries"],
    }
    for sigma in cfg["sigmas"]:
        for role, canaries in split_canaries.items():
            print(f"features sigma={sigma:g}, split={role}", flush=True)
            for canary in canaries:
                for label, probability in [(0, 0.0), (1, cfg["canary_probability"])]:
                    for sequence in range(cfg["sequences_per_canary"]):
                        seed = cfg["seed"] + int(sigma * 10000) + int(canary) + sequence
                        run = run_sequence(initial, x, labels, population, background,
                                           int(canary), cfg, sigma, seed, probability)
                        matrix, analytic = trajectory_features(
                            run, sigma, cfg["clip"], cfg["batch_size"])
                        features.append(matrix); schedules.append(run["included"].astype(np.float32))
                        records.append(dict(
                            sigma=float(sigma), split=role, canary=int(canary), label=label,
                            sequence=sequence, sequence_llr=analytic["sequence_llr"],
                            mixture_llr=mixture_llr(analytic["round_llr"], cfg["canary_probability"]),
                            last_llr=analytic["last_llr"], max_llr=analytic["max_llr"],
                            inclusion_rate=float(run["included"].mean())))
    arrays = dict(features=np.stack(features), schedules=np.stack(schedules),
                  labels=np.asarray([r["label"] for r in records], dtype=np.int64),
                  sigmas=np.asarray([r["sigma"] for r in records], dtype=np.float32),
                  canaries=np.asarray([r["canary"] for r in records], dtype=np.int64),
                  is_holdout=np.asarray([r["split"] == "holdout" for r in records]))
    np.savez_compressed(output / "compact_features.npz", **arrays)
    write_csv(output / "trajectory_index.csv", records)
    return cfg, records, arrays


def summarize(records, learned, output, cfg):
    methods = ["mixture_llr", "sequence_llr", "lstm", "diffusion", "last_llr", "max_llr"]
    rows, per_canary = [], []
    for sigma in cfg["sigmas"]:
        calibration = [i for i, r in enumerate(records) if r["sigma"] == sigma and r["split"] == "calibration"]
        holdout = [i for i, r in enumerate(records) if r["sigma"] == sigma and r["split"] == "holdout"]
        for method in methods:
            values = learned[method] if method in learned else np.asarray([r[method] for r in records])
            cal_score = values[calibration]; cal_label = np.asarray([records[i]["label"] for i in calibration])
            test_score = values[holdout]; test_label = np.asarray([records[i]["label"] for i in holdout])
            threshold_1 = threshold_at_fpr(cal_score, cal_label, .01)
            threshold_5 = threshold_at_fpr(cal_score, cal_label, .05)
            tpr1, fpr1 = rates(test_score, test_label, threshold_1)
            tpr5, fpr5 = rates(test_score, test_label, threshold_5)
            rows.append(dict(
                sigma=sigma, method=method, calibration_auc=auc(cal_score, cal_label),
                holdout_auc=auc(test_score, test_label),
                holdout_tpr_at_calibrated_1pct=tpr1, holdout_fpr_at_calibrated_1pct=fpr1,
                holdout_tpr_at_calibrated_5pct=tpr5, holdout_fpr_at_calibrated_5pct=fpr5,
                threshold_1pct=threshold_1, threshold_5pct=threshold_5))
            for canary in sorted(set(records[i]["canary"] for i in holdout)):
                selected = [i for i in holdout if records[i]["canary"] == canary]
                per_canary.append(dict(sigma=sigma, method=method, canary=canary,
                                       auc=auc(values[selected], [records[i]["label"] for i in selected])))
    write_csv(output / "summary.csv", rows); write_csv(output / "per_canary.csv", per_canary)
    return rows


def paired_auc_differences(records, learned, output, cfg, repetitions=2000):
    """Paired trajectory bootstrap for learned-minus-analytic AUC differences."""
    values = {
        "lstm": learned["lstm"],
        "diffusion": learned["diffusion"],
        "sequence_llr": np.asarray([r["sequence_llr"] for r in records]),
        "mixture_llr": np.asarray([r["mixture_llr"] for r in records]),
    }
    rng = np.random.default_rng(99173)
    rows = []
    for sigma in cfg["sigmas"]:
        selected = np.asarray([i for i, r in enumerate(records)
                               if r["sigma"] == sigma and r["split"] == "holdout"])
        labels = np.asarray([records[i]["label"] for i in selected])
        positive = selected[labels == 1]; negative = selected[labels == 0]
        for method in ["lstm", "diffusion"]:
            for comparator in ["sequence_llr", "mixture_llr"]:
                observed = auc(values[method][selected], labels) - auc(values[comparator][selected], labels)
                samples = []
                for _ in range(repetitions):
                    index = np.r_[rng.choice(positive, len(positive), replace=True),
                                  rng.choice(negative, len(negative), replace=True)]
                    boot_labels = np.r_[np.ones(len(positive), dtype=int),
                                        np.zeros(len(negative), dtype=int)]
                    samples.append(auc(values[method][index], boot_labels) -
                                   auc(values[comparator][index], boot_labels))
                low, high = np.quantile(samples, [.025, .975])
                rows.append(dict(sigma=sigma, method=method, comparator=comparator,
                                 auc_difference=observed, ci_low=float(low), ci_high=float(high),
                                 bootstrap_probability_positive=float(np.mean(np.asarray(samples) > 0)),
                                 repetitions=repetitions,
                                 unit="paired trajectories; stratified by membership"))
    write_csv(output / "paired_auc_differences.csv", rows)
    return rows


def plot_results(summary, output):
    labels = {"mixture_llr": "analytic mixture LLR", "sequence_llr": "sum LLR",
              "lstm": "cross-fitted BiLSTM", "diffusion": "schedule diffusion",
              "last_llr": "final release"}
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for method in labels:
        rows = [row for row in summary if row["method"] == method]
        ax.plot([row["sigma"] for row in rows], [row["holdout_auc"] for row in rows],
                marker="o", label=labels[method])
    ax.set_xscale("log"); ax.set_ylim(.45, 1.02)
    ax.set_xlabel("DP-SGD noise multiplier σ"); ax.set_ylabel("Unseen-canary holdout AUC")
    ax.set_title("Learned trajectory models versus analytic fingerprints")
    ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"learned_comparison.{extension}", dpi=220)
    plt.close(fig)


def plot_auc_differences(rows, output):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for offset, (method, label, color) in enumerate([
            ("lstm", "BiLSTM minus sum LLR", "#2ca02c"),
            ("diffusion", "diffusion minus sum LLR", "#d62728")]):
        selected = [row for row in rows if row["method"] == method and
                    row["comparator"] == "sequence_llr"]
        x = np.asarray([row["sigma"] for row in selected], dtype=float) * (1.02 ** (2 * offset - 1))
        center = np.asarray([row["auc_difference"] for row in selected], dtype=float)
        low = np.asarray([row["ci_low"] for row in selected], dtype=float)
        high = np.asarray([row["ci_high"] for row in selected], dtype=float)
        ax.errorbar(x, center, yerr=np.vstack([center - low, high - center]),
                    marker="o", capsize=4, linewidth=1.8, color=color, label=label)
    ax.axhline(0, color="black", linewidth=1, alpha=.7)
    ax.set_xscale("log"); ax.set_xticks([.25, 1, 4], labels=["0.25", "1", "4"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("DP-SGD noise multiplier σ")
    ax.set_ylabel("Paired holdout AUC difference")
    ax.set_title("Learned model gain over analytic trajectory score")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"auc_difference.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Cross-fitted learned canary detectors")
    parser.add_argument("--source", default="runs/canary_holdout")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/canary_learned.json")
    parser.add_argument("--output", default="runs/canary_learned")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        raise RuntimeError("Completed output exists; choose a new output directory")
    if not (source / "completion.json").exists():
        raise ValueError("Source canary holdout run is incomplete")
    torch.set_num_threads(1); started = time.time()
    learned_cfg = json.loads(Path(args.config).read_text())
    (output / "config.json").write_text(json.dumps(learned_cfg, indent=2))
    source_cfg, records, arrays = build_dataset(source, args.data, output)
    calibration_canaries = sorted(set(r["canary"] for r in records if r["split"] == "calibration"))
    holdout_index = np.asarray([i for i, r in enumerate(records) if r["split"] == "holdout"])
    oof_lstm = np.full(len(records), np.nan); oof_diffusion = np.full(len(records), np.nan)
    holdout_lstm, holdout_diffusion = [], []
    training_rows = []
    models = output / "models"; models.mkdir(exist_ok=True)
    for fold, validation_canary in enumerate(calibration_canaries):
        train_index = np.asarray([i for i, r in enumerate(records)
                                  if r["split"] == "calibration" and r["canary"] != validation_canary])
        validation_index = np.asarray([i for i, r in enumerate(records)
                                       if r["split"] == "calibration" and r["canary"] == validation_canary])
        mean, scale, train_x, validation_x, test_x = standardize(
            arrays["features"][train_index], arrays["features"][validation_index], arrays["features"][holdout_index])
        print(f"fold {fold + 1}/4: validation canary {validation_canary}, BiLSTM", flush=True)
        lstm, log = fit_lstm(train_x, arrays["labels"][train_index], validation_x,
                             arrays["labels"][validation_index], learned_cfg, learned_cfg["seed"] + fold)
        oof_lstm[validation_index] = lstm_score(lstm, validation_x)
        holdout_lstm.append(lstm_score(lstm, test_x))
        torch.save(lstm.state_dict(), models / f"fold{fold}_bilstm.pt")
        for row in log: training_rows.append(dict(fold=fold, validation_canary=validation_canary,
                                                   model="bilstm", **row))
        print(f"fold {fold + 1}/4: validation canary {validation_canary}, diffusion", flush=True)
        diffusion, schedule, log = fit_diffusion(
            train_x, arrays["schedules"][train_index], validation_x,
            arrays["schedules"][validation_index], arrays["labels"][validation_index],
            learned_cfg, learned_cfg["seed"] + 100 + fold)
        oof_diffusion[validation_index], _ = diffusion_score(
            diffusion, schedule, validation_x, learned_cfg, learned_cfg["seed"] + 500 + fold)
        test_score, _ = diffusion_score(
            diffusion, schedule, test_x, learned_cfg, learned_cfg["seed"] + 700 + fold)
        holdout_diffusion.append(test_score)
        torch.save(diffusion.state_dict(), models / f"fold{fold}_diffusion.pt")
        np.savez_compressed(models / f"fold{fold}_normalization.npz", mean=mean, scale=scale)
        schedule.save_pretrained(models / f"fold{fold}_scheduler")
        for row in log: training_rows.append(dict(fold=fold, validation_canary=validation_canary,
                                                   model="diffusion", **row))
    learned = {
        "lstm": oof_lstm.copy(),
        "diffusion": oof_diffusion.copy(),
    }
    learned["lstm"][holdout_index] = np.mean(holdout_lstm, axis=0)
    learned["diffusion"][holdout_index] = np.mean(holdout_diffusion, axis=0)
    if not np.isfinite(learned["lstm"]).all() or not np.isfinite(learned["diffusion"]).all():
        raise RuntimeError("Cross-fitted predictions are incomplete")
    write_csv(output / "training.csv", training_rows)
    predictions = [dict(index=i, split=r["split"], sigma=r["sigma"], canary=r["canary"],
                        label=r["label"], sequence=r["sequence"], lstm=learned["lstm"][i],
                        diffusion=learned["diffusion"][i], mixture_llr=r["mixture_llr"],
                        sequence_llr=r["sequence_llr"], last_llr=r["last_llr"])
                   for i, r in enumerate(records)]
    write_csv(output / "predictions.csv", predictions)
    summary = summarize(records, learned, output, source_cfg)
    differences = paired_auc_differences(records, learned, output, source_cfg)
    plot_results(summary, output); plot_auc_differences(differences, output)
    completion = dict(
        seconds=time.time() - started, dataset_sha256=digest(args.data),
        source_completion_sha256=hashlib.sha256((source / "completion.json").read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cross_fitting="four calibration canaries; leave one identity out per fold",
        holdout_canaries=sorted(set(r["canary"] for r in records if r["split"] == "holdout")),
        trajectories=len(records), feature_names=FEATURE_NAMES, torch=torch.__version__)
    (output / "completion.json").write_text(json.dumps(completion, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
