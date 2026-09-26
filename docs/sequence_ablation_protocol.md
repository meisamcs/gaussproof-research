# Does the noisy sequence help beyond the current release?

This matched experiment reuses the completed `runs/sequence_images` trajectories,
public PCA representation, and frozen two-image generator. It generates no new
private releases and does not retrain the image generator.

## Fixed and varied information

Every model uses EXACTLY the same architecture: a 128-unit input embedding,
a bidirectional LSTM with 96 units per direction, and a linear gradient head.
A model is queried for a particular round t. Each query receives a 16-position
sequence; unavailable observations are zeroed and explicitly marked unavailable.
Relative round position is supplied to every model. All models get only the
**current checkpoint's** public fingerprint, identically repeated at each position.
Other checkpoint fingerprints are excluded, including from normalization.

Raw noisy gradients are projected into the existing public basis. All tokens are
centered against the current queried checkpoint's fingerprint, not their own
checkpoint fingerprint. This avoids giving future checkpoint information to the
current-only or causal models through preprocessing. The image generator always
receives the same current fingerprint and reconstructed gradient.

Four separately trained access settings:

1. `current`: only noisy release t is visible.
2. `past`: releases 1 through t are visible.
3. `full`: all 16 releases are visible in their original order.
4. `shuffled`: release t stays fixed; all OTHER releases from the same sequence
   are randomly permuted among the remaining positions during training and
   evaluation. The multiset of observations is preserved, but the correspondence
   between other observations and temporal positions is broken.

`full_test_shuffled` additionally evaluates the SAME full-trained model with the
other observations permuted. This is a distribution-shift diagnostic, not a
separately optimized attacker. The shuffled-trained model avoids conflating that
shift with the effect of ordering during learning. Both shuffled evaluations
average ERRORS over five fixed permutations, not predicted images.

A current fingerprint is itself computed at an evolving checkpoint and can
contain information from prior updates. The question is therefore the incremental
benefit of surrounding noisy releases GIVEN that identical public context.

## Matched training and selection

Two noise conditions (0.05, 0.5), C=1, B=2, 16-round sequences, 48 public training
sequences, 12 public calibration sequences, and the same eight private evaluation
sequences per condition. Three initialization seeds (101, 202, 303) per mode.
Within a seed, parameter initialization, 1,200 optimizer updates, 32 probes per
update, and sampled (public sequence, target round) training probes are identical
across modes. Shuffle randomness uses a separate stream. Every model is evaluated
on public calibration every 200 steps; lowest calibration gradient error selects
the checkpoint. All models use the same optimizer and hyperparameters. No private
image or gradient outcomes select models, epochs or settings.

Full gradient squared error / C² and permutation-matched two-image pixel MSE are
measured on the identical private batches. Checkpoint selection uses gradient
error, not private image error. Public representation and decoder limitations
from `sequence_images_protocol.md` still apply.

## Reporting

Report raw per-round measurements, per-(training seed, target run) averages,
per-training-seed means, and paired contrasts. Bootstrap intervals resample the
three training seeds and eight target runs as two crossed axes, preserving paired
comparisons; 2,000 replicates. These are descriptive pilot intervals. Runs reuse
a role-specific data pool, and the representation/decoder are fixed, so they are
not confidence intervals over independent training datasets or arbitrary models.

The primary contrast is full versus current. A trajectory claim additionally
requires a consistent benefit from order (shuffling worsens recovery). A shuffled
model matching full may imply the collection helps but not its order. Improvement
only on the shuffled-at-test diagnostic may reflect distribution shift.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib \
python -m gaussproof.sequence_ablation --source runs/sequence_images \
  --output runs/sequence_ablation --config configs/sequence_ablation.json
python scripts/summarize_sequence_ablation.py
```

Run the earlier sequence image experiment first if its trajectory files are absent.
Source completion metadata and hashes of the frozen representation and image
generator are recorded. All models and their selection histories remain saved
locally under `runs/sequence_ablation`.
