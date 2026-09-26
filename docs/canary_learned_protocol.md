# Learned trajectory models on unseen canaries

This experiment asks whether a learned sequence model can recover persistent
canary evidence better than the analytic matched-fingerprint statistic. It uses
the exact 576 trajectories and disjoint calibration/holdout identities from the
unseen-canary audit.

## Access and features

Every method has the same white-box audit inputs at each round: the noisy
released gradient, the candidate canary's clipped checkpoint gradient, and a
public background-gradient estimate. Seven dimensionless per-round features are
derived from those inputs: matched log likelihood, standardized matched
projection, fingerprint signal-to-noise ratio, observation RMS, cosine
alignment, round position, and log noise multiplier. Learned models never read
the private batch indices, digit labels, or holdout membership labels.

The analytic controls are:

- `sequence_llr`: the sum of the per-round Gaussian matched scores;
- `mixture_llr`: the per-round Bernoulli-mixture likelihood for a canary that is
  inserted independently with probability 0.5;
- `last_llr`: one final release only.

The BiLSTM maps the full feature sequence to a membership score. The temporal
diffusion model is trained to reconstruct the hidden 16-round inclusion schedule
from the observed feature sequence. It uses an upstream Diffusers DDPM training
schedule, velocity prediction, and deterministic DDIM sampling; its membership
score is the mean reconstructed inclusion probability.

## Identity cross-fitting

The four calibration canaries are cross-fitted by identity. Each fold trains on
three canaries and selects its checkpoint using the fourth. That fold produces
out-of-fold calibration scores. All four fold models then score each of the four
untouched holdout canaries, and their scores are averaged. The holdout labels do
not affect training, checkpoint selection, normalization, or thresholds.

The models train jointly across `sigma ∈ {0.25, 1, 4}`. The BiLSTM receives
1,000 optimization steps per fold. The diffusion model receives 1,600 steps per
fold and uses eight posterior draws with 30 DDIM steps for final scoring.

## Results

| sigma | sum LLR | mixture LLR | BiLSTM | schedule diffusion | final release |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.929 | 0.903 | 0.900 | 0.944 | 0.644 |
| 1 | 0.884 | 0.857 | 0.864 | 0.852 | 0.559 |
| 4 | 0.609 | 0.610 | 0.610 | 0.566 | 0.520 |

Values are unseen-canary ROC AUC. Against the strongest sum-LLR baseline, the
paired trajectory-bootstrap AUC differences for diffusion are `+0.015`
(`95% CI -0.003 to +0.036`) at sigma 0.25, `-0.032` (`-0.086 to +0.015`) at
sigma 1, and `-0.043` (`-0.113 to +0.026`) at sigma 4. The low-noise gain is
suggestive but not resolved by this sample. Diffusion does not rescue the
high-noise setting.

The fold models have different raw score scales. Consequently, thresholds
formed from pooled out-of-fold learned scores are poorly calibrated for the
four-model holdout ensemble; the learned low-FPR operating points are included
for transparency but are exploratory. AUC and paired AUC differences are the
primary comparisons.

These remain known-canary, white-box, weak-privacy results. The conservative
no-amplification epsilon upper bounds are 665.6, 70.4, and 11.6. The experiment
does not demonstrate reconstruction of an unknown MNIST image, and it provides
no evidence that increasing DP noise increases vulnerability.

## Reproduce

First complete `runs/canary_holdout`, then run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib \
python -m gaussproof.canary_learned \
  --source runs/canary_holdout --data /path/to/mnist_train.csv \
  --config configs/canary_learned.json --output runs/canary_learned
```

The run stores compact feature sequences, cross-fitted predictions, trained
fold models, aggregate metrics, paired uncertainty intervals, and vector plots.
