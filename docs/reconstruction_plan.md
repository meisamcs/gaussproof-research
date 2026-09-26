# Learned-prior reconstruction: executable plan

This is a new experiment, separate from the earlier MIA pilot. The aim is to test
whether a learned prior over clean gradient sequences improves reconstruction of
unseen inputs from Gaussian releases. It is not a claim that more noise creates
information or that an attack defeats differential privacy.

## Experimental scope

Train a small CNN on a public MNIST subset and retain two public checkpoints. At
these fixed checkpoints, compute exact full-model per-example gradients and clip
each full vector to C. Concatenate the gradients across checkpoints. Generate
independent Gaussian releases at each checkpoint, repeated R times. The mean has
coordinate variance (sigma*C)^2/R. This is a controlled gradient-release mechanism,
not a DP-SGD training trajectory: the released private gradients do not update the
models. The target digit label is known to every method; the image is withheld.

Separate row-disjoint public model-training data, prior-fitting data, calibration
data, target images, and candidate distractors. Fit a low-rank conditional Gaussian
prior and a noise-conditioned neural denoiser using only the public prior subset.
Calibration chooses between neural training checkpoints. Target gradients are used
only to simulate releases and evaluate estimators, never to fit either learned prior.

## Comparisons

1. Observation mean (unrestricted continuous reconstruction baseline).
2. Per-label public prior mean, with no private release.
3. Low-rank projection of the observation.
4. Gaussian posterior mean with empirically fitted covariance (learned Wiener filter).
5. Neural posterior estimator in the same public subspace.
6. Gaussian posterior using a stale first-checkpoint fingerprint prior.
7. Release-permuted Gaussian posterior: tests whether sample-specific information matters.

Separately compare summed alignment and Gaussian likelihood identification on the
same exact, target-containing candidate bank, restricted to the known class. This
stronger-access diagnostic is not a fair continuous-reconstruction competitor: it
already knows the candidate images. Include posterior-mean gradient estimation from
this bank, and identify its access explicitly in CSVs and plots.

Test both known background (exact subtraction, analytically equivalent to isolated
target releases) and unknown random batch background. In the latter, subtract only
the expected public background and use its estimated diagonal covariance in the
learned basis. This Gaussian approximation need not match the true background mixture.
Exact-bank likelihood is reported as exact only for the known-background experiment.

## Tests, execution and outputs

- Algebraic tests: Gaussian posterior formula, high-noise prior limit, clipping,
  exact-bank norm correction, and equivalence of repeated-release likelihood to
  mean likelihood for fixed Gaussian signals.
- Protocol tests: disjoint splits, deterministic training/noise, all target records
  shared across methods, and prior-only invariance to observations.
- Run three independent seeds, three noise levels and three release counts; include
  unseen test targets in each seed, unknown-background and stale-prior controls.
- Invert estimated gradients to pixels from identical random starts; do not supply
  target pixels to inversion. Compare clean-gradient oracle, observation mean,
  prior-only mean, Gaussian posterior and neural posterior.
- Save raw per-record measurements locally and publishable aggregate CSVs with paired
  bootstrap intervals conditional on trained models, seed summaries, vector PDFs,
  PNGs, image reconstructions, configs, provenance and checkpoints.
- Report absolute error and improvement relative to prior-only prediction, not just
  percentage improvement over an increasingly noisy observation.

## Interpretation gates

Success requires improvement over a prior-only estimate on unseen targets, not just
denoising toward a plausible digit. Image gains must be measured separately from
gradient gains. Stale-prior and unknown-background losses limit practical claims.
Public-prior training and a neural architecture alone are not a novelty claim. The
matched Gaussian candidate likelihood is optimal for identity in its specified model;
we do not claim to beat it with the same information. Dynamic DP-SGD, unknown labels,
unknown participation, cross-architecture transfer and formal privacy auditing remain
separate extensions beyond this first complete reconstruction experiment.

Motivation: [RAoPT](https://arxiv.org/abs/2210.09375), learned denoising of protected
location trajectories. Prior-aware baseline:
[Hayes et al.](https://arxiv.org/abs/2302.07225). Neither source establishes the proposed
learned-gradient-prior advantage; that is the hypothesis tested here.
