# Experimental protocol

## Actual training channel

At step t, sample B records uniformly without replacement from the target training set.
Compute per-record full-parameter gradients and clip each to norm C. Release

```text
g_i,t = grad_theta loss(model_theta_t(x_i), label_i)
h_i,t = g_i,t * min(1, C / ||g_i,t||_2)
U_t = sum_{i in batch_t} h_i,t + sigma*C*Normal(0,I)
theta_{t+1} = theta_t - learning_rate * U_t/B
```

The attack sees `Y_t = U_t[coordinates]` and constructs `D_t[i] = h_i,t[coordinates]`.
The coordinates are a prespecified random subset of head parameters. Clipping is never
performed independently in the reduced space. No proxy feature vector replaces gradients.
Model weights before each update are assumed observable to the gradient attacker.

The candidate dictionary includes both final-evaluation and calibration candidates but
does not include every target training record. Unrepresented training contributions are
background model mismatch, shared by both trajectory attacks. Participation coefficients
are unknown to the attacks. No training participation trace is passed to either decoder.

RERO-style scores average `<D_t[i], Y_t>` over t. GAUSSPROOF solves each convex problem

```text
minimize_z  0.5 * ||D_t.T z - Y_t||_2^2 + lambda * sum(z)
subject to 0 <= z_i <= 1
```

using accelerated projected proximal gradients with a spectral Lipschitz constant.
Scores average fitted coefficients over steps. `lambda = sparse_penalty*C^2` is fixed
before evaluation. This version does not implement HMM persistence, covariance whitening,
AMP state evolution, or pixel reconstruction; those require distinct validated methods.

## Black-box and hybrid

Shokri-style is a pooled regularized logistic classifier over probability vector,
one-hot true class, negative loss, confidence, negative entropy and margin. Training
observations come solely from shadow models. It is an adaptation of the shadow-training
principle, not an exact reproduction of the original architecture or classwise classifiers.

RMIA uses true-label probabilities. Let q(x) be the mean probability from OUT reference
models, all trained on a disjoint pool. With fixed a=0.5 by default,

```text
p_hat(x) = ((1+a)*q(x) + (1-a))/2
r(x) = p_target(x) / p_hat(x)
score(x) = mean_{z in independent population}[r(x) > gamma*r(z)]
```

The same approximation is applied to z. Default gamma=1. This is the offline population
test, not a Gaussian member/nonmember loss-density ratio. Reference model count, a, gamma
and population size are recorded. The whole population is OUT for targets and references.

The hybrid standardizes RMIA and GAUSSPROOF using the fitting half of known calibration
members/nonmembers, then fits a regularized logistic classifier. The second half chooses
thresholds; no final-evaluation labels influence weights or thresholds. This assumes
known membership for an attacker calibration subset; all calibrated attacks get that same
threshold subset. The hybrid has this extra fitting access, which must be stated in papers.

## Metrics and controls

ROC points are computed at distinct score thresholds, treating all ties together.
AUC is trapezoidal area on those points. Reported low-FPR TPR is the maximum achievable
empirical TPR at a threshold satisfying the budget; no extrapolated or interpolated TPR
is invented. Separately, a calibration threshold satisfying the calibration FPR budget
is applied to evaluation and its **actual** achieved FPR and TPR are reported. Calibration
at a small sample size is not a guarantee of a population-level FPR bound.

Bootstrap intervals resample members and nonmembers independently, conditional on one
trained target. They do not incorporate target-training variability. Multiple seeds
address that separately. Plots show seed standard deviations where available.

Controls include independent synthetic dictionaries with known member-only signal,
pure Gaussian noise, nonmember-only signal, and a coupled high-noise release. Nonmember-only
signal should reverse the intended membership ranking, not necessarily give AUC 0.5.
Real-run membership-label permutations are summarized over repeated random permutations.
Real-dictionary noise-only scores are also reported; a target-dependent dictionary can
itself correlate with membership, so a single noise-only observation is not hard-coded
to pass a chance-AUC assertion. Sweeps do not assert that every empirical point decreases
with noise or that GAUSSPROOF beats any baseline.

## Privacy accounting

The code reports a deliberately loose bound under fixed-size **replace-one** adjacency.
Clipped-sum sensitivity is at most 2C, so one release of Gaussian noise sigma*C satisfies
rho=2/sigma² zCDP. Adaptive composition over T updates gives rho_total=2T/sigma², and

```text
epsilon = rho_total + 2*sqrt(rho_total*log(1/delta))
```

No subsampling amplification is claimed. Sigma=0 is marked with no finite epsilon.
The Gaussian zCDP and conversion formulas follow
[Bun and Steinke (2016)](https://arxiv.org/abs/1605.02065).
This conservative bound includes the entire full-vector update sequence, so observing
coordinate projections and postprocessed model states is covered by that same upper bound.
The benchmark does not implement an optimized accountant. AUC gains do not contradict DP.
Any broader release of private metadata, member lists or metrics is outside the model's
DP mechanism: the saved benchmark artifacts are research data, not privacy-protected releases.

## Reproducibility and reporting

CPU deterministic PyTorch, fixed seeds, independent sampling/noise generators, immutable
dataset SHA256, sample IDs, configs, model checkpoints, scores, observed trajectories,
calibration fit parameters and dependency versions are saved. The compressed dictionaries
can be large. A completed run is marked only after all CSVs, controls and figures finish.
Interrupted runs retain partial outputs and must use a new output directory when restarted.

Always report target train/held-out accuracy beside attack metrics, threat-model differences,
negative controls, member/nonmember counts, FPR resolution, seed count, and the limitations
of the conservative privacy bound. Do not infer general leakage trends from the quick run.
