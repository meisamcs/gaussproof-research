# GAUSSPROOF: return to the first-principles question

## The question actually supported by the initial experiments

When an observer sees a sequence of noisy DP-SGD updates and knows a candidate
record, can repeated participation make that record's otherwise weak clipped
gradient fingerprint detectable across evolving model states? The candidate's
participation times remain hidden. This is a **longitudinal white-box exposure
question**, not a claim that more Gaussian noise increases leakage at a fixed
number of releases, that the auditor beats every same-access projection, or
that it reconstructs an unknown image.

At checkpoint \(\theta_t\), compute the candidate fingerprint
\(h_t=\operatorname{clip}_C(\nabla_\theta\ell(\theta_t,z))\). If its inclusion
replaces an ordinary record whose expected clipped gradient is \(b_t\), the
candidate-specific shift in the averaged release is \(s_t=(h_t-b_t)/B\).
Writing \(v=(\sigma C/B)^2\), the Gaussian-shift log likelihood is

\[
\ell_t=\langle s_t,y_t-b_t\rangle/v-\|s_t\|^2/(2v).
\]

For unknown Bernoulli participation with probability \(q\), the trajectory
score is \(M_T=\sum_{t=1}^T\log(1-q+q e^{\ell_t})\). The observer needs the
candidate and released model states/updates, but never receives private batch
membership or the sampled Gaussian noise. This is a matched-fingerprint
accumulator, not a diffusion denoiser.

In the weak, approximately stationary Gaussian channel, separation scales
roughly as \(q\sqrt{T}\|h-b\|/(\sigma C)\). This predicts that more observations
and more frequent participation can strengthen detection, while more noise
weakens it at fixed exposure. It does **not** predict that increasing noise
backfires. The paper gives the exact single-round chi-squared relation
\(\chi^2(P_1\|P_0)=q^2(e^\kappa-1)\), with
\(\kappa=\|s\|^2/v\), for the fixed Gaussian channel.

## What was measured

The original high-noise test used an exact evolving MNIST CNN at
\(\sigma=4\), \(C=1\), \(B=8\), and \(q=0.5\). The **same paired trajectories**
were scored at nested prefixes, so the length comparison is direct:

| Releases | Trajectory AUC | Paired gain over 16 releases |
| ---: | ---: | ---: |
| 16 | 0.609 | -- |
| 32 | 0.675 | +0.067 |
| 64 | 0.768 | +0.159 |
| 128 | 0.819 | +0.211, 95% interval [0.144, 0.272] |

The first test had four unseen canary identities and 12 paired IN/OUT
trajectories per identity. Its identity-level interval is descriptive because
there are only four independent identities. A subsequent 20-identity
sensitivity experiment at 128 releases reproduced the high-participation
effect and measured its boundary:

| Conditional participation \(q\) | Expected inclusions | Trajectory AUC |
| ---: | ---: | ---: |
| 0.5 | 64 | 0.805 |
| 0.1 | 12.8 | 0.571 |
| 0.02 | 2.56 | 0.531 |
| 0.004 | 0.512 | 0.503 |

At \(q=0.004\), about 60% of eligible records are never included in 128
rounds, making near-chance detection unsurprising. These data support a
**frequent-participation, repeated-release** result. They do not establish
leakage for an ordinary record with \(q\approx B/N=0.004\) over this horizon.
They also do not establish a small-\(\epsilon\) failure: the reported
conservative no-amplification upper bound at \(T=128,\sigma=4\) is
\(\epsilon\leq43.14\) for \(\delta=10^{-5}\).

## The distinction the later work must preserve

The longitudinal signal is real in the tested high-\(q\) regime. A different
question is whether the q-aware likelihood **outperforms another auditor with
the same trajectory and candidate fingerprint**. In a useful-model
80-identity replication, the q-aware score had AUC 0.818 and a simple raw
projection 0.831. A later one-run joint-scoring patch also failed to improve
clearly on the paper's projection score. Those null comparisons limit claims
of a superior *scoring algorithm*; they do not negate the measured growth of
record-specific evidence with repeated releases.

The defensible contribution is therefore a characterized exposure regime and
an audit procedure: recompute a known natural candidate's fingerprint at each
checkpoint, track its evidence over time, and report the participation and
release horizon at which the signal becomes useful. Nasr-style projection is
an essential equal-access control and part of the relevant prior art, not a
competitor that GAUSSPROOF must necessarily defeat to establish the exposure.

## New first-principles replication

An 80-identity exact evolving-model replication now extends the horizon to
256 rounds at sigma 4. The q-aware score's AUC from T=16 to T=256 is
0.633 to 0.843 at q=0.5, 0.572 to 0.696 at q=0.25, and 0.530 to 0.589
at q=0.1. Paired identity-bootstrap intervals for all three length gains
are above zero. On the same 80 identities and one matched sequence each,
the T=256 AUC at q=0.5 is 0.910, 0.843, and 0.737 for sigma 2, 4, and 8.
Both adjacent paired noise contrasts have intervals above zero in the
expected direction: **more noise weakens detection at fixed exposure**.
The [complete report](../reports/canary_first_principles_holdout/README.md)
contains the figure, CSVs, intervals, and protocol caveats.

The remaining usability questions are calibrated low-FPR detection and
candidate-gallery identity ranking on this same held-out regime, with
same-class and mismatched-background controls. An equal-access projection
remains the algorithmic control. These are distinct from the now-replicated
trajectory-accumulation hypothesis; none should be described as unknown-image
reconstruction or an empirical DP lower bound without the required audit
protocol.

Source results: [long-trajectory audit](../reports/canary_long_sigma4/README.md),
[q-sensitivity](../reports/canary_q_sensitivity/README.md),
[matched-access replication](../reports/trajectory_endpoint_replication/README.md),
and [one-run joint-score test](../reports/one_run_audit_joint/README.md).
