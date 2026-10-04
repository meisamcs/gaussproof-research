# GAUSSPROOF: fixed-canary interference as the decision gate

## Research question

When an auditor must test a prespecified collection of natural records in one
DP-SGD training run, can a **joint trajectory score** recover more of their
independent IN/OUT audit bits than separate canary projections, given exactly
the same checkpoints, noisy updates, public background, and mechanism
parameters? The operational payoff would be stronger valid one-run audit
lower bounds or more natural canaries audited at the same confidence level.
The project has **not yet demonstrated that payoff**.

For fixed dataset-level bits S_i and independently sampled participation
B_it, the released update is approximately

    Y_t = b_t + (1/B) sum_i S_i B_it h_it + Z_t,

where h_it is the clipped gradient of natural canary i recomputed at the
pre-update checkpoint. A separate projection onto h_it receives the target
signal plus terms proportional to h_it^T h_jt from other canaries. Those
cross-terms vanish for orthogonal fingerprints. If all fingerprints coincide
throughout training, individual bits are unidentifiable. The interesting
middle case has substantial overlap **and** enough variation across time to
identify the contributors. GAUSSPROOF's proposed role is to model the
persistent bits jointly while allowing uncertain per-round participation.

## Position relative to recent work

- Steinke, Nasr, and Jagielski (2023) supply the one-run randomized-canary
  protocol, clipped-gradient score, and conservative audit test. Their strong
  Dirac canaries are intentionally near orthogonal; this is a negative
  control for an interference decoder.
- Dagréou and Bellet (2026) improve one-run auditing by **selecting or
  crafting canaries** to reduce mutual interference. Our proposed use is
  different only where the audited natural cohort is fixed in advance and
  cannot be replaced with easy canaries. Their selected natural canaries are
  still a valuable additional benchmark when selection is allowed.
- Agrawal et al. (2026), *Let's Ask Gauss*, improve the statistical analysis
  of **scalar canary scores** for one-run DP-SGD and DP-FTRL auditing. A new
  joint score must be evaluated against strong scalar scores under the same
  valid audit conversion. We must not attribute a gain from a different
  confidence procedure to gradient decoding.

## Prespecified overlap test

The runnable pilot is `gaussproof/interference_audit.py` with
`configs/one_run_interference_pilot.json`. Two digit classes, the number of
canaries per class, Poisson q for ordinary and IN canaries, the guess budget,
and all attack hyperparameters are fixed in the config. Before any hidden
audit bits or private draws, public-initial-model gradients select low- and
high-overlap natural-image cohorts **within those same classes**. Selection
does not consult the noisy trajectory. Every score gets the same observations
and canary bank within a run. The report records within-class cosine overlap,
effective rank, correct fixed guesses, descriptive AUC, and the paper's
one-run epsilon lower bound. Repetitions use new inclusion bits, batches,
and noise; uncertainty is clustered by training run.

The primary test is a *paired interaction*: does the joint score's gain over
each separate score grow from low to high overlap? The joint score must also
beat the strongest same-access comparator in the high-overlap condition on
fixed guesses and, ideally, on a valid audit lower bound. Orthogonal
canaries should show no gain, while near-duplicate fingerprints may be
unidentifiable for any method. AUC alone cannot establish audit utility.

The eight-run pilot is exploratory. Its high-overlap result suggests a small
joint-score lead over separate projections, but the per-round sparse decoder
is close and the overlap interaction is unresolved. We therefore freeze a
fresh 20-run confirmation at sigma=2 (`configs/one_run_interference_confirm.json`)
with unchanged classes, cohort selection, sampler, score hyperparameters, and
guess budget. The **primary contrast** is joint trajectory minus per-round
sparse on correct guesses in the high-overlap cohort. The paired high-minus-low
interaction against sparse is a separate, necessary check of the mechanism.
We stipulated that a positive 95% run-bootstrap interval on *both* would
support the proposed use. The paper's valid epsilon lower bound would also
need to increase before claiming a tighter privacy audit.

**Observed decision:** the confirmation failed. In the high-overlap cohort,
joint-minus-per-round-sparse was +0.70 of 32 correct guesses with paired
interval [-0.10, 1.55]; the high-minus-low interaction was -0.05
[-1.60, 1.50]. Joint-minus-paper-dot was -0.50 [-2.20, 1.35]. No method
yielded a positive one-run epsilon lower bound in the 20 confirmation runs.
The full [confirmation report](../reports/one_run_interference_confirm/README.md)
therefore rules out promoting the exploratory lead into a paper claim.

The strongest remaining technical explanation is **identifiability**: the
high-overlap natural canaries have initial effective Gram rank only about
2.55 for 64 bits. A joint decoder cannot create independent information
from near-collinear fingerprints. A toy Gaussian model with asymmetric,
better-conditioned fingerprints shows that joint inference can help, but
that is a diagnostic of the method, not evidence for a practical privacy
audit. Any future positive claim needs a new real-data protocol whose
fingerprint geometry is both nonorthogonal and identifiable, and then a
held-out win over the strongest same-access score and audit bound.

## Sources

- Steinke, Nasr, Jagielski, *Privacy Auditing with One (1) Training Run*:
  https://arxiv.org/abs/2305.08846
- Dagréou, Bellet, *Detectability in Diversity*:
  https://arxiv.org/abs/2605.27292
- Agrawal et al., *Let's Ask Gauss*:
  https://arxiv.org/abs/2606.12733
