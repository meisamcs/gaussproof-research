# GAUSSPROOF: scope and equal-access experiment plan

## What the project has established

GAUSSPROOF has explored three related but distinct tasks: (1) detecting a
*known* candidate's participation from noisy DP-SGD updates, (2) identifying
one record in a supplied gallery, and (3) estimating a clean gradient or an
unknown image. The strongest repeatable result so far concerns the first task
under a privileged, frequently repeated canary: at sigma 4, q=0.5, and 128
releases, known-candidate AUC is about 0.8. At q=0.004 it is near chance.
The existing gallery experiment reports above-random exact linkage but does
not generate an unknown image. Full-history diffusion and LSTM pilots did not
beat simpler same-access estimators. Increasing Gaussian noise at fixed
exposure consistently weakened the tested attacks. These negative controls
are part of the claim, not exceptions to discard.

LiRA and RMIA answer a final-model membership question with different access
and reference data. They may characterize a deployment's endpoint exposure,
but they are not the primary algorithmic comparison for a white-box gradient
method. ReRo is a *reconstruction robustness* definition/bound, and the
Hayes–Balle–Mahloujifar attack assumes an informed adversary who knows the
other training records and subtracts their gradients. Our public-background
alignment is inspired by its prior-aware inner-product attack; it is **not**
an exact implementation of its informed-background protocol. An informed
ReRo attack belongs in a separately labeled stronger-access comparison.

## Primary position: a white-box auditing score patch

The defensible operational use is a **white-box privacy-auditing score patch**.
An internal auditor independently randomizes each of many known canaries to be
IN or OUT *before one training run*. The auditor then uses only released
updates/checkpoints, the known canaries, mechanism parameters, and disjoint
public background examples to rank those bits. The output is a stronger or
weaker empirical audit under a stated confidence procedure, not a new privacy
guarantee. Known-candidate gallery linkage and clean-gradient estimation are
diagnostic tasks that help explain whether a score patch is plausible; neither
is an audit by itself or evidence of unknown-image reconstruction.

The reference score is the clipped-gradient dot product across checkpoints
from Algorithm 3 of Steinke, Nasr, and Jagielski, *Privacy Auditing with One
(1) Training Run*. Their strongest gradient-space audit uses inserted Dirac
canaries with nearly orthogonal gradients. For those known Gaussian signals,
matched filtering is close to the right detector, so we should **not** claim
that generic denoising beats it. A useful GAUSSPROOF patch would instead
address ordinary image canaries whose gradients overlap each other and the
non-canary batch, or measured nuisance structure from a disjoint public bank.
It must use the same trajectory, checkpoints, canaries, and auxiliary examples
as a strengthened Nasr-style score, and it must improve the *auditing result*
without using hidden inclusion bits or choosing the best score after seeing
them. Fingerprint recomputation cost belongs in the usability comparison.

An algorithm-specific claim requires a gain over attacks with the **same**
observations and auxiliary records. The candidate-conditioned Gaussian
likelihood is the natural benchmark, not a weak straw man: under its exact
single-candidate Gaussian model it is optimal for that hypothesis test. A
sparse decoder could outperform this approximation only when its extra
structural assumptions (e.g. several possible simultaneous contributors,
correlated background, incomplete fingerprint bank) match the actual data.

## Experiment ladder and decision gates

1. **One-run randomized-canary audit (three pilots completed).** Draw each canary's
   dataset-level inclusion bit independently with probability 1/2 before
   training one evolving CNN. Conditional on being IN, sample the canary into
   each round according to the declared mechanism; keep the ordinary batch
   draw independent of the bits. Compare the original Algorithm 3 dot-product
   score with public-background residualization, a q-aware Gaussian mixture,
   per-round sparse decoding, and a joint persistent-bit trajectory score.
   Give every score exactly the same
   checkpoint trajectory and public bank. Fix the sparse penalty, number of
   positive/negative guesses, and methods before seeing audit bits. Measure
   correct guesses, AUC as a descriptive diagnostic, and a conservative
   simultaneous lower bound on epsilon. The q=0.1 canary-oversampling pilot
   gives no reliable gain over the paper's dot product on fixed guesses or
   epsilon lower bounds, despite a modest AUC gain against an independent
   Gaussian score at sigma=1. The uniform-Poisson q=0.03125 pilot finds
   near-chance audit power and no patch gain. These are small-m protocol
   checks, not a claimed reproduction of the CIFAR-10 Dirac audit.
   Scale canary count and repeat over new models only if a patch survives the
   strongest same-access score under realistic sampling. Orthogonal Dirac canaries are a negative
   control where GAUSSPROOF should have no advantage.
2. **Equal-access gallery and denoising diagnostic (completed).** Train
   actual evolving MNIST CNN DP-SGD trajectories from a public warm start.
   Give all methods the identical release, checkpoint-derived clipped
   32-record gallery, 64-record public background, C, sigma, B, and q. Compare
   centered alignment, norm-corrected Gaussian score, q-aware mixture score,
   and a nonnegative bounded sparse decoder. Use one target record eligible
   per round, hidden inclusion, 10 disjoint calibration identities and 20
   held-out identities, balanced and long-tail private distributions, and
   prefixes 64/128. Select decoder penalty only on calibration identities.
   Primary metrics: exact top-1, same-class top-1, mean reciprocal rank,
   and paired identity-clustered differences. Secondary metric: full-vector
   clean batch-gradient MSE against public-background, raw-update, and
   q-aware online posterior-mean estimators. The latter uses the same gallery
   and noisy trajectory through the current round. Include candidate-absent
   and background-mismatch controls.
   A mere gain over raw noisy gradients is insufficient; a claim of
   GAUSSPROOF's algorithmic advantage needs a positive held-out gain over
   the strongest same-access comparator with an uncertainty interval.
3. **Multi-contributor diagnostic (pilot completed).** Test several simultaneous
   known candidate contributors, where sparse recovery has a principled
   role, against an exact small-gallery posterior with identical access.
   The first MNIST CNN pilot finds a modest sparse *clean-gradient MSE* gain
   at sigma=1 but weaker contributor support AUC; at sigma=4 the sparse
   decoder loses to both the posterior and public prior. Next independently
   vary background accuracy, candidate-bank coverage, checkpoint frequency,
   and q knowledge. Keep hyperparameters fixed after calibration and repeat
   over new public initial models, galleries and seeds. Report computation
   and memory beside recovery.
4. **Actual ordinary sampling and distribution transfer.** Sample fixed-size
   minibatches from N=2000 so q=B/N arises from the sampler, rather than a
   privileged canary slot. Increase T to obtain several expected inclusions,
   and account for the *actual* sampling mechanism. Repeat identity-held-out
   evaluation on a second dataset/architecture and class skew. Keep model
   utility visible. A chance result bounds the useful scope of the attack.
5. **Unknown-image reconstruction (separate gate).** Exclude each target
   image from all candidate and public-prior training sets. Optimize a public
   generator's latent code against the full observed trajectory, compare it
   with direct gradient inversion and a public prior-only generator under
   identical access, and evaluate pixel MSE, identity-aware retrieval, and
   nearest-neighbor memorization. Do not relabel gallery selection as this task.
6. **Informed ReRo comparison.** Reproduce the paper's known-other-records
   subtraction and prior-aware reconstruction on a matched full-batch or
   minibatch protocol. Show its extra auxiliary information explicitly;
   compare reconstruction error under matched release budgets, not a
   membership AUC against an unrelated interface.

## Non-claims

No experiment here proves that adding noise reduces privacy or violates DP.
An empirical AUC or gallery rate is not an epsilon lower bound. A denoised
batch gradient is not an unclipped individual gradient. A selected gallery
image is not generated unknown data. The q=0.5 pilot intentionally tests
repeat exposure and does not represent ordinary uniform minibatch sampling.
The one-run audit's canary-inclusion probability 1/2 is a randomized audit
design; it is distinct from the per-round sampling probability q. A strong
result for natural image canaries would not supersede the paper's stronger
Dirac-gradient worst-case audit.

## Primary sources

- Hayes, Balle, and Mahloujifar, *Bounding training data reconstruction in
  DP-SGD*, NeurIPS 2023, Sections 2–4, especially Algorithm 2:
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/f8928b073ccbec15d35f2a9d39430bfd-Abstract.html
- Steinke, Nasr, and Jagielski, *Privacy Auditing with One (1) Training Run*,
  Algorithm 3, Theorem 5.2, and Section 6:
  https://arxiv.org/abs/2305.08846
- Existing results and exact commands: `reports/canary_gallery_replication/`,
  `reports/trajectory_access_robustness/`, `reports/ordinary_q_budget/`,
  and `reports/trajectory_dp/`.
