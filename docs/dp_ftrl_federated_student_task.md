# Federated DP-FTRL audit: student handoff

## Objective and current boundary

Determine whether a **known client that contributes once** can be detected from
the sequence of released global-model checkpoints in federated DP-FTRLM, and
whether the actual GAUSSPROOF sparse trajectory method adds value over matched
baselines. This is an empirical, client-level membership audit, not an attempt
to disprove differential privacy or reconstruct an unknown client's data.

Start with the existing [centralized MNIST audit](../reports/dp_ftrl/README.md)
and its [code](../experiments/dp_ftrl/README.md). That audit used the official
centralized PyTorch tree and optimizer, **64 records per round and zero
federated clients**. Its informed 128-checkpoint AUC was 0.693, but a
public-background version scored 0.518. It did not run the GAUSSPROOF sparse
decoder. Do not describe it as a federated privacy break.

The next experiment must use the authors' separate
[federated DP-FTRL implementation](https://github.com/google-research/federated/tree/master/dp_ftrl),
which implements DP-FTRLM with momentum and has EMNIST and Stack Overflow
drivers. Pin and record the exact upstream commit. The centralized identity
`-alpha * W_t = noisy prefix` **does not automatically carry over** to a
momentum server optimizer. Derive and test what an observer can infer from
the checkpoints before using any prefix-based score.

## Fixed threat models

Use one private client as the candidate. In the presence world, that client's
clipped model update occupies one scheduled client slot in exactly one round;
in the absence world, that slot is empty under an add/remove definition.
Keep all other clients, local training settings, client schedule, model
initialization, and random streams paired where the official mechanism allows.
The candidate must not reappear. Document whether the client identity and
participation round are known to the attacker. They are known only in an
explicitly labeled informed diagnostic; the main score must also search over
an unknown round or justify why the round is externally observable.

Measure these access levels separately:

1. **Informed diagnostic:** all global checkpoints; known target client's
   local data and participation round; exact other-client data and schedule.
   This approximates the current toy audit and is an upper-access control.
2. **Candidate-known trajectory attack:** all global checkpoints and a known
   target client's local data or auditor-controlled canary, but no other
   private client updates, batch identities, or tree-noise draws. Construct a
   fingerprint bank only from public/disjoint clients or from the known
   candidate's data. This is the principal test of practical added signal.
3. **Endpoint-only:** the same known candidate and only the final global
   model. A score called endpoint-only must not use intermediate states to
   estimate or subtract a background.

State who could actually see every checkpoint: a server-log observer or a
separately justified coalition. Ordinary clients do not automatically see the
whole sequence. Never feed server-internal Gaussian nodes, true client
inclusion indicators, or private other-client deltas to access level 2.

## Implementation order

1. **Reproduce upstream first.** Make a locked environment for the authors'
   federated code. Record the upstream SHA, dependency versions, data version,
   and an unmodified baseline run. If its older TensorFlow Federated stack
   cannot run, document the exact blocker before replacing any component.
2. **Minimal paired pilot.** Prefer federated EMNIST, whose clients have
   naturally partitioned data. Propose eight clients per round and nested
   windows of 16, 32, 64, and 128 rounds, with disjoint background clients
   when available. A 128-round, eight-client one-pass schedule needs at least
   1,024 distinct client slots. Use ten target clients for the plumbing pilot.
   Choose clipping and two noise levels from the *federated* accountant and
   validation utility before viewing attack outcomes; do not copy the toy
   sigma=4 setting without recalibration.
3. **Verify observations.** Save only the global checkpoints that the stated
   attacker can see. Unit-test the server optimizer recurrence, clipping,
   one-time insertion, tree-noise correlation, paired worlds, and identity
   separation. Confirm that scores are unchanged when hidden inclusion labels
   and sampled noise are removed from attacker inputs.
4. **Run matched attacks.** On identical paired runs, evaluate (a) final-model
   mean candidate loss, (b) a stronger final-model reference attack if
   affordable, (c) naive RERO-style time-aligned gradient/update alignment,
   (d) a covariance-aware known-fingerprint Gaussian likelihood as an
   informed analytic reference, and (e) the actual GAUSSPROOF sparse
   trajectory decoder adapted to client-update dictionaries. All trajectory
   methods must have the same checkpoints, candidate knowledge, and public
   bank. The analytic likelihood is a comparator, not evidence that the
   sparse decoder itself succeeded.
5. **Scale only after the pilot passes.** Aim for 40 disjoint calibration and
   60 held-out target clients, with three paired repetitions per identity;
   increase independent clients before repetitions. Report the count actually
   achieved and identity-clustered uncertainty. At least 100 negative trials
   are needed even to resolve an empirical 1% FPR point, and repeated trials
   on 60 clients are not 180 independent identities. Label low-FPR estimates
   underpowered until there are enough independent negative clients.

## Controls and decision rule

Report AUC, TPR at 1% and 5% FPR, attack advantage, calibrated achieved FPR,
model utility, privacy parameters, and intervals for **paired differences**.
Include same-class or same-population decoys, candidate-identity shuffling,
noise-only controls, unknown-background ablations, and late insertion. Repeat
across at least two client-data distributions only after the primary setup is
stable. Do not condition the headline result on observing an inclusion; any
such calculation is a labeled diagnostic. Do not compare different checkpoint
counts at an unstated or mismatched privacy budget.

A positive finding requires the public/candidate-known GAUSSPROOF score to
exceed both the matched RERO-style score and a credible endpoint baseline on
held-out clients, with a paired interval excluding zero. If only the informed
diagnostic works, report precisely that. If all methods are near chance,
retain the negative result. More noise lowering attack power is expected;
do not call a growing trajectory signal a violation of the formal accountant.

## Deliverables and review

The implementation lead opens one branch and a pull request containing a
reproduction command, pinned configuration, targeted tests, aggregate-only
CSVs, vector PDF figures, and a short report that traces every claim to a
table. Keep client data, individual membership scores, raw updates, and
credentials outside the repository. The independent reviewer checks upstream
fidelity, the checkpoint interface, absence-world construction, data leakage
into attack inputs, paired statistics, privacy accounting, and every paper
sentence before approval. Do not amend the manuscript with a federated claim
until this review is complete.
