# Undergraduate course project: public-background mismatch

## One question

**Does the known-record GAUSSPROOF trajectory signal remain useful when the public data used to estimate the minibatch background have the wrong digit distribution?** This tests a practical requirement of the audit: the candidate and noisy updates may be observable, but a representative public background may not be available. A negative answer is as valuable as a positive one.

## Fixed setting

Use **MNIST and the existing CNN only**. Start from the [matched-access experiment](../reports/trajectory_access_robustness/README.md) and its [configuration](../configs/trajectory_access_robustness.json): clipping norm `C=1`, batch size `B=8`, noise multiplier `sigma=4`, learning rate `0.005`, 128 updates, and candidate eligibility `q=0.5`. This is deliberately a high-participation stress test, not ordinary MNIST sampling or a small-epsilon result. The 20 calibration identities and 80 disjoint evaluation identities remain fixed. Do not add another dataset, model, denoiser, or attack family.

Compare exactly two public background estimates of the same size (64 images), selected **before** evaluating attacks and disjoint from all candidate and private population identities:

1. A digit-balanced public sample from the public checkpoint-training pool.
2. A public sample containing only digit `0`, from that same pool.

Generate paired IN/OUT trajectories with the existing simulator. The chosen public background affects **only the observer's background estimate**, never the private minibatch, DP noise, candidate inclusion schedule, or model update. Reuse the same seed and assert that the two public-background conditions produce identical noisy releases (and therefore identical checkpoint states) for each paired world. If that assertion fails, fix the experiment before interpreting AUC.

Score each observation with (a) the repository's q-aware Gaussian-mixture score and (b) its equal-access centered gradient projection. Both receive the same candidate fingerprints, releases, checkpoints, and public background estimate. Use no evaluation labels for score tuning or threshold selection. The final-model candidate loss may be shown as a context-only endpoint control, but no shadow models or LiRA are required.

## Primary result and deliverables

The primary result is the **paired change in within-digit membership AUC** for the q-aware score when moving from balanced to digit-0 public background. Compute AUC separately within each candidate digit and macro-average across the ten digits, so a digit-distribution shortcut cannot alone appear as record identification. Resample candidate identities (keeping each IN/OUT pair together) for a 95% interval. Report the same comparison for the equal-access projection, overall AUC, per-digit AUC, model accuracy, and the number of independent identities. If reporting a calibrated threshold, show its **achieved** holdout FPR and TPR; the existing 20-negative calibration set is too small to certify a 5% FPR point.

Submit one focused pull request containing:

- a fixed JSON configuration and a small test proving private releases are invariant to the public-background choice;
- aggregate CSVs and one PDF/PNG figure comparing the two backgrounds and two equal-access scores;
- a two-page report stating the threat model, exact split construction, paired uncertainty, compute time, failure modes, and what the outcome means for deploying a trajectory audit.

Keep `data/`, `runs/`, model files, raw identity-level scores, and personal information out of Git. The [reproducibility guide](../REPRODUCIBILITY.md) creates the exact MNIST CSV from public IDX files. First run the existing data-free tests, then reproduce the balanced-background control before changing the scoring pipeline. The earlier [gallery mismatch report](../reports/canary_gallery_replication/README.md) is motivation, not a result to copy or fit against.

This project answers an **audit-usability** question. It should not claim that more DP noise backfires, that a private image was reconstructed, that a formal privacy guarantee was broken, or that GAUSSPROOF improves on the equal-access projection unless the paired evidence actually supports it.
