# October system-model and evidence revision

The current paper source is `main.tex`. This revision makes the paper's
information boundary and empirical claim match the recent experiments.

## Major changes made

1. **Put the system and adversary model before the attack derivation.** The
   compact full-width figure now appears in the Threat Model section. It
   distinguishes the private batch and hidden inclusion schedule from the
   visible noisy updates and checkpoints, and shows the candidate and public
   data that the observer needs. The Gaussian LLR stays in the text. The
   figure is reproducible from `scripts/make_system_adversary_model.py`.
2. **Promote the 80-identity exposure and participation replication.** The
   abstract, introduction, setup, results, security implications, and
   conclusion now report the 16-to-256 release check at `sigma=4`, including
   `q=0.5, 0.25, 0.1`. The new two-panel figure and the matched `sigma=2,4,8`
   noise check show opposite directions for more exposure and more noise.
3. **Preserve the ordinary-sampling boundary.** The `q=0.004,T=128` pilot
   remains at chance; the 256-release replication does not include that arm.
   The paper does not extrapolate the frequent-participation result to an
   ordinary MNIST record.
4. **Keep comparisons honest.** The paper retains the useful-model LiRA and
   same-access Nasr-style projection results: neither establishes a distinct
   advantage for the Gaussian-mixture score. The 80 identities in the new
   trajectory run are the same fixed holdout block as the endpoint study,
   though the trajectories and seeds differ. They are not two independent
   identity samples.
5. **State the operational boundary.** At `sigma=4,T=256`, conservative
   no-amplification accounting gives epsilon at most 70.39 for delta `1e-5`.
   The nominal 5% FPR threshold realizes about 10-12% FPR in the new holdout,
   so the paper makes no validated low-FPR or empirical-epsilon claim. A
   separate natural-canary one-run extension did not establish a positive
   empirical epsilon lower bound or a gain over projection.

## Evidence and remaining gates

- New data and replication details: `reports/canary_first_principles_holdout/`.
- Same-access useful-model comparison: `reports/trajectory_endpoint_replication/`.
- Negative one-run follow-up: `reports/one_run_interference_confirm/`.
- Before submission or circulation, rebuild the manuscript PDF and check page
  limits and float placement. The checked-in September PDF is stale.
- A validated low-FPR detector needs a larger disjoint calibration set and
  prespecified threshold; an empirical DP lower bound needs a fixed-neighbor
  audit with appropriate confidence bounds. Neither is supplied by AUC.
- The author-anonymized artifact URL and its independence from the named
  GitHub research repository should be verified separately before review.
