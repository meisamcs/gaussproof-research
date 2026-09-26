# GAUSSPROOF: student execution plan through Monday, September 28, 2026

## Objective

Independently validate the focused experiment requested by Reza and deliver a
submission-ready result package by Monday. The repository already implements the
requested experiment. This is a reproduction, audit, and paper-integration task;
do not expand the scientific scope before the deadline.

The three required questions are:

1. Does the trajectory signal persist as participation probability decreases from
   `q=0.5` to `0.1`, `0.02`, and `0.004`?
2. How does the trajectory attack compare with a candidate-conditioned attack that
   sees only the final trained model?
3. Does the conclusion transfer beyond the original four holdout identities?

## Ownership

| Owner | Responsibility | Required output |
| --- | --- | --- |
| **Ananya, experiment lead** | Coordinate the run, reproduce the q study, own the result table and figure, and prepare the final pull request | Clean run directory, aggregate result package, updated paper, Monday status note |
| **Vikrant, independent verifier** | Audit the threat model, sampler, endpoint access, pairing, uncertainty calculation, and paper claims without relying on Ananya's interpretation | Signed verification checklist and review of the pull request |
| **Ananya + Vikrant** | Discuss the paper, resolve discrepancies, and rehearse the precise claim | A 10-minute explanation and a one-paragraph result summary |
| **Meisam** | Provide the MNIST CSV and compute access, decide any disputed claim, and approve the Monday freeze | Final decision on the submission text |

Use separate branches. Ananya should work on `satml/reza-experiment`; Vikrant should
review from `satml/reza-verification` or submit review comments without modifying the
result files.

## Read before running anything

The canonical manuscript is `paper/satml2027/main.tex`. If a copy named `main.txt`
was circulated, use the repository version to resolve differences.

Read these sections in order:

1. Abstract and Introduction: the exact claim and access model.
2. Threat Model and Evaluation Scope: what the observer sees and knows.
3. GAUSSPROOF: the evolving fingerprint, per-round LLR, and q-aware mixture score.
4. Experimental Setup: splits, pairing, identities, and endpoint baseline.
5. “Participation Rate Bounds the Accumulation Effect”: Reza's requested result.
6. Security Implications, Limitations, Conclusion, and Claim Checklist.

Then read:

- `docs/canary_q_sensitivity_protocol.md`
- `reports/canary_q_sensitivity/README.md`
- `gaussproof/canary_q_sensitivity.py`
- `tests/test_canary_q_sensitivity.py`

Before the run, Ananya should explain the following to Vikrant without reading from
notes: why the score is a mixture likelihood, why lower-q schedules are nested, why
q=0.004 often gives no realized inclusion, and why final candidate loss is an endpoint
attack while the last noisy update is not.

## Friday: environment and design audit

### Ananya

1. Clone the private repository and create the experiment branch.
2. Obtain `mnist_train.csv`. Confirm that its SHA-256 is
   `fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d`.
3. Create a fresh environment and run the test suite:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-tested.txt -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=.cache/matplotlib python -m gaussproof.cli smoke
```

Expected result: `48 tests passed`.

### Vikrant

Audit the implementation against this checklist:

- The q values are exactly `0.5, 0.1, 0.02, 0.004`.
- Prefixes are exactly `16, 32, 64, 128` from the same trajectory.
- There are 20 fixed unseen identities, two per digit.
- Each identity has three positive and three negative runs at each q.
- Candidate inclusion has a dedicated RNG; lower-q schedules are subsets of
  higher-q schedules.
- Positive and negative pairs share batch and Gaussian-noise seeds.
- The trajectory detector uses the correct q in the Bernoulli-mixture LLR.
- The endpoint score uses only the model after step T and the known candidate.
- Bootstrap resampling clusters by identity and paired sequence index.
- Evaluation identities never select a threshold or hyperparameter.

Any failed item blocks the experiment until Ananya and Vikrant agree on a fix and add
a regression test.

## Saturday: exact reproduction

First regenerate the public initialization and fixed splits. This run also reproduces
the original holdout experiment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=.cache/matplotlib python -u -m gaussproof.canary_holdout \
  --data /path/to/mnist_train.csv \
  --config configs/canary_holdout.json \
  --output runs/student_canary_holdout
```

Then run Reza's focused experiment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=.cache/matplotlib python -u -m gaussproof.canary_q_sensitivity \
  --source runs/student_canary_holdout \
  --data /path/to/mnist_train.csv \
  --config configs/canary_q_sensitivity.json \
  --output runs/student_canary_q_sensitivity
```

The measured reference runtime was about three minutes for the holdout source and
fourteen minutes for the q study on CPU. Preserve the complete output directory.
Never rerun into a completed directory.

### Reproduction gates

The new run must report:

- 480 exact trajectories;
- 1,920 prefix score rows;
- 20 unseen identities;
- nested inclusion schedules across q;
- identical negative endpoint scores across paired q conditions;
- finite aggregate values and a readable one-page PDF figure.

At `T=128`, the reference AUCs are:

| q | Expected inclusions | Trajectory AUC | Endpoint AUC |
| ---: | ---: | ---: | ---: |
| 0.5 | 64.0 | 0.805 | 0.730 |
| 0.1 | 12.8 | 0.571 | 0.549 |
| 0.02 | 2.56 | 0.531 | 0.517 |
| 0.004 | 0.512 | 0.503 | 0.498 |

Results should match the checked-in aggregate report to rounding. A discrepancy above
0.005 AUC, a changed sample count, or a failed structural check triggers investigation;
do not average conflicting runs or select the more favorable result.

## Sunday: independent verification and paper update

### Vikrant

1. Compare the new `summary.csv`, `trajectory_vs_endpoint.csv`, and
   `inclusion_summary.csv` with the checked-in report.
2. Confirm that 61.7% of q=0.004 positive runs at T=128 realize zero inclusions,
   close to the theoretical 59.9%.
3. Confirm that the q=0.5, T=128 trajectory-minus-endpoint difference is about
   `+0.076` and that its paired 95% interval `[-0.001, 0.148]` includes zero.
4. Check every manuscript number against an aggregate CSV cell.
5. Confirm that no row IDs, raw trajectories, checkpoints, dataset files, or local
   absolute paths enter the pull request.

### Ananya

Prepare one pull request containing only:

- any justified code or test correction;
- the aggregate CSV outputs and regenerated PDF/PNG figure;
- a short verification note with the exact command, dataset hash, revision, runtime,
  and test count;
- manuscript changes required by the reproduced values.

Keep the existing limitations and negative findings. Do not add a diffusion claim,
new dataset, new architecture, or free-form reconstruction experiment this weekend.

## Monday: freeze and deliver

By Monday noon Central time, Ananya and Vikrant should jointly deliver:

1. A passing 48-test log.
2. Reproduced aggregate CSVs and the q-sensitivity figure.
3. Vikrant's completed independent checklist.
4. A manuscript PDF in which the abstract, setup, result table, limitations, and
   conclusion agree with the CSVs.
5. A five-sentence note for Reza stating the design, main values, endpoint result,
   low-q boundary, and interpretation.

The paper can freeze when all five items exist and the pull request has no unresolved
scientific comments. Leave Tuesday for author certification, artifact availability,
formatting, and HotCRP submission checks.

## Allowed claims

- Weak known-candidate gradient evidence accumulates across repeated white-box releases
  when participation is frequent.
- At sigma=4 and q=0.5, trajectory AUC rises with T and reaches about 0.81 at T=128.
- The signal attenuates sharply at lower q and is at chance for q=0.004 within 128 rounds.
- The current endpoint comparison does not establish a statistically clear trajectory
  advantage because the paired difference interval includes zero.

## Claims the submission must not make

- Increasing Gaussian noise makes records more vulnerable at fixed exposure.
- The q=0.5 result describes ordinary uniform minibatch participation.
- GAUSSPROOF violates differential privacy or exceeds the stated composed guarantee.
- The gallery result is unknown-image reconstruction.
- Diffusion outperforms the analytic high-noise detector.

## Five-sentence Monday note to Reza

> We completed the requested focused sensitivity study at sigma=4 for q in
> {0.5, 0.1, 0.02, 0.004} and T in {16, 32, 64, 128}, using 20 unseen MNIST
> identities and three paired positive/negative repetitions per identity. At T=128,
> trajectory AUC is 0.805, 0.571, 0.531, and 0.503 as q decreases, while the
> final-checkpoint candidate-loss attack reaches 0.730, 0.549, 0.517, and 0.498.
> The q=0.004 result is chance-level because 61.7% of positive runs realize no
> inclusion within 128 rounds, close to the theoretical 59.9%. The q=0.5 trajectory
> point estimate exceeds the endpoint by 0.076, but its paired 95% interval
> [-0.001, 0.148] includes zero. We therefore present the result as a clear
> participation-sensitivity boundary rather than evidence of leakage for ordinary
> records or a statistically established trajectory advantage.
