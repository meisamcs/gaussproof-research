# Calibration-selected endpoint attacks in the DP-FTRLM pilot

**Result.** This controlled, source-aligned NumPy pilot finds no operational
high-noise membership advantage for GAUSSPROOF over calibrated endpoint LiRA
or RMIA. At tree multiplier σ=4, every tested public attack is near chance.
At σ=.25, a diagnostic given the *private* background updates and insertion
round detects the candidate, but the tested public attacks still have little
separation. These are client-membership results, not sample reconstruction.
The model is a 7×7-pooled softmax classifier, not the official DP-FTRL
TensorFlow Federated CNN; the result cannot establish deployed-system safety
or leakage.

## Protocol

Each trial asks whether one known 16-record EMNIST client was inserted once,
in round 33 of a 64-round run with eight client slots per round. Presence and
absence worlds share the seven background clients and tree-noise draws. The
observer for GAUSSPROOF, RERO, and public GLS sees every global checkpoint,
the candidate records, the mechanism parameters, and a bank of 12 disjoint
public clients. It does **not** see the target's other clients, their updates,
tree-noise draws, or the insertion round. Endpoint methods see only the final
model and candidate records, plus 32 independently trained OUT reference
models and 64 disjoint population clients. The informed likelihood uses the
private background and insertion round and is labeled only as a diagnostic.

The target uses genuine federated EMNIST digits, clipped one-step client
updates (`C=1`, client learning rate 1), the source-aligned DP-FTRLM server
recurrence (learning rate .2, momentum .9), and efficient-tree noise. Both
σ=.25 and σ=4 use exactly the same eight-slot design. The 16 images per writer
are sampled uniformly without replacement and transformed into pooled ink
intensities. We reserve all 1,124 writers used in a separate utility sweep
before forming attack roles. Natural writers, IID-mixed virtual clients, and
label-sorted virtual clients use disjoint candidate, reference/public,
population, and background writer partitions. The latter two are artificial
rearrangements of the *same* EMNIST records, not new datasets. The label-sorted
condition has about 98% mean single-class purity, versus about 23% for the
other two conditions.

For each distribution and noise level, 40 calibration client identities
select LiRA/RMIA score variants, mean/max record aggregation, and membership
score direction. A distinct set of 80 clients estimates the final AUC and
operating points. There are three independent target-training seeds. Offline
LiRA uses 10 predeclared variants; offline RMIA uses 48 combinations of its
reference correction, threshold, population denominator, and aggregation.
Original-paper scores and the pinned Pierre-Joly variants are retained as
separate, unselected rows in the [per-seed CSVs](replicated/seed_auc.csv).
This is a generous *offline endpoint adaptation* to client membership; it is
not an exhaustive upper bound on all possible LiRA/RMIA variants or online
shadow attacks. Variant selection never sees held-out membership labels.

## Held-out results

Numbers below are mean cross-identity AUC over three seeds. Seed ranges and
the other methods are in the [full aggregate](replicated/summary_over_seeds.csv),
with [public-attack figures for low noise](replicated/low_noise_auc.pdf) and
[high noise](replicated/high_noise_auc.pdf). AUC is computed only between
*different* client identities; self-comparisons of paired worlds are excluded.

| σ | Distribution | GAUSSPROOF max | GAUSSPROOF mean | RERO max | Calibrated LiRA | Calibrated RMIA | Informed diagnostic | Model accuracy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| .25 | Natural writer | .506 | .509 | .510 | .505 | .504 | .973 | .536 |
| .25 | IID mixed | .502 | .511 | .509 | .504 | .505 | .971 | .539 |
| .25 | Label sorted | .500 | .533 | .520 | .517 | .514 | 1.000 | .322 |
| 4 | Natural writer | .500 | .503 | .501 | .501 | .500 | .568 | .229 |
| 4 | IID mixed | .500 | .505 | .500 | .500 | .500 | .565 | .227 |
| 4 | Label sorted | .500 | .512 | .501 | .503 | .506 | .652 | .199 |

The high-noise model itself has low utility (roughly 20–23% accuracy on ten
digits). Therefore the near-chance attacks do not demonstrate safety for a
useful official model. Conversely, the informed result at low noise cannot be
presented as a public GAUSSPROOF break: it uses information excluded from the
public threat model. The public sparse `max` score often saturates at 0.5.
The slight `mean` AUC in the artificial label-sorted condition remains a weak
membership effect, not usable reconstruction evidence. There is no evidence
here that adding more noise *increases* attack success.

The [seed AUC file](replicated/seed_auc.csv) also reports calibrated 5% FPR
TPR and the **achieved** held-out FPR. The latter sometimes exceeds 5%; do
not cite the TPR alone as a fixed-FPR result. Forty calibration negatives
cannot resolve a 1% FPR threshold, so the individual CSVs mark that operating
point `nan`. Seed error bars are minimum and maximum across three seeds,
not deployment-wide confidence intervals. The conservative one-pass privacy
upper bound is ε≈3.39 at σ=4 and ε≈106.78 at σ=.25 for δ=10⁻⁵; it is not the
official TFF accountant and does not cover repeated client participation.

## Data-order correction and reproducibility

Earlier [DP-FTRLM distribution-pilot results](../dp_ftrl_distributions/README.md)
used the *first* 16 records of every writer. The archive's within-writer
ordering heavily favors digits 0 and 1 in those prefixes, making the pilot's
data and utility unrepresentative. This report supersedes those AUCs for
interpretation. The earlier files remain visible for audit, with a warning.
Across the 2,256 eligible writers after the utility reservation, digits 0–1
make up 58.0% of first-16 records but 21.5% of uniformly sampled records;
the [aggregate label histogram](sampling_histogram.csv) and
[`sampling_audit.py`](../../experiments/dp_ftrl_independent/sampling_audit.py)
reproduce this check.
The corrected experiment samples within each writer uniformly, excludes
short writers, inverts the image background into ink intensity, and uses a
separate utility-tuning writer set. We also raised endpoint resources from
eight to 32 OUT reference models and from 20 to 64 population clients.

The code is in
[`run_distributions.py`](../../experiments/dp_ftrl_independent/run_distributions.py),
[`calibrated_endpoint.py`](../../experiments/dp_ftrl_independent/calibrated_endpoint.py),
and [`summarize_calibrated.py`](../../experiments/dp_ftrl_independent/summarize_calibrated.py).
The [test suite](../../experiments/dp_ftrl_independent/test_calibrated_endpoint.py)
checks the score equations and that held-out labels cannot affect variant
selection. Each seed directory contains aggregate `summary.csv`,
`paired_comparisons.csv`, `calibration_selections.csv`, metadata, and plots;
there are no raw examples, client identities, per-client scores, or checkpoints.

From the repository root, with the FedJAX SQLite file in a private location:

```bash
python -m unittest discover -s experiments/dp_ftrl_independent -p 'test_*.py' -v
for seed in 20261011 20261013 20261017; do
  OPENBLAS_NUM_THREADS=1 python experiments/dp_ftrl_independent/run_distributions.py \
    --sqlite /private/path/federated_emnist_digitsonly_train.sqlite \
    --output "reports/dp_ftrl_calibrated_uniform/seed_${seed}" \
    --modes natural_writer iid_mixed label_sorted \
    --rounds 64 --position 32 --clients-per-round 8 \
    --calibration-identities 40 --holdout-identities 80 \
    --public-bank 12 --public-pool 512 --population-clients 64 \
    --background-pool 500 --reference-models 32 --examples 16 \
    --client-example-selection uniform --pixel-transform ink \
    --reserve-utility-clients 1124 --utility-seed 20260929 \
    --sigma 0.25 4 --clip 1 --client-lr 1 --server-lr 0.2 \
    --momentum 0.9 --bootstrap 400 --seed "$seed" --calibrate-endpoint
done
python experiments/dp_ftrl_independent/summarize_calibrated.py \
  --root reports/dp_ftrl_calibrated_uniform \
  --output reports/dp_ftrl_calibrated_uniform/replicated
```

The runner refuses to overwrite a populated output directory. The dataset
digest, all parameters, and baseline selections are recorded in metadata and
CSV. The pinned external attack code and original-paper formula validation
are discussed in the [baseline documentation](../../experiments/dp_ftrl_independent/DISTRIBUTIONS.md).
