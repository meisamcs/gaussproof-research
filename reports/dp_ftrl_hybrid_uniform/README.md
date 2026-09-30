# LiRA and RMIA with and without GAUSSPROOF

**Result.** Combining the public GAUSSPROOF trajectory score with either
calibrated endpoint LiRA or RMIA produces small AUC gains in this controlled
DP-FTRLM pilot. The largest mean gain from the fixed 50:50 combination is
+.0106 AUC (RMIA, low-noise label-sorted clients). For natural writers at
high noise, the gains are only +.0014 (LiRA) and +.0020 (RMIA); the combined
AUCs are .5023 and .5021. These are **not useful membership attacks** and do
not demonstrate a high-noise privacy failure. The high-noise model also has
poor classification utility. The pilot uses a source-aligned NumPy softmax
model, not the official TensorFlow Federated CNN or a deployed system.

## Paired comparison

Every row below uses the same 40 calibration identities and 80 disjoint
evaluation identities for both endpoint and combined attacks. Values are mean
cross-identity held-out AUC over three target-training seeds. The [complete
seed AUCs](replicated/seed_auc.csv), [paired gains](replicated/paired_seed_gains.csv),
and [vector figure](replicated/hybrid_auc_gain.pdf) retain the seed-specific
results; error bars in the figure show seed **minima and maxima**, not a
confidence interval.

| Noise σ | Client distribution | LiRA | LiRA + GP 50:50 | LiRA + GP selected | RMIA | RMIA + GP 50:50 | RMIA + GP selected | GP alone |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| .25 | Natural writer | .5055 | .5107 | .5093 | .5037 | .5093 | .5080 | .5090 |
| .25 | IID mixed | .5038 | .5094 | .5109 | .5049 | .5117 | .5105 | .5111 |
| .25 | Label sorted | .5167 | .5213 | .5167 | .5137 | .5243 | .5156 | .5332 |
| 4 | Natural writer | .5008 | .5023 | .5027 | .5001 | .5021 | .5018 | .5029 |
| 4 | IID mixed | .5004 | .5038 | .5039 | .5004 | .5037 | .5038 | .5051 |
| 4 | Label sorted | .5033 | .5041 | .5033 | .5063 | .5050 | .5063 | .5120 |

The calibration-selected combination sometimes chooses zero trajectory
weight, particularly for label-sorted clients. Its choice varies across
seeds; it does not reliably outperform the fixed combination. GP alone is the
mean bounded sparse-coordinate score across the full checkpoint trajectory.
The `max` variant often saturates and was not used for fusion. At low noise,
the **informed** known-background likelihood diagnostic reaches AUC .973 on
natural writers. It knows private background contributions and the insertion
round, so it is not a public GAUSSPROOF or combined-attack result.

## What is combined

LiRA and RMIA receive the final model, the candidate's 16 labeled records,
32 independent OUT reference models, and 64 population clients. Their score
variant, record aggregation, and direction are chosen using only the 40
calibration identities. The GP observer additionally receives all 64 global
checkpoints, public mechanism parameters, and a disjoint 12-client public
fingerprint bank. It does not receive the other clients' private updates,
tree-noise draws, or the candidate's insertion round. Therefore the comparison
tests **incremental value from stronger trajectory access**; it is not an
equal-access contest between endpoint and trajectory attacks.

For endpoint score \(E\) and trajectory score \(G\), calibration scores in
both membership worlds determine their pooled means and standard deviations.
The fused score is
\[
S_w=(1-w)(E-\mu_E)/s_E+w(G-\mu_G)/s_G.
\]
We report a predeclared \(w=.5\) and a separate choice from
\(\{0,.25,.5,.75,1\}\) that maximizes **calibration** cross-identity AUC.
No held-out score or membership label influences scaling, weight selection,
or endpoint variant selection. Zero trajectory weight is a fallback to the
endpoint. [Selections by condition and seed](replicated/fusion_selections.csv)
are available for audit.

The task is whether one known client participates **once** in round 33 of
64, with seven background clients in an eight-slot round. Presence and
absence worlds share the background schedules and tree-noise draws; the
attacker sees neither. Client updates are clipped to \(C=1\). We test tree
noise multipliers \(\sigma=.25\) and \(4\), server learning rate .2, and
momentum .9. Natural EMNIST writers, IID-mixed virtual clients, and
label-sorted virtual clients are different arrangements of the same digit
data. All role partitions and a separate utility-tuning set are disjoint.
The [corrected baseline report](../dp_ftrl_calibrated_uniform/README.md)
explains why we sample 16 records uniformly within writers instead of using
their digit-skewed first 16 records. Its 90 baseline method-condition rows
match the corresponding hybrid runs **exactly**, including AUC, bootstrap
limits, model accuracy, and advantage, for each of the three seeds.

## Interpretation and limits

The high-noise endpoint scores are near .5 and adding GP changes AUC by
thousandths. A positive difference over a near-chance baseline should not be
described as a successful attack. Some individual identity-bootstrap gain
intervals include zero; three target seeds are too few to claim a general
improvement. The per-seed [summary files](seed_20261011/summary.csv) include
TPR at calibration-selected 1% and 5% FPR thresholds **and the achieved
held-out FPR**. With just 40 calibration negatives, 1% FPR cannot be resolved
and even the nominal 5% threshold often exceeds 5% on held-out identities.
Do not cite those TPRs without their achieved FPRs. The runs show client
membership scores, not reconstructed images or sensitive records.

## Reproduce

The dataset stays local; only aggregate results and figures are published.
With the FedJAX federated EMNIST digits-only training SQLite file at a private
path, run from the repository root:

```bash
python -m unittest discover -s experiments/dp_ftrl_independent -p 'test_*.py' -v
for seed in 20261011 20261013 20261017; do
  OPENBLAS_NUM_THREADS=1 python experiments/dp_ftrl_independent/run_distributions.py \
    --sqlite /private/path/federated_emnist_digitsonly_train.sqlite \
    --output "reports/dp_ftrl_hybrid_uniform/seed_${seed}" \
    --modes natural_writer iid_mixed label_sorted \
    --rounds 64 --position 32 --clients-per-round 8 \
    --calibration-identities 40 --holdout-identities 80 \
    --public-bank 12 --public-pool 512 --population-clients 64 \
    --background-pool 500 --reference-models 32 --examples 16 \
    --client-example-selection uniform --pixel-transform ink \
    --reserve-utility-clients 1124 --utility-seed 20260929 \
    --sigma 0.25 4 --clip 1 --client-lr 1 --server-lr 0.2 \
    --momentum 0.9 --bootstrap 400 --seed "$seed" \
    --calibrate-endpoint --fuse-trajectory
done
python experiments/dp_ftrl_independent/summarize_hybrid.py \
  --root reports/dp_ftrl_hybrid_uniform \
  --output reports/dp_ftrl_hybrid_uniform/replicated
```

The runner refuses to overwrite populated directories. The dataset digest,
mechanism settings, attack settings, and aggregate calibration selections are
recorded in each seed's `metadata.json` and CSVs. Code is in
[`hybrid_fusion.py`](../../experiments/dp_ftrl_independent/hybrid_fusion.py),
[`run_distributions.py`](../../experiments/dp_ftrl_independent/run_distributions.py),
and [`summarize_hybrid.py`](../../experiments/dp_ftrl_independent/summarize_hybrid.py).
