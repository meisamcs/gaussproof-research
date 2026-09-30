# Paired trajectory versus final-model attack replication

These are completed exact DP-SGD MNIST CNN experiments at clipping norm 1,
batch size 8, noise multiplier 4, and 64 or 128 releases. For every condition,
100 stratified natural target identities were evaluated in paired present and
absent runs. The first two identities per digit (20 total) calibrated attack
variants; the remaining **80 per condition** were held out. Thirty-two OUT
reference DP-SGD models and 128 separate public population queries support
offline LiRA and RMIA. Those attacks receive only the final checkpoint,
candidate, and public reference predictions. The trajectory scores additionally
receive every intermediate checkpoint and noisy update. The raw alignment is
a simple same-access control, **not** a reproduction of RERO's full informed
reconstruction algorithm.

The original learning rate 0.05 produced poor final-model utility. We
therefore screened four learning rates on calibration identities and public
utility only (`lr_screen.csv`) and selected 0.005 for an **exploratory**
higher-utility replication. No evaluation identity was used for that choice.
At 128 releases, accuracy on the 128 disjoint public query images was 0.244
at rate 0.05 and 0.662 at rate 0.005. The attack comparison changes sharply:

| Learning rate | Participation | q-aware trajectory AUC | Raw alignment AUC | LiRA AUC | RMIA AUC | Trajectory − best calibrated endpoint, paired 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.5 | 0.810 | 0.784 | 0.693 | 0.667 | +0.117 [0.070, 0.170] |
| 0.05 | 0.1 | 0.563 | 0.562 | 0.537 | 0.531 | +0.027 [-0.002, 0.057] |
| 0.005 | 0.5 | 0.818 | 0.831 | 0.830 | 0.782 | -0.012 [-0.043, 0.020] |
| 0.005 | 0.1 | 0.574 | 0.583 | 0.580 | 0.575 | -0.006 [-0.022, 0.011] |

Every AUC in the table is on the **same 80 unseen identities** under its
condition. LiRA and RMIA variants were selected solely on the 20 calibration
identities. Paired confidence intervals resample identity pairs, preserving
the relationship between a target's present and absent trajectories. At
rate 0.005, simple raw trajectory alignment is numerically above the q-aware
likelihood, but its paired gain over LiRA is also not clearly nonzero. A
calibration-selected linear fusion of LiRA and the q-aware score chooses
**zero trajectory weight** at q=0.5,T=128. Fusion with raw alignment has AUC
0.835 versus LiRA's 0.830, paired gain interval [-0.001, 0.010]. The
current evidence therefore does **not** establish a useful extra trajectory
signal beyond a strong final-model attack when the model retains useful
accuracy.

At q=0.1 the methods are weak, and an ordinary-record proxy q=0.004 was
near chance in the earlier sensitivity pilot. The strong q=0.5 condition
corresponds to about 64 expected appearances in 128 updates, far above
ordinary minibatch participation. These experiments are not evidence that
increasing noise worsens privacy; the conservative no-amplification bound at
128 releases is epsilon 43.14 for delta 1e-5. AUC is not an empirical DP
lower bound. The 5% FPR threshold is calibrated on only 20 negative
identities, and achieved holdout FPR is reported beside TPR in `summary.csv`;
1% calibrated FPR cannot be resolved with this calibration size.

The two subdirectories contain aggregate score summaries, variant choices,
paired differences, conditions, hybrid results, provenance, and PDF/PNG
figures. Private score rows and reference predictions remain in ignored
`runs/`. Reproduction commands and split details are in
`docs/trajectory_endpoint_replication_protocol.md`.
