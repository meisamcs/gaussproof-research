# Does sequence access improve reconstruction?

Completed a matched LSTM ablation on the existing MNIST DP-SGD trajectories. All modes use identical architecture, initialization seeds, training probes, 1,200 optimizer updates, and public calibration selection. The public representation and image generator are frozen. Only observation access/order changes.

Each mode receives the same current-checkpoint fingerprint; other checkpoint fingerprints are excluded even from preprocessing. This isolates the added value of surrounding noisy releases given the current public context.

24 models trained: four access modes × three initialization seeds × two noise settings. Evaluation uses the same eight private 16-round trajectories per setting. Shuffled errors average five permutations, preserving the current release.

| Access setting | Image MSE, σ=0.05 | Image MSE, σ=0.5 |
|---|---:|---:|
| Current only | 0.05447 | 0.07265 |
| Past + current | 0.05457 | 0.07334 |
| Full sequence | 0.05503 | 0.07361 |
| Shuffled trained | 0.05422 | 0.07230 |
| Full, shuffled test | 0.05508 | 0.07331 |

Lower error is better. Dots in the figure show means for three independently initialized reconstructors, not three independent datasets.

## Primary paired comparison: full sequence versus current release

- σ=0.05, pixel_mse: full-sequence improvement -1.02%; absolute gain -0.000556, descriptive 95% bootstrap interval [-0.001234, +0.000248]. Positive in 1/3 training seeds.
- σ=0.05, gradient_error: full-sequence improvement -9.29%; absolute gain -0.009086, descriptive 95% bootstrap interval [-0.011330, -0.007189]. Positive in 0/3 training seeds.
- σ=0.5, pixel_mse: full-sequence improvement -1.33%; absolute gain -0.000963, descriptive 95% bootstrap interval [-0.002450, +0.000297]. Positive in 0/3 training seeds.
- σ=0.5, gradient_error: full-sequence improvement +0.92%; absolute gain +0.003836, descriptive 95% bootstrap interval [-0.003858, +0.011799]. Positive in 2/3 training seeds.

## Does order matter?

- σ=0.05: ordered full-model image improvement over shuffled: -1.49%; positive in 0/3 seeds. Absolute-gain interval [-0.001324, -0.000265].
- σ=0.05: ordered full-model image improvement over full_test_shuffled: +0.09%; positive in 1/3 seeds. Absolute-gain interval [-0.000315, +0.000507].
- σ=0.5: ordered full-model image improvement over shuffled: -1.82%; positive in 0/3 seeds. Absolute-gain interval [-0.002087, -0.000489].
- σ=0.5: ordered full-model image improvement over full_test_shuffled: -0.40%; positive in 0/3 seeds. Absolute-gain interval [-0.000869, +0.000270].

## Limits

- These are the earlier C=1, B=2, sigma=0.05/0.5 feasibility conditions, not tiny-epsilon demonstrations.
- The current fingerprint can already contain information from past model updates. Conclusions concern additional explicit sequence access given that context.
- Shuffling preserves other observations but destroys their temporal positions. The separately trained shuffled model and full-model test-time shuffle diagnose different effects; the latter can suffer distribution shift.
- All eight target runs reuse one private pool; all reconstructors share the same public representation and decoder. Crossed seed/run bootstrap intervals are descriptive, not population-level guarantees.
- Image errors use the same fixed generator and optimal two-image assignment. Gradient errors include discarded representation components. More information need not help a finite trained estimator.
- No private outcomes were used for model selection. No new DP-SGD trajectories were generated for this comparison.

See ../../docs/sequence_ablation_protocol.md for exact masks, normalization, shuffle construction, and reproduction.
