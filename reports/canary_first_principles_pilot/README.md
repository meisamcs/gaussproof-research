# First-principles participation pilot (exploratory)

This pilot extends the original exact evolving MNIST CNN fingerprint audit to
256 releases at noise multiplier 4, clip norm 1, and batch size 8. It uses
20 fixed stratified natural candidate identities, one paired IN/OUT sequence
per identity, and hidden Bernoulli participation with shared underlying
inclusion draws across \(q\). The score is the checkpoint-specific,
background-subtracted Gaussian-mixture fingerprint likelihood. A final-model
candidate-loss score is shown only as an access control; this short training
run is not a strong final-model MIA benchmark.

| Participation \(q\) | AUC at 16 releases | AUC at 256 releases | Final-model loss AUC at 256 |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.637 | 0.855 | 0.710 |
| 0.25 | 0.595 | 0.688 | 0.620 |
| 0.1 | 0.531 | 0.603 | 0.575 |

The 256-release identity-bootstrap intervals are [0.797, 0.931],
[0.649, 0.775], and [0.550, 0.695] in the displayed \(q\) order. These
intervals are descriptive: 20 identities with one sequence each are too few
for a final low-FPR claim. The 80-identity, two-sequence confirmation uses
different identities and a separate seed.

The result is consistent with repeated high-participation signal accumulation.
It does not show that increasing Gaussian noise increases leakage, that the
likelihood beats an equal-access raw projection, or that a candidate can be
reconstructed without an auxiliary record or gallery.

See the [summary](summary.csv), [inclusion diagnostics](inclusion_summary.csv),
[endpoint comparison](trajectory_vs_endpoint.csv), [completion manifest](completion.json),
and [vector figure](q_sensitivity_endpoint.pdf). Reproduce with
`configs/canary_first_principles_pilot.json` and the command in
`docs/canary_q_sensitivity_protocol.md`, substituting this config and a new
output directory.
