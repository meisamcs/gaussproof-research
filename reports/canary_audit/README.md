# Known-canary fingerprint audit

This report measures detection of a known canary inserted into half of the
batches over 16 noisy DP-SGD rounds. The auditor receives the full sequence and
the canary's public gradient fingerprint at every evolving checkpoint, but not
batch membership or the hidden inclusion schedule.

| σ | epsilon upper bound | sequence-sum AUC | final-release AUC | max-release AUC |
|---:|---:|---:|---:|---:|
| 0.25 | 665.553 | 0.946 | 0.612 | 0.787 |
| 1 | 70.388 | 0.864 | 0.621 | 0.744 |
| 4 | 11.597 | 0.646 | 0.531 | 0.563 |

The sequence score is the sum of per-round known-fingerprint matched scores. It
beats the final release at the two lower noise levels, supporting persistent
fingerprint detectability. It approaches chance at sigma 4. This is a known-canary
audit, not unknown-image reconstruction, and the privacy settings are weak by
modern standards because the deliberately conservative no-amplification bounds
are large.

![Canary detection AUC](../../runs/canary_audit/canary_auc.png)

See [the protocol](../../docs/canary_audit_protocol.md) and the raw [summary CSV](summary.csv). The final numbers come from the paired-seed run in `runs/canary_audit_paired`.
