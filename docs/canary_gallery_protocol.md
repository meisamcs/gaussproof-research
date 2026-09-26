# Canary gallery protocol

## Question

This experiment asks whether a white-box observer can identify a real sensitive
record from repeated high-noise DP-SGD gradient releases. It deliberately
separates three claims:

1. **Membership detection:** decide whether a specified record contributed.
2. **Closed-gallery record recovery:** identify the contributor from a declared
   set of candidate images.
3. **Open reconstruction:** synthesize or recover a record that is absent from
   the attacker's candidate set.

The implemented experiment establishes only the second claim and measures an
open-gallery control for the third. Selecting a known image is record linkage;
it is not pixel synthesis.

## Canary specification

The canaries are ordinary, real MNIST images. They are not synthetic patterns,
watermarks, outliers, or modified digits.

- A stratified 32-image gallery is drawn from the held-out `test` split.
- The first ten gallery records, one for each digit, are the possible true
  private contributors.
- Gallery records are disjoint from the 3,000-image public pretraining split,
  64-image public background split, and 2,000-image private population split.
- Each true canary has two independently sampled trajectories in each data
  distribution.
- At each training round, the true canary is included independently with
  probability 0.5. The remaining records come from the private population.
- The canary identity, inclusion decisions, batch members, and sampled Gaussian
  noise are hidden from the attacker. They are retained only as ground truth.

This construction provides an exact answer to “which sensitive record was
used?” while preserving natural image and gradient statistics. Canary recovery
is important because a real audit needs ground truth that cannot be obtained for
unknown private records. If a canary with the same access and sampling behavior
as a private record can be re-identified, then a candidate private record with
a known fingerprint may face the same record-linkage risk.

## DP-SGD channel and attacker access

The public CNN is pretrained for 100 updates on 3,000 disjoint MNIST records.
Every audited trajectory then evolves for 128 private updates with:

| parameter | value |
|---|---:|
| batch size | 8 |
| clip norm, `C` | 1.0 |
| noise multiplier, `sigma` | 4.0 |
| learning rate | 0.05 |
| canary inclusion probability | 0.5 |
| observed prefixes | 16, 32, 64, 128 releases |

For round `t`, the released averaged gradient is

```text
y_t = (sum of clipped per-record gradients + N(0, sigma^2 C^2 I)) / B.
```

The white-box observer receives `y_t`, the pre-update model state, all 32
labeled gallery images, and a disjoint public background set. It can therefore
compute each candidate's current clipped gradient fingerprint. This is the same
strong gradient/model access assumed by the trajectory attack; the observer
does not see the private batch composition.

Let `f_{j,t}` be candidate `j`'s clipped fingerprint, `b_t` the public estimate
of the mean background gradient, and `s_{j,t}=(f_{j,t}-b_t)/B`. Under the
Gaussian shift approximation, one-round evidence for candidate `j` is

```text
ell_{j,t} = <s_{j,t}, y_t-b_t>/v - ||s_{j,t}||^2/(2v),
v = (sigma C/B)^2.
```

Because the canary is included with probability `q=0.5`, the primary score
accumulates the Bernoulli-mixture evidence

```text
S_j(T) = sum_t log((1-q) + q exp(ell_{j,t})).
```

The highest-scoring gallery image is returned. The computation is polynomial
time, `O(TKd)`, for `T` releases, `K` candidates and `d` gradient coordinates.
No diffusion model is required for this matched-filter accumulation.

## Distribution tests

Four private label distributions are evaluated with paired noise seeds:

- `balanced`: uniform digit probability.
- `long_tail`: digit probability proportional to `1/(digit+1)`.
- `canary_rare`: the canary's class has probability 0.05; the other classes
  share 0.95.
- `canary_dominant`: the canary's class has probability 0.70; the other classes
  share 0.30.

The primary observer uses a public background distribution matched to the
private distribution. A robustness observer always uses a balanced public
background. The mismatch comparison tests whether apparent recovery comes from
an individual fingerprint or a coarse label-distribution shift.

## Metrics and controls

- Exact top-1, top-5 and mean reciprocal rank measure closed-gallery recovery.
- Predicted-label agreement measures class leakage.
- Same-class top-1 ranks the true image only against gallery images of the same
  digit, helping separate class leakage from individual identity leakage.
- In the open-gallery control, the true image is removed before choosing the
  highest-scoring alternative. Label agreement and pixel MSE are then reported.

For the fixed gallery, random exact top-1 is 3.125%, random top-5 is 15.625%,
and random MRR is 0.127. An oracle that knows only the correct digit and chooses
uniformly among same-digit gallery images has 31.7% expected exact top-1. The
random open-gallery controls are 7.1% label agreement and 0.147 pixel MSE.

Confidence intervals use 2,000 cluster-bootstrap repetitions over the ten true
canary identities, retaining both trajectory repetitions within a sampled
identity. With ten identities and two repetitions, the intervals remain wide
and the results are a pilot rather than a population guarantee.

## Claim boundary

Exact recovery from the 32-image gallery is meaningful re-identification when
an attacker has an auxiliary candidate list, such as a suspected patient or
user roster. It does not show that the pixels of an arbitrary, previously
unknown training record can be reconstructed. The open-gallery control removes
the target and tests the nearest available alternative; it is still retrieval,
not free-form generation. A future reconstruction claim requires the target to
be absent from every attack prior, followed by latent or pixel optimization and
comparison with random and same-class baselines.

