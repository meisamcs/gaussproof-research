# Identifiability sanity check: where joint scores can help

This is a **fully specified synthetic Gaussian-mixture model**, not a
DP-SGD run or an empirical privacy lower bound. It asks whether the current
joint mean-field decoder has any useful mechanism when several known,
nonorthogonal contributors have persistent dataset-level IN/OUT bits. For
eight canaries in dimension d, each trial draws independent fair bits and,
conditional on IN, Bernoulli(q=0.5) participation at each of 12 releases.
Each unit-norm fingerprint is a random asymmetric direction fixed across
the trajectory. The observation is the sum of active fingerprints plus
independent Gaussian noise of standard deviation one. Every method sees
the same observations and fingerprints.

We compare an independent q-aware per-canary Gaussian-mixture score, the
GAUSSPROOF joint mean-field score, and an **exact** posterior that enumerates
the eight dataset bits and per-round active subsets. Every trial makes two
IN and two OUT guesses. Results average 200 trials for each of eight
independent random dictionaries per dimension; intervals bootstrap whole
dictionaries, not individual canary decisions.

| dimension | mean effective Gram rank | independent correct / 4 | joint correct / 4 | exact correct / 4 | joint gain, paired 95% interval |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 2 | 1.84 | 2.788 | 2.905 | 2.960 | +0.117 [0.069, 0.168] |
| 3 | 2.38 | 2.811 | 3.025 | 3.058 | +0.214 [0.152, 0.275] |
| 4 | 2.96 | 3.011 | 3.182 | 3.206 | +0.171 [0.097, 0.251] |
| 8 | 4.38 | 3.236 | 3.367 | 3.366 | +0.131 [0.111, 0.152] |

Thus the joint implementation can extract information that a per-canary
score leaves unused **under its assumed model**. The exact posterior is
usually a little stronger, which makes this a useful code/model check.
The accompanying [plot](identifiability_sanity.pdf),
[per-dictionary results](per_dictionary.csv),
[summary](summary_with_ci.csv), and
[paired differences](paired_differences.csv) are reproducible with:

```bash
python -m scripts.identifiability_sanity \
  --report reports/identifiability_sanity
```

The [real MNIST confirmation](../one_run_interference_confirm/README.md)
found no such advantage. In particular, its 64 high-overlap natural
fingerprints had effective rank only about 2.55, a very different geometry
from eight random asymmetric directions. This toy result cannot be cited
as evidence of a better natural-record privacy audit, and it does not
justify a high-noise backfire claim.
