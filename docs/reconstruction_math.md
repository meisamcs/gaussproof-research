# What this experiment estimates

Let a target be an image x with a known digit label c. At public checkpoint theta_t,
define the full-model clipped gradient

$$h_t(x,c)=\operatorname{clip}_C(\nabla_\theta\ell(\theta_t;x,c)).$$

The target signal is the concatenation H(x,c) = [h_1; ...; h_K] in dimension D.
Public checkpoints are trained on disjoint public examples. Their gradients are
computable for any proposed input, but the target's clean gradient is not given to
the attacker. It is used by the experiment to generate releases and evaluate errors.

## Gaussian releases and sufficient statistics

In the known-background experiment, the attacker subtracts all other batch
contributions, leaving R independent releases per checkpoint:

$$Y_r=H+\sigma C Z_r,\qquad Z_r\sim\mathcal N(0,I_D).$$

For fixed H and checkpoints, the sample mean is sufficient:

$$\bar Y=H+\tau Z,\qquad \tau^2=\sigma^2C^2/R.$$

No sequence network can extract additional information from the centered residuals
Y_r - mean(Y), which are independent of H in this specified channel. This does not
extend unchanged to a changing private-training trajectory or unknown participation.

## Public learned prior

From disjoint public examples, estimate a class mean mu_c, an orthonormal basis U,
principal variances Lambda, and an isotropic residual variance v:

$$H\mid c\ \approx\ \mathcal N(\mu_c,\Sigma),\quad
\Sigma=U\Lambda U^\top+v(I-UU^\top).$$

The Gaussian posterior mean is

$$\widehat H_G=\mu_c+\Sigma(\Sigma+\tau^2 I)^{-1}(\bar Y-\mu_c).$$

It shrinks principal coordinates by lambda_j/(lambda_j + tau^2) and residual
coordinates by v/(v + tau^2). The neural estimator learns the principal-coordinate
posterior mean using public clean/noisy pairs and known class/noise features; it
uses the same analytic residual shrinkage. Calibration selects a training checkpoint
without observing any target gradients or pixels. It is not guaranteed to beat the
Gaussian estimator and is not a proven posterior for the unknown real distribution.

Under the ideal correctly specified Gaussian prior, expected squared error is

$$\operatorname{tr}[\Sigma\tau^2(\Sigma+\tau^2 I)^{-1}]
=\sum_j \frac{\lambda_j\tau^2}{\lambda_j+\tau^2}
+(D-k)\frac{v\tau^2}{v+\tau^2}.$$

This grows monotonically with tau^2, approaching the prior-only error tr(Sigma).
Raw noisy averaging has error D tau^2. A large percentage improvement relative to
averaging can coexist with little target-specific information. Hence we measure
absolute error and paired improvement over the no-release prior mean.

For a hard subspace projection, conditional mean squared error decomposes into

$$\|(I-UU^\top)(H-\mu_c)\|^2+k\tau^2.$$

The first term is prior mismatch. Stale fingerprints can increase it. A learned
low-dimensional gradient geometry helps only to the extent it transfers to targets.

## Unknown batch background

We add B-1 random public-record gradients per release. After subtracting their
expected sum, the sample mean contains additional mean-zero nuisance with covariance
(B-1) Cov(H_public)/R. The implementation approximates this covariance diagonally
in the learned basis and isotropically outside it. Public background samples can be
non-Gaussian and correlated across coordinates; this approximation is a limitation,
not an exact likelihood for the unknown-background channel.

## Exact-bank identification is a different access model

For a target-containing candidate bank H_i and a uniform prior among candidates
with the known label, Gaussian log likelihood, up to a candidate-independent term, is

$$s_i=\frac{\bar Y^\top H_i-\|H_i\|^2/2}{\tau^2}.$$

The norm correction matters unless candidate norms coincide. Argmax(s_i) is the MAP
identity rule in this model. The bank posterior mean sums H_i with softmax(s_i)
weights. Both already know all candidate images. They do not constitute recovery of
a novel image, nor may the learned-prior method claim to beat this optimal identity
rule with the same information and assumptions.

The candidate bank in this experiment is label-restricted, but contains the target
plus distractors and other target records; no membership truth or identity is passed
to the scoring function. ReRo-style alignment is a comparator, not a claimed complete
reproduction of all attacks in the reconstruction-robustness literature.

## Pixel inversion

For each estimated gradient, reconstruct x by optimizing

$$\min_{x\in[0,1]^{28\times28}}
\frac{\|H(x,c)-\widehat H\|^2}{\max(\|\widehat H\|^2,10^{-6})}
+10^{-4}\operatorname{TV}(x).$$

Every method receives the same initial images and optimization budget. Restarts are
selected using gradient mismatch, never target pixel error. A clean-gradient oracle
tests the optimizer; a public class-mean image and prior-only gradient inversion
test whether plausible digits alone explain apparent success. Smaller gradient error
does not mathematically guarantee smaller image error.

## Noise and privacy

For sigma_2 > sigma_1 and fixed H/access/release count, the noisier observation can
be generated by adding independent Gaussian noise to the less noisy observation.
Thus its optimal Bayes reconstruction risk cannot be lower. An empirical estimator
may behave nonmonotonically, but this is not a theorem that excess Gaussian noise
creates privacy leakage. Training dynamics or changing release counts are different
experiments and must be accounted for separately.

With replace-one target adjacency, each clipped-gradient release has sensitivity at
most 2C. A conservative Gaussian zCDP composition over K R releases gives

$$\rho=2KR/\sigma^2,\qquad
\epsilon=\rho+2\sqrt{\rho\log(1/\delta)}.$$

The known label is fixed auxiliary information: the privacy statement concerns
changes to the image with that label held fixed. It does not protect an explicitly
disclosed label. The composition assumes fixed public checkpoints and target-independent
public background; no subsampling amplification or private-model training guarantee
is claimed. More repeated observations improve reconstruction while spending more
privacy budget. A large noise multiplier alone is not evidence of a small epsilon.
