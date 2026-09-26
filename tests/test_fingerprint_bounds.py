import unittest
import numpy as np
from scipy.stats import norm
from gaussproof.fingerprint_bounds import detection_power,minimum_sigma,posterior_probability,binary_mse_lower_bound


class FingerprintBoundTests(unittest.TestCase):
    def test_noise_threshold_exactly_attains_power_cap(self):
        energy=np.array([.1,1.,2.]);sigma=minimum_sigma(energy)
        np.testing.assert_allclose(detection_power(np.sqrt(16*energy)/sigma),.05,atol=1e-12)
        self.assertAlmostEqual(float(detection_power(0)),.01)

    def test_risk_bound_and_noise_monotonicity(self):
        mu=np.array([0.,.1,1.,5.])
        self.assertTrue(np.all(np.diff(detection_power(mu))>0))
        self.assertTrue(np.all(np.diff(binary_mse_lower_bound(mu))<0))
        self.assertEqual(float(binary_mse_lower_bound(0)),.25)
        # Deterministic quadrature approximates Bayes MMSE independently of Monte Carlo.
        z=np.linspace(-12,12,50001);density=norm.pdf(z)
        for m in mu:
            p0=posterior_probability(z,m);p1=posterior_probability(z+m,m)
            risk=np.trapz(.5*(p0*p0+(1-p1)**2)*density,z)
            self.assertGreaterEqual(risk+1e-10,float(binary_mse_lower_bound(m)))
            self.assertLessEqual(risk,.250000001)
