"""Attack functions receive observations only, never evaluation membership."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def output_scores(probs, labels):
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1)
    truth = p[np.arange(len(p)), labels]
    ordered = np.sort(p, axis=1)
    return dict(loss=np.log(truth), confidence=ordered[:, -1],
                negative_entropy=(p*np.log(p)).sum(1), margin=ordered[:, -1]-ordered[:, -2])


def features(probs, labels):
    s = output_scores(probs, labels)
    return np.column_stack([probs, np.eye(10)[labels], *s.values()])


class LogisticAttack:
    """L2 regularized pooled, class-conditioned Shokri-style classifier."""
    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if set(np.unique(y)) != {0, 1} or not np.isfinite(x).all():
            raise ValueError('Attack fitting needs finite features and both classes')
        self.mean = x.mean(0)
        self.scale = np.maximum(x.std(0), 1e-6)
        a = np.column_stack([(x-self.mean)/self.scale, np.ones(len(x))])
        def objective(w):
            z = a @ w
            loss = np.mean(np.logaddexp(0, z)-y*z) + 0.005*np.dot(w[:-1], w[:-1])
            grad = a.T @ (expit(z)-y)/len(y)
            grad[:-1] += 0.01*w[:-1]
            return loss, grad
        fit = minimize(objective, np.zeros(a.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter':500})
        if not fit.success:
            raise RuntimeError(f'Attack optimizer failed: {fit.message}')
        self.weights = fit.x
        return self

    def score(self, x):
        return ((np.asarray(x)-self.mean)/self.scale) @ self.weights[:-1] + self.weights[-1]

    def state(self):
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist(), weights=self.weights.tolist())


def rmia(target, reference, population_target, population_reference, a=0.5, gamma=1.0):
    """Offline RMIA: calibrated OUT probabilities and population ratio comparisons.

    p(x) = ((1+a)*mean_OUT(x) + (1-a))/2, following the authors' offline
    approximation. a and gamma are fixed in the config, never tuned on evaluation.
    Arguments contain probabilities assigned to each record's true class.
    """
    if not 0 <= a <= 1 or gamma <= 0:
        raise ValueError('RMIA requires 0<=a<=1 and gamma>0')
    def ratio(t, r):
        den = ((1+a)*np.asarray(r).mean(axis=0) + (1-a))/2
        return np.asarray(t) / np.maximum(den, 1e-12)
    x = ratio(target, reference)
    z = np.sort(ratio(population_target, population_reference))
    if len(z) == 0:
        raise ValueError('RMIA requires independent population samples')
    # The paper's dominance event is LR(x,z) >= gamma, including equality.
    return np.searchsorted(z, x/gamma, side='right')/len(z)


def trajectory_scores(dictionary, releases, penalty, iterations=80):
    """Time-varying clipped gradient dictionaries: [steps,candidates,coordinates].

    RERO-style first-order alignment and a bounded nonnegative sparse decoder.
    The latter minimizes .5||D_t.T z-y_t||² + penalty*sum(z), 0<=z<=1.
    No membership labels, participation indices, or noise realization are inputs.
    """
    d = np.asarray(dictionary)
    y = np.asarray(releases, dtype=np.float64)
    if d.ndim != 3 or y.shape != (d.shape[0], d.shape[2]) or penalty < 0:
        raise ValueError('Invalid dictionary/release shapes or negative penalty')
    rero = np.einsum('tnd,td->n', d, y)/len(y)
    sparse = np.zeros(d.shape[1])
    for bank, obs in zip(d, y):
        bank = np.asarray(bank, dtype=np.float64)
        # Eigenvalues in coordinate space are cheaper than a candidates² matrix.
        lipschitz = max(float(np.linalg.eigvalsh(bank.T @ bank)[-1]), 1e-12)
        x = np.zeros(len(bank)); v = x.copy(); momentum = 1.0
        for _ in range(iterations):
            nxt = np.clip(v - (bank @ (bank.T @ v-obs)+penalty)/lipschitz, 0, 1)
            new_momentum = (1+np.sqrt(1+4*momentum*momentum))/2
            v = nxt + (momentum-1)/new_momentum*(nxt-x)
            x, momentum = nxt, new_momentum
        sparse += x
    return dict(rero=rero, gaussproof=sparse/len(y))
