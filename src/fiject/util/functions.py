import numpy as np


def weightedMean(samples: np.ndarray, weights: np.ndarray=None):
    if weights is None:
        weights = np.ones_like(samples)
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)

    return float(1/np.sum(weights) * np.sum(weights * samples))


def weightedVariance(samples: np.ndarray, weights: np.ndarray=None, ddof: int=1):
    """
    Unbiased sample variance S².

    Note that if the weights are not integers, the Bessel correction 1/(sum(w)-1) is no longer the correct factor that causes
    the estimator to be unbiased. Some say you should use 1/(sum(w)*(n-1)/n) as the correction factor, but this does NOT
    seem to be founded in the requirement that E[S^2] = E[(X - E[X])^2]. https://stats.stackexchange.com/q/6534/360389
    """
    if weights is None:
        weights = np.ones_like(samples)
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)

    mu = weightedMean(samples, weights)
    n  = np.sum(weights)
    return float(1/(n - ddof)*np.sum(weights * (samples - mu)**2))
