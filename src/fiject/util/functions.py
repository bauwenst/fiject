import numpy as np


def weightedMean(samples: np.ndarray, weights: np.ndarray=None):
    if weights is None:
        weights = np.ones_like(samples)
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)

    return 1/np.sum(weights) * np.sum(weights * samples)


def weightedVariance(samples: np.ndarray, weights: np.ndarray=None, ddof: int=1):  # S² with ddof=1
    if weights is None:
        weights = np.ones_like(samples)
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)

    mu = weightedMean(samples, weights)
    n  = len(samples)
    return 1/(n - ddof)*np.sum(weights * (n/np.sum(weights)) * (samples - mu)**2)  # You can't subtract the ddof from sum(weights), so you need to ensure that the weights have been renormalised so as to sum to n.
