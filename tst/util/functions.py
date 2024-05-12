from fiject.util.functions import *


def test_mean_std():
    samples = [1,2,4,5,8,9]
    print(np.mean(samples), weightedMean(samples))
    print(np.std(samples, ddof=1)**2, weightedVariance(samples, ddof=1))


if __name__ == "__main__":
    test_mean_std()