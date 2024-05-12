from tst.preamble import *

from math import sqrt
import numpy.random as npr
from fiject import LineGraph, StochasticLineGraph, CacheMode


def test_slg():
    RNG = npr.default_rng(0)

    slg = StochasticLineGraph("test", caching=CacheMode.WRITE_ONLY, overwriting=True)
    for x in range(100):
        for _ in range(100):
            slg.addSample("a", x, RNG.normal(loc=1+0.1*x, scale=0.01+sqrt(0.1*x)), weight=1)

    slg.commit(StochasticLineGraph.ArgsGlobal(uncertainty_opacity=0.2), LineGraph.ArgsPerLine(show_line=True, show_points=False))

    slg = StochasticLineGraph("test", caching=CacheMode.READ_ONLY, overwriting=True)
    slg.commit(StochasticLineGraph.ArgsGlobal(uncertainty_opacity=0.2), LineGraph.ArgsPerLine(show_line=True, show_points=False))



if __name__ == "__main__":
    test_slg()
