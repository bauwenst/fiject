from tst.preamble import *

from math import sqrt
import numpy.random as npr
from fiject import LineGraph, StochasticLineGraph, CacheMode, StreamingStochasticLineGraph


def test_slg():
    """
    Generate a stochastic line graph whose standard deviation widens as it advances.
    """
    RNG = npr.default_rng(0)

    slg        = StochasticLineGraph("test-stochastic", caching=CacheMode.WRITE_ONLY, overwriting=True)
    slg_stream = StreamingStochasticLineGraph("test-stochastic-streamed", caching=CacheMode.WRITE_ONLY, overwriting=True)
    for x in range(100):
        for _ in range(100):
            y = RNG.normal(loc=1+0.1*x, scale=0.001+sqrt(0.1*x))
            slg.addSample("a", x, y, weight=1)
            slg_stream.addSample("a", x, y, weight=1)

    # Test writing
    slg.commit(StochasticLineGraph.ArgsGlobal(uncertainty_opacity=0.2), LineGraph.ArgsPerLine(show_line=True, show_points=False))
    slg_stream.commit(StochasticLineGraph.ArgsGlobal(uncertainty_opacity=0.2), LineGraph.ArgsPerLine(show_line=True, show_points=False))

    # Test reading and again writing
    slg        = StochasticLineGraph(slg.name, caching=CacheMode.READ_ONLY, overwriting=True)
    slg_stream = StreamingStochasticLineGraph(slg_stream.name, caching=CacheMode.READ_ONLY, overwriting=True)
    slg.commit(StochasticLineGraph.ArgsGlobal(uncertainty_opacity=0.2), LineGraph.ArgsPerLine(show_line=True, show_points=False))
    slg_stream.commit(StochasticLineGraph.ArgsGlobal(uncertainty_opacity=0.2), LineGraph.ArgsPerLine(show_line=True, show_points=False))



if __name__ == "__main__":
    test_slg()
