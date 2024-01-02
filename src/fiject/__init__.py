from .general import CacheMode
from .graphs import LineGraph, MergedLineGraph
from .histos import MultiHistogram, Histogram
from .bars import Bars
from .scatter import ScatterPlot

__all__ = ["LineGraph", "MergedLineGraph", "Bars", "ScatterPlot", "MultiHistogram", "Histogram", "CacheMode"]  # What will be imported by 'from fiject import *'


# --- Global font setup (applied even if a file never imports general.py) ---
import matplotlib.font_manager
from matplotlib import rc

# Enable LaTeX support
rc('font', **{'serif': ['Computer Modern']})
rc('text', usetex=True)
