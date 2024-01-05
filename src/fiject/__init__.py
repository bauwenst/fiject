from .general import CacheMode
from .visuals.graphs import LineGraph, MergedLineGraph
from .visuals.histos import MultiHistogram, Histogram
from .visuals.bars import Bars
from .visuals.scatter import ScatterPlot

__all__ = ["LineGraph", "MergedLineGraph", "Bars", "ScatterPlot", "MultiHistogram", "Histogram", "CacheMode"]  # What will be imported by 'from fiject import *'


# Configuration stuff that you should access as "import fiject; fiject.setOutputFolder(...)"
from .defaults import setOutputFolder, setAllDefaults, DEFAULTS, Defaults


# --- Global font setup (applied even if a file never imports general.py) ---
import matplotlib.font_manager
from matplotlib import rc

# Enable LaTeX support
rc('font', **{'serif': ['Computer Modern']})
rc('text', usetex=True)
