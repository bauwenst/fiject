from .general import CacheMode
from .visuals.graphs import LineGraph, MergedLineGraph
from .visuals.histos import MultiHistogram, Histogram
from .visuals.bars import Bars
from .visuals.scatter import ScatterPlot, ScatterPlot_DiscreteContinuous
from .visuals.tables import Table, ColumnStyle
from .defaults import setFijectOutputFolder, setFijectDefaults, FIJECT_DEFAULTS, FijectDefaults

__all__ = ["LineGraph", "MergedLineGraph",
           "Bars",
           "ScatterPlot", "ScatterPlot_DiscreteContinuous",
           "MultiHistogram", "Histogram",
           "Table", "ColumnStyle",
           "CacheMode", "setFijectOutputFolder"]  # What will be imported by 'from fiject import *'


# --- Global font setup (applied even if a file never imports general.py) ---
import matplotlib.font_manager
from matplotlib import rc

# Enable LaTeX support
rc('font', **{'serif': ['Computer Modern']})
rc('text', usetex=True)
