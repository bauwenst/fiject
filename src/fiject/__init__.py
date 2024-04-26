from .general import CacheMode
from .visuals.graphs import LineGraph, MergedLineGraph
from .visuals.histos import MultiHistogram, Histogram
from .visuals.bars import Bars, HistoBars
from .visuals.scatter import ScatterPlot, ScatterPlot_DiscreteContinuous
from .visuals.tables import Table, ColumnStyle
from .defaults import setFijectOutputFolder, FIJECT_DEFAULTS

__all__ = ["LineGraph", "MergedLineGraph",
           "Bars", "HistoBars",
           "ScatterPlot", "ScatterPlot_DiscreteContinuous",
           "MultiHistogram", "Histogram",
           "Table", "ColumnStyle",
           "CacheMode", "setFijectOutputFolder", "FIJECT_DEFAULTS"]  # What will be imported by 'from fiject import *'


# --- Global font setup (applied even if a file never imports general.py) ---
import matplotlib.font_manager
from matplotlib import rc

# Enable LaTeX support if available (https://stackoverflow.com/a/75478997/9352077)
import shutil

if shutil.which('latex'):
    rc('font', **{'serif': ['Computer Modern']})
    rc('text', usetex=True)
