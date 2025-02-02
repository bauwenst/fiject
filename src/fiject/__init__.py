from .general import CacheMode, ExportMode
from .visuals.graphs import LineGraph, MergedLineGraph, StochasticLineGraph, StreamingStochasticLineGraph
from .visuals.histos import MultiHistogram, Histogram, StreamingMultiHistogram, VariableGranularityHistogram, StreamingVariableGranularityHistogram, BinSpec, BinOverlapMode
from .visuals.bars import Bars, HistoBars
from .visuals.scatter import ScatterPlot, ScatterPlot_DiscreteContinuous
from .visuals.tables import Table, ColumnStyle
from .defaults import setFijectOutputFolder, FIJECT_DEFAULTS

__all__ = ["LineGraph", "MergedLineGraph", "StochasticLineGraph", "StreamingStochasticLineGraph",
           "Bars", "HistoBars",
           "ScatterPlot", "ScatterPlot_DiscreteContinuous",
           "MultiHistogram", "Histogram", "StreamingMultiHistogram", "VariableGranularityHistogram", "StreamingVariableGranularityHistogram", "BinSpec", "BinOverlapMode",
           "Table", "ColumnStyle",
           "CacheMode", "ExportMode", "setFijectOutputFolder", "FIJECT_DEFAULTS"]  # What will be imported by 'from fiject import *'


# --- Global font setup (applied even if a file never imports general.py) ---
import matplotlib.font_manager
from matplotlib import rc

# Enable LaTeX support if available (https://stackoverflow.com/a/75478997/9352077)
import shutil

if shutil.which("latex"):
    rc("font", **{"serif": ["Computer Modern"]})
    rc("text", usetex=True)
