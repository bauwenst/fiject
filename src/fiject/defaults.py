from typing import Tuple
from pathlib import Path
from dataclasses import dataclass
import os


@dataclass
class Defaults:
    # Graphical defaults
    ASPECT_RATIO_SIZEUP: float
    ASPECT_RATIO: Tuple[float, float]
    GRIDWIDTH: float

    # Strings that appear in graphs (change these to the language of choice)
    LEGEND_TITLE_CLASS: str

    # File storage
    OUTPUT_DIRECTORY: Path


DEFAULTS = Defaults(
    ASPECT_RATIO_SIZEUP=1.5,  # Make this LARGER to make fonts and lines SMALLER.
    ASPECT_RATIO=(4,3),
    GRIDWIDTH=0.5,
    #
    LEGEND_TITLE_CLASS="class",
    #
    OUTPUT_DIRECTORY=Path(os.getcwd())  # Will create a fiject/ folder under this. If you have a data/ folder with input and output subfolders, you should give the path to your output data folder specifically, because I'm not going to suggest that your data/ folder should have a specific subfolder for outputs.
)


def setAllDefaults(d: Defaults):
    # Inspired by logging.setLoggerClass().
    global DEFAULTS
    DEFAULTS = d


def setOutputFolder(folder_path: Path):
    DEFAULTS.OUTPUT_DIRECTORY = folder_path


# Colours
from matplotlib import colors as mcolors
MPL_COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
NICE_COLORS = [
    MPL_COLORS.get("r"), MPL_COLORS.get("g"), MPL_COLORS.get("b"),
    MPL_COLORS.get("lime"), MPL_COLORS.get("darkviolet"), MPL_COLORS.get("gold"),
    MPL_COLORS.get("cyan"), MPL_COLORS.get("magenta")
]
def getColours():
    return list(NICE_COLORS)
