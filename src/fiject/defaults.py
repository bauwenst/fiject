from typing import Tuple, Iterator
from pathlib import Path
from dataclasses import dataclass
import os
import itertools


@dataclass
class FijectDefaults:
    # Graphical defaults
    ASPECT_RATIO_SIZEUP: float
    ASPECT_RATIO: Tuple[float, float]
    GRIDWIDTH: float

    # Strings that appear in graphs (change these to the language of choice)
    LEGEND_TITLE_CLASS: str

    # File storage
    OUTPUT_DIRECTORY: Path
    GLOBAL_STEM_PREFIX: str
    GLOBAL_STEM_SUFFIX: str
    RENDERING_FORMAT: str
    DPI_IF_NOT_PDF: int


FIJECT_DEFAULTS = FijectDefaults(
    ASPECT_RATIO_SIZEUP=1.5,  # Make this LARGER to make fonts and lines SMALLER.
    ASPECT_RATIO=(4,3),
    GRIDWIDTH=0.5,
    #
    LEGEND_TITLE_CLASS="class",
    #
    OUTPUT_DIRECTORY=Path(os.getcwd()),  # Will create a fiject/ folder under this. If you have a data/ folder with input and output subfolders, you should give the path to your output data folder specifically, because I'm not going to suggest that your data/ folder should have a specific subfolder for outputs.
    GLOBAL_STEM_PREFIX="",
    GLOBAL_STEM_SUFFIX="",
    RENDERING_FORMAT="pdf",
    DPI_IF_NOT_PDF=600,
)


# def setFijectDefaults(d: FijectDefaults):  # Actually, it is a bad idea to have if you import FIJECT_DEFAULTS. After a file has imported FIJECT_DEFAULTS, that variable name will, in that file, always be bound to the object reference at that moment. Changing the object reference here won't do anything there. The only way to solve this is to have a function getFijectDefaults() to be imported instead.
#     # Inspired by logging.setLoggerClass().
#     global FIJECT_DEFAULTS
#     FIJECT_DEFAULTS = d


def setFijectOutputFolder(folder_path: Path):
    FIJECT_DEFAULTS.OUTPUT_DIRECTORY = folder_path


# Colours
from matplotlib import colors as mcolors
MPL_COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
NICE_COLORS = [
    MPL_COLORS.get("r"), MPL_COLORS.get("g"), MPL_COLORS.get("b"),
    MPL_COLORS.get("lime"), MPL_COLORS.get("darkviolet"), MPL_COLORS.get("gold"),
    MPL_COLORS.get("cyan"), MPL_COLORS.get("magenta")
]

def niceColours() -> list:  # Can be popped from
    return list(NICE_COLORS)

def cycleNiceColours() -> Iterator:  # Can be iterated repeatedly
    return itertools.cycle(NICE_COLORS)

import matplotlib.pyplot as plt
import numpy as np
def cycleRainbowColours(amount_of_points: int) -> Iterator:
    return itertools.cycle(plt.cm.rainbow(np.linspace(0, 1, amount_of_points)))  # TODO: "gist_rainbow" and "jet" are both superior, and "hsv" is cyclic.
