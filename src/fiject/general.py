# --- Imports needed by basically every file ---
from pathlib import Path
from typing import Union, Sequence, Tuple, List, Dict, Callable, Iterable, Optional
from enum import Enum
from abc import ABC, abstractmethod

import time
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Globally used variables ---
from .defaults import DEFAULTS, getColours


def newFigAx(aspect_ratio: Optional[Tuple[float,float]]) -> Tuple[plt.Figure, plt.Axes]:
    if aspect_ratio is None:
        aspect_ratio = DEFAULTS.ASPECT_RATIO
    return plt.subplots(figsize=(DEFAULTS.ASPECT_RATIO_SIZEUP*aspect_ratio[0], DEFAULTS.ASPECT_RATIO_SIZEUP*aspect_ratio[1]))


class PathHandling:
    """
    Appends _0, _1, _2 ... to file stems so that there are no collisions in the file system.
    """

    @staticmethod
    def makePath(folder: Path, stem: str, modifier: int, suffix: str) -> Path:
        return folder / f"{stem}_{modifier}{suffix}"

    @staticmethod
    def getSafeModifier(folder: Path, stem: str, suffix: str) -> int:
        modifier = 0
        path = PathHandling.makePath(folder, stem, modifier, suffix)
        while path.exists():
            modifier += 1
            path = PathHandling.makePath(folder, stem, modifier, suffix)
        return modifier

    @staticmethod
    def getSafePath(folder: Path, stem: str, suffix: str) -> Path:
        return PathHandling.makePath(folder, stem, PathHandling.getSafeModifier(folder, stem, suffix), suffix)

    @staticmethod
    def getHighestAlias(folder: Path, stem: str, suffix: str) -> Union[Path, None]:
        safe_modifier = PathHandling.getSafeModifier(folder, stem, suffix)
        if safe_modifier == 0:
            return None
        return PathHandling.makePath(folder, stem, safe_modifier-1, suffix)

    @staticmethod
    def getRawFolder():
        raw = DEFAULTS.OUTPUT_DIRECTORY / "fiject" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        return raw

    @staticmethod
    def getProductionFolder():
        prod = DEFAULTS.OUTPUT_DIRECTORY / "fiject" / "final"
        prod.mkdir(parents=True, exist_ok=True)
        return prod


class CacheMode(Enum):
    """
    Diagram data is stored in JSON files if desired by the user, and read back if desired by the user.
    The user likely wants one of the behaviours grouped below.

    | Exists | Read | Write | What does this mean?                                                  |
    | ---    | ---  | ---   | ---                                                                   |
    | no     | no   | no    | always compute never cache; good for examples or easy experiments     |
    | yes    | no   | no    | idem                                                                  |

    | no     | no   | yes   | equivalent to first run; always-write                                 |
    | yes    | no   | yes   | re-compute and refresh cache; always-write                            |

    | no     | yes  | yes   | most common situation during the first run                            |
    | yes    | yes  | no    | most common situation after 1 run                                     |

    | no     | yes  | no    | cache miss that you won't correct (can't know that it'll miss though) |

    | yes    | yes  | yes   | makes no sense because you are writing what you just read             |
    """
    # Don't read, don't write. Pretend like there is no cache.
    NONE = 1
    # Don't read, but always write, regardless of whether a file already exists. Useful for prototyping. Note: there is no "do read and always refresh" because if you want to refresh using what you read, you should use a different file.
    WRITE_ONLY = 2
    # Read but never write. Useful in a very select amount of cases, e.g. testing the reading system.
    READ_ONLY = 3
    # Read, and only write *IF* that fails. Note: there is no "don't read, but still check whether file exists and only write *if* missing" mode.
    IF_MISSING = 4


class Diagram(ABC):

    def __init__(self, name: str, caching: CacheMode=CacheMode.NONE):
        """
        Constructs a Diagram object with a name (for file I/O) and space to store data.
        The reason why the subclasses don't have initialisers is two-fold:
            1. They all looked like
                def __init__(self, name: str, use_cached: bool):
                    self.some_kind_of_dictionary = dict()
                    super().__init__(name, use_cached)
            2. It's not proper OOP to put the superclass's initialiser after the subclass's initialiser, but the way
               this initialiser is written, it is inevitable: it calls self.load() which accesses the subclass's fields,
               and if those fields are defined by the subclass, you get an error.
               While it is allowed in Python (https://stackoverflow.com/q/45722427/9352077), I avoid it.

        :param name: The file stem to be used for everything produced by this object.
        :param caching: Determines how the constructor will attempt to find the most recent data file matching the name
                        and load those data into the object. Also determines whether commit methods will store data.
        """
        self.name = name
        self.data = dict()  # All figure classes are expected to store their data in a dictionary by default, so that saving doesn't need to be re-implemented each time.
        self.clear()        # Can be used to initialise the content of self.data.
        self.creation_time = time.perf_counter()

        self.needs_computation = (caching == CacheMode.NONE or caching == CacheMode.WRITE_ONLY)
        self.will_be_stored    = (caching == CacheMode.WRITE_ONLY)
        if caching == CacheMode.READ_ONLY or caching == CacheMode.IF_MISSING:
            already_exists = False

            # Find file, and if you find it, try to load from it.
            cache_path = PathHandling.getHighestAlias(PathHandling.getRawFolder(), self.name, ".json")
            if cache_path is not None:  # Possible cache hit
                try:
                    self.load(cache_path)
                    print(f"Successfully preloaded data for diagram '{self.name}'.")
                    already_exists = True
                except Exception as e:
                    print(f"Could not load cached diagram '{self.name}':", e)

            if not already_exists:  # Cache miss
                self.needs_computation = True
                self.will_be_stored    = (caching == CacheMode.IF_MISSING)

    ### STATIC METHODS (should only be used if the non-static methods don't suffice)

    @staticmethod
    def safeFigureWrite(stem: str, suffix: str, figure, show=False):
        """
        Write a matplotlib figure to a file. For best results, use suffix=".pdf".
        The write is "safe" because it searches for a file name that doesn't exist yet, instead of overwriting.
        """
        if show:
            plt.show()  # Somtimes matplotlib hangs on savefig, and showing the figure can "slap the TV" to get it to work.
        print(f"Writing figure {stem} ...")
        figure.savefig(PathHandling.getSafePath(PathHandling.getProductionFolder(), stem, suffix).as_posix(), bbox_inches='tight')

    @staticmethod
    def safeDatapointWrite(stem: str, data: dict):
        """
        Write a json of data points to a file. Also safe.
        """
        print(f"Writing json {stem} ...")
        with open(PathHandling.getSafePath(PathHandling.getRawFolder(), stem, ".json"), "w") as file:
            json.dump(data, file)

    ### IMPLEMENTATIONS

    def exportToPdf(self, fig, stem_suffix: str=""):
        Diagram.safeFigureWrite(stem=self.name + stem_suffix, suffix=".pdf", figure=fig)

    def save(self, metadata: dict=None):
        Diagram.safeDatapointWrite(stem=self.name, data={
            "time": {
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                "start-to-finish-secs": round(time.perf_counter() - self.creation_time, 2),
            },
            "metadata": metadata or dict(),
            "data": self._save()
        })

    def load(self, json_path: Path):
        if not json_path.suffix == ".json" or not json_path.is_file():
            raise ValueError(f"Cannot open JSON: file {json_path.as_posix()} does not exist.")

        with open(json_path, "r") as handle:
            object_as_dict: dict = json.load(handle)

        if "data" not in object_as_dict:
            raise KeyError(f"Cannot read JSON file: 'data' key missing.")

        self._load(object_as_dict["data"])

    ### INSTANCE METHODS (can be overridden for complex objects whose data dictionaries aren't JSON-serialisable and/or have more state and fields)

    def clear(self):
        """
        Reset all data in the object.
        """
        self.data = dict()

    def _save(self) -> dict:
        """
        Serialise the object.
        """
        return self.data

    def _load(self, saved_data: dict):
        """
        Load object from the dictionary produced by _save().
        It is recommended to override this with appropriate sanity checks!
        """
        self.data = saved_data


class ProtectedData:
    """
    Use this in any commit method to protect data like this:
        def commit(...):
            with ProtectedData(self):
                ...
                self.exportToPdf(fig)
    If the user demanded that those data be cached, this will be done immediately.
    If an error occurs before (or during) the export, caching is also done, AND the exception is swallowed;
    this way, if you are calling multiple commits in a row, the other commits can still be reached even if one fails.

    Why not just wrap this commit method in a try...except in the abstract class? Two reasons:
        1. A commit method has many graphical arguments, and they differ from subclass to subclass. The user would have
           to call the wrapper, but there is no way to let PyCharm autocomplete the arguments in that case. A wrapper
           would need *args and **kwargs, and if you use a decorator, PyCharm automatically shows *args and **kwargs.
        2. Some classes have different commit methods (e.g. histograms to violin plot vs. boxplot). Hence, the abstract
           class can't anticipate the method name.
    """

    def __init__(self, protected_figure: Diagram, metadata: dict=None):
        self.protected_figure = protected_figure
        self.metadata = metadata or dict()

    def __enter__(self):
        if self.protected_figure.will_be_stored:
            self.protected_figure.save(metadata=self.metadata)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:  # EXITED WITH ERROR!
            if not self.protected_figure.will_be_stored:  # If it hasn't been stored already: do it now.
                self.protected_figure.save(metadata=self.metadata)

            print(f"Error writing figure '{self.protected_figure.name}'. Often that's a LaTeX issue (illegal characters somewhere?). ")
            print(f"Don't panic: your datapoints were cached, but you may have to go into the JSON to fix things.")
            print("Here's the exception, FYI:")
            print("==========================")
            time.sleep(0.5)
            import traceback
            traceback.print_exception(exc_type, exc_val, exc_tb)
            time.sleep(0.5)
            print("==========================")

        # Swallow exception
        return True
