# --- Imports needed by basically every file ---
from pathlib import Path
from typing import Union, Sequence, Tuple, List, Dict, Callable, Iterable, Optional
from typing_extensions import Self
from enum import Enum
from abc import ABC, abstractmethod

import time
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Globally used variables ---
from .defaults import FIJECT_DEFAULTS, niceColours, cycleNiceColours, cycleRainbowColours


AspectRatio = Tuple[float,float]
def newFigAx(aspect_ratio: Optional[AspectRatio]) -> Tuple[plt.Figure, plt.Axes]:
    if aspect_ratio is None:
        aspect_ratio = FIJECT_DEFAULTS.ASPECT_RATIO
    return plt.subplots(figsize=(FIJECT_DEFAULTS.ASPECT_RATIO_SIZEUP * aspect_ratio[0], FIJECT_DEFAULTS.ASPECT_RATIO_SIZEUP * aspect_ratio[1]))


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
        raw = FIJECT_DEFAULTS.OUTPUT_DIRECTORY / "fiject" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        return raw

    @staticmethod
    def getProductionFolder():
        prod = FIJECT_DEFAULTS.OUTPUT_DIRECTORY / "fiject" / "final"
        prod.mkdir(parents=True, exist_ok=True)
        return prod


class CacheMode(Enum):
    """
    Data are stored in JSON files if desired by the user, and read back if desired by the user.
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


class ExportMode(Enum):
    """
    Matplotlib keeps figures in memory unless they are destroyed explicitly. We allow users to get Matplotlib objects
    when they commit a `Visual`, but then the user has to destroy them themselves. When they don't need the objects and
    just want the export, we destroy the figure for them.
    """
    SAVE_ONLY = 1
    RETURN_ONLY = 2
    SAVE_AND_RETURN = 3


class Visual(ABC):

    def __init__(self, name: str, caching: CacheMode=CacheMode.NONE, overwriting: bool=False):
        """
        Constructs a `Visual` object with a name (for file I/O) and space to store data.
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
        :param overwriting: Whether to overwrite the youngest found versions of the files to save.
        """
        self.raw_name = name

        self.data  = dict()  # All figure classes are expected to store their data in a dictionary by default, so that saving doesn't need to be re-implemented each time.
        self.cache = dict()  # For runtime acceleration if you want it. Is not stored.
        self.clear()        # Can be used to initialise the content of self.data.
        self.creation_time = time.perf_counter()

        self.needs_computation = (caching == CacheMode.NONE or caching == CacheMode.WRITE_ONLY)
        self.will_be_stored    = (caching == CacheMode.WRITE_ONLY)
        self._overwrite = overwriting
        self._caching   = caching
        if caching == CacheMode.READ_ONLY or caching == CacheMode.IF_MISSING:
            already_exists = False

            # Find file, and if you find it, try to load from it.
            cache_path = PathHandling.getHighestAlias(PathHandling.getRawFolder(), self.name, ".json")
            if cache_path is not None:  # Possible cache hit
                try:
                    metadata = self.load(cache_path)
                    seconds = int(metadata['time']['start-to-finish-secs'])
                    print(f"Successfully preloaded data for visual '{self.name}' sparing you {seconds//60}m{seconds%60}s of computation time.")
                    already_exists = True
                except Exception as e:
                    print(f"Could not load cached visual '{self.name}':", e)

            if not already_exists:  # Cache miss
                self.needs_computation = True
                self.will_be_stored    = (caching == CacheMode.IF_MISSING)

    ### STATIC METHODS (should only be used if the non-static methods don't suffice)

    @staticmethod
    def writeFigure(figure: plt.Figure, stem: str, suffix: str, overwrite_if_possible: bool=False, show: bool=False) -> Path:
        """
        Write a matplotlib figure to a file. For best results, use suffix=".pdf".

        :param overwrite_if_possible: If False, doesn't overwrite existing files and hence preserves them.
                                      If True and the old file is still locked by e.g. Acrobat, will pretend this is False.
        """
        if show:
            plt.show()  # Sometimes matplotlib hangs on savefig, and showing the figure can "slap the TV" to get it to work.

        def writeGivenFigure(try_this_path: Path):
            figure.savefig(try_this_path.as_posix(), bbox_inches='tight', dpi=FIJECT_DEFAULTS.DPI_IF_NOT_PDF)  # DPI is ignored for PDF output.

        print(f"Writing figure {stem} ...")
        return Visual.write(PathHandling.getProductionFolder(), stem, suffix, overwrite_if_possible, writeGivenFigure)

    @staticmethod
    def writeDictionary(data: dict, stem: str, overwrite_if_possible: bool=False) -> Path:
        """
        Write a json of data points to a file.

        :param overwrite_if_possible: See writeFigure().
        """
        def writeGivenDict(try_this_path: Path):
            with open(try_this_path, "w", encoding="utf-8") as file:
                json.dump(data, file)

        print(f"Writing dictionary {stem} ...")
        return Visual.write(PathHandling.getRawFolder(), stem, ".json", overwrite_if_possible, writeGivenDict)

    @staticmethod
    def writeLines(lines: Iterable[str], stem: str, suffix: str=".txt", overwrite_if_possible: bool=False) -> Path:
        def writeGivenLines(try_this_path: Path):
            with open(try_this_path, "w", encoding="utf-8") as file:
                file.writelines(map(lambda line: line + "\n"*(not line.endswith("\n")), lines))

        print(f"Writing lines {stem} ...")
        return Visual.write(PathHandling.getProductionFolder(), stem, suffix, overwrite_if_possible, writeGivenLines)

    @staticmethod
    def write(folder: Path, stem: str, suffix: str,
              overwrite_if_possible: bool, save_function: Callable[[Path],None]) -> Path:
        existing_path = PathHandling.getHighestAlias(folder, stem, suffix)
        safe_path     = PathHandling.getSafePath(folder, stem, suffix)
        if overwrite_if_possible and existing_path is not None:
            try:
                save_function(existing_path)
                return existing_path
            except:  # Probably open in some kind of viewer.
                save_function(safe_path)
                return safe_path
        else:
            save_function(safe_path)
            return safe_path

    ### IMPLEMENTATIONS

    @property
    def name(self) -> str:
        name = self.raw_name
        if FIJECT_DEFAULTS.GLOBAL_STEM_PREFIX and not name.startswith(FIJECT_DEFAULTS.GLOBAL_STEM_PREFIX + "_"):
            name = FIJECT_DEFAULTS.GLOBAL_STEM_PREFIX + "_" + name
        if FIJECT_DEFAULTS.GLOBAL_STEM_SUFFIX and not name.endswith("_" + FIJECT_DEFAULTS.GLOBAL_STEM_SUFFIX):
            name = name + "_" + FIJECT_DEFAULTS.GLOBAL_STEM_SUFFIX
        return name

    def exportToPdf(self, fig, export_mode: ExportMode=ExportMode.SAVE_ONLY, stem_suffix: str=""):
        if export_mode != ExportMode.RETURN_ONLY:  # if export_mode != don't save
            Visual.writeFigure(figure=fig, stem=self.name + stem_suffix, suffix="." + FIJECT_DEFAULTS.RENDERING_FORMAT, overwrite_if_possible=self._overwrite)
        if export_mode == ExportMode.SAVE_ONLY:  # if export_mode == don't return
            plt.close(fig)

    def checkpoint(self, metadata: dict=None) -> dict:
        """Return the visual's data as a dictionary, along with metadata about timing."""
        return {
            "time": {
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                "start-to-finish-secs": round(time.perf_counter() - self.creation_time, 2),
            },
            "metadata": metadata or dict(),
            "data": self._save()  # TODO: This is very, very inefficient. You should store data not as ASCII but as a binary file or in some other compressed format.
        }

    def save(self, metadata: dict=None):
        """Checkpoint the visual and safely write it to a dictionary."""
        Visual.writeDictionary(data=self.checkpoint(metadata), stem=self.name, overwrite_if_possible=self._overwrite)

    def load(self, json_path: Path) -> dict:
        """
        Wrapper around _load() that unpacks the json file, calls _load(), and returns metadata.
        """
        if not json_path.suffix == ".json" or not json_path.is_file():
            raise ValueError(f"Cannot open JSON: file {json_path.as_posix()} does not exist.")

        with open(json_path, "r", encoding="utf-8") as handle:
            object_as_dict: dict = json.load(handle)

        if "data" not in object_as_dict:
            raise KeyError(f"Cannot read JSON file: 'data' key missing.")

        self._load(object_as_dict["data"])
        object_as_dict.pop("data")
        return object_as_dict

    ### INSTANCE METHODS (can be overridden for complex objects whose data dictionaries aren't JSON-serialisable and/or have more state and fields)

    def clear(self):
        """
        Reset all data in the object.
        """
        self.data = dict()
        self.cache = dict()

    def copy(self, name: str) -> Self:
        """
        Copy the data stored in this object into a new object with a different name.
        If you override the constructor, you also have to override this method (and not call it with super).
        """
        new = self.__class__(name, caching=self._caching, overwriting=self._overwrite)
        new.data  = deepcopy(self.data)
        new.cache = deepcopy(self.cache)
        return new

    def isEmpty(self) -> bool:
        return Visual._isEmpty(self.data)

    @staticmethod
    def _isEmpty(iterable: Iterable) -> bool:
        if isinstance(iterable, str):  # Strings have to be treated specially because a 1-character string returns itself when iterating over it.
            return len(iterable) == 0

        for value in (iterable.values() if isinstance(iterable, dict) else iterable):
            if not isinstance(value, Iterable):  # Found a value by iterating that is not an iterable itself. That's non-emptiness!
                return False
            elif not Visual._isEmpty(value):  # Found a value that is an iterable and it isn't empty itself.
                return False

        return True  # All values were empty iterables.

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

    def __init__(self, protected_figure: Visual, metadata: dict=None):
        self.protected_figure = protected_figure
        self.metadata = metadata or dict()

    def __enter__(self):
        if self.protected_figure.will_be_stored:
            self.protected_figure.save(metadata=self.metadata)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:  # EXITED WITH ERROR!
            SEP = "="*50
            print(SEP)
            print(f"Error writing figure '{self.protected_figure.name}'. Often that's a LaTeX issue (illegal characters somewhere?).")

            if not self.protected_figure.will_be_stored:  # If it hasn't been stored already: do it now.
                self.protected_figure.save(metadata=self.metadata)

            print(f"Don't panic: your datapoints were cached, but you may have to go into the JSON to fix things.")
            print("Here's the exception, FYI:")
            print(SEP)
            time.sleep(0.5)
            import traceback
            traceback.print_exception(exc_type, exc_val, exc_tb)
            time.sleep(0.5)
            print(SEP)

        # Swallow exception
        return True
