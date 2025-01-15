from collections import defaultdict

from ..general import *
from .scatter import ScatterPlot

from dataclasses import dataclass

from math import floor, ceil, sqrt
import scipy
import warnings
import pandas as pd
from natsort import natsorted

import seaborn as sns
import matplotlib.legend as lgd  # Only for type-checking.
import matplotlib.ticker as tkr

from ..util.iterables import cat


class MultiHistogram(Diagram):
    """
    A histogram plots the distribution of a SINGLE variable.
    On the horizontal axis is that variable's domain. On the vertical axis is the frequency/fraction/... of each value.
    That means: you only give values of the variable, and the heights are AUTOMATICALLY computed, unlike in a graph.

    Can be seen as a cross between a graph and a bar chart.
    """

    def _load(self, saved_data: dict):
        for name, values in saved_data.items():
            if not isinstance(values, list):
                raise ValueError("Histogram data corrupted.")
            MultiHistogram.addMany(self, name, values)  # Specifically mentions the parent class to prevent using a child's method here.

    def add(self, series_name: str, x_value: float):
        if series_name not in self.data:
            self.data[series_name] = []
        self.data[series_name].append(float(x_value))

    def addMany(self, series_name: str, values: Iterable[float]):
        if series_name not in self.data:
            self.data[series_name] = []
        self.data[series_name].extend(map(float, values))  # The map() ensures you can also use numpy.float datatypes.

    def toDataframe(self):
        # You would think all it takes is pd.Dataframe(dictionary), but that's the wrong way to do it.
        # If you do it that way, you are pairing up the i'th value of each family as if it belongs to one
        # object. This is not correct, and crashes if you have different sample amounts per family.
        rows = []  # For speed, instead of appending to a dataframe, make a list of rows as dicts. https://stackoverflow.com/a/17496530/9352077
        for name, x_values in self.data.items():
            for v in x_values:
                rows.append({"value": v, FIJECT_DEFAULTS.LEGEND_TITLE_CLASS: name})
        df = pd.DataFrame(rows)
        return df if len(self.data) != 1 else df.drop(columns=[FIJECT_DEFAULTS.LEGEND_TITLE_CLASS])

    @dataclass
    class ArgsGlobal:
        binwidth: float = 1
        relative_counts: bool = False
        average_over_bin: bool = False

        do_kde: bool = True
        kde_smoothing: bool = True
        border_colour: bool = None
        fill_colour: str = None  # Note: colour=None means "use default colour", not "use no colour".
        do_hatch: bool = False

        aspect_ratio: AspectRatio = None
        x_lims: Tuple[Optional[int], Optional[int]] = None
        x_label: str = ""
        y_label: str = ""
        log_x: bool = False
        log_y: bool = False
        x_tickspacing: float = 1
        y_tickspacing: float = None
        center_ticks: bool = False

    def commit_histplot(self, binwidth: float=1, log_x=False, log_y=False,
                        relative_counts: bool=False, average_over_bin: bool=False,
                        x_lims: Tuple[Optional[float],Optional[float]]=None, aspect_ratio=None,
                        x_tickspacing: float=1, y_tickspacing: float=None, center_ticks=False,
                        do_kde=True, kde_smoothing=True,
                        border_colour=None, fill_colour=None, do_hatch=False, # Note: colour=None means "use default colour", not "use no colour".
                        x_label: str="", y_label: str="",
                        **seaborn_args):
        """
        :param x_lims: left and right bounds. Either can be None to make them automatic. These bound are the edge of
                       the figure; if you have a bar from x=10 to x=11 and you set the right bound to x=10, then you
                       won't see the bar but you will see the x=10 tick.
        :param center_ticks: Whether to center the ticks on the bars. The bars at the minimal and maximal x_lim are
                             only half-visible.
        """
        self.commitWithArgs_histplot(MultiHistogram.ArgsGlobal(
            binwidth=binwidth,
            relative_counts=relative_counts,
            average_over_bin=average_over_bin,

            do_kde=do_kde,
            kde_smoothing=kde_smoothing,
            border_colour=border_colour,
            fill_colour=fill_colour,
            do_hatch=do_hatch,

            aspect_ratio=aspect_ratio,
            x_lims=x_lims,
            x_label=x_label,
            y_label=y_label,
            log_x=log_x,
            log_y=log_y,
            x_tickspacing=x_tickspacing,
            y_tickspacing=y_tickspacing,
            center_ticks=center_ticks
        ), **seaborn_args)

    def commitWithArgs_histplot(self, diagram_options: ArgsGlobal, **seaborn_args):
        do = diagram_options
        with ProtectedData(self):
            if do.relative_counts:
                if do.average_over_bin:
                    mode = "density"  # Total area is 1.
                else:
                    mode = "percent"  # Total area is 100.
            else:
                if do.average_over_bin:
                    mode = "frequency"
                else:
                    mode = "count"

            df = self.toDataframe()
            if len(self.data) != 1:
                legend_title = FIJECT_DEFAULTS.LEGEND_TITLE_CLASS
                print(df.groupby(FIJECT_DEFAULTS.LEGEND_TITLE_CLASS).describe())
            else:
                legend_title = None
                print(df.value_counts())

            fig, ax = newFigAx(do.aspect_ratio)
            if not do.log_x:
                sns.histplot(df, ax=ax, x="value", hue=legend_title,  # x and hue args: https://seaborn.pydata.org/tutorial/distributions.html
                             binwidth=do.binwidth, binrange=(floor(df["value"].min() / do.binwidth) * do.binwidth,
                                                             ceil(df["value"].max() / do.binwidth) * do.binwidth),
                             discrete=do.center_ticks, stat=mode, common_norm=False,
                             kde=do.do_kde, kde_kws={"bw_adjust": 10} if do.kde_smoothing else seaborn_args.pop("kde_kws", None),  # Btw, do not use displot: https://stackoverflow.com/a/63895570/9352077
                             color=do.fill_colour, edgecolor=do.border_colour,
                             **seaborn_args)  # Do note use displot: https://stackoverflow.com/a/63895570/9352077
            else:
                sns.histplot(df, ax=ax, x="value", hue=legend_title,
                             log_scale=True,
                             discrete=do.center_ticks, stat=mode, common_norm=False,
                             color=do.fill_colour, edgecolor=do.border_colour,
                             **seaborn_args)  # Do note use displot: https://stackoverflow.com/a/63895570/9352077

            # Cross-hatching
            if do.do_hatch:
                # Note that this is actually quite difficult for multi-histograms: surprisingly, you can't pass all the
                # hatch patterns you want to sns.histplot, only one. Hence, we need a hack, see
                #   https://stackoverflow.com/a/40293705/9352077
                #   and https://stackoverflow.com/a/76233802/9352077
                HATCHING_PATTERNS = ['/', '\\', '.', '*', '+', '|', '-', 'x', 'O', 'o']  # https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_style_reference.html
                legend: lgd.Legend = ax.get_legend()
                for pattern, bar_collection, legend_handle in zip(HATCHING_PATTERNS, ax.containers, legend.legendHandles[::-1]):  # FIXME: .legend_handles in newer versions of matplotlib.
                    legend_handle.set_hatch(pattern)
                    for bar in bar_collection:
                        bar.set_hatch(pattern)

            # Axes
            ax.set_xlabel(do.x_label)
            ax.set_ylabel(do.y_label + r" [\%]" * (mode == "percent" and do.y_label != ""))
            if do.x_lims:
                if do.x_lims[0] is not None and do.x_lims[1] is not None:
                    ax.set_xlim(do.x_lims[0], do.x_lims[1])
                elif do.x_lims[0] is not None:
                    ax.set_xlim(left=do.x_lims[0])
                elif do.x_lims[1] is not None:
                    ax.set_xlim(right=do.x_lims[1])
                else:  # You passed (None,None)...
                    pass

            # Weird tick spacing hack that somehow works https://stackoverflow.com/a/44525175/9352077
            if not do.log_x:
                ax.xaxis.set_major_locator(tkr.MultipleLocator(do.x_tickspacing))
                ax.xaxis.set_major_formatter(tkr.ScalarFormatter())

            if not do.log_y:
                if do.y_tickspacing:
                    ax.yaxis.set_major_locator(tkr.MultipleLocator(do.y_tickspacing))
                    ax.yaxis.set_major_formatter(tkr.ScalarFormatter())
            else:
                ax.set_yscale("log")

            # ax.set_yticklabels(np.arange(0, max(self.x_values), ytickspacing, dtype=int))  # Don't do this. It literally overwrites existing ticks, rather than placing more of them, so the result is mathematically wrong.
            ax.set_axisbelow(True)
            ax.grid(True, axis="y", linewidth=FIJECT_DEFAULTS.GRIDWIDTH)
            self.exportToPdf(fig, stem_suffix="_histplot")

    @dataclass
    class ArgsGlobal_BoxPlot:
        iqr_limit: float=1.5
        sort_classes: bool=True

        aspect_ratio: AspectRatio=None
        horizontal: bool=False
        log: bool=False
        value_tickspacing: float=None
        value_axis_label: str=""
        class_axis_label: str=""

    def commit_boxplot(self, value_axis_label: str= "", class_axis_label: str= "",
                       aspect_ratio=None,
                       log=False, horizontal=False, iqr_limit=1.5,
                       value_tickspacing=None):
        """
        Draws multiple boxplots side-by-side.
        Note: the "log" option doesn't just stretch the value axis, because in
        that case you will get a boxplot on skewed data and then stretch that bad
        boxplot. Instead, this method applies log10 to the values, then computes
        the boxplot, and plots it on a regular axis.
        """
        self.commitWithArgs_boxplot(MultiHistogram.ArgsGlobal_BoxPlot(
            iqr_limit=iqr_limit,

            aspect_ratio=aspect_ratio,
            horizontal=horizontal,
            log=log,
            value_tickspacing=value_tickspacing,
            value_axis_label=value_axis_label,
            class_axis_label=class_axis_label
        ))

    def commitWithArgs_boxplot(self, diagram_options: ArgsGlobal_BoxPlot):
        do = diagram_options
        with ProtectedData(self):
            rows = []
            for name, x_values in self.data.items():
                for v in x_values:
                    if do.log:
                        rows.append({"value": np.log10(v), FIJECT_DEFAULTS.LEGEND_TITLE_CLASS: name})
                    else:
                        rows.append({"value": v, FIJECT_DEFAULTS.LEGEND_TITLE_CLASS: name})
            df = pd.DataFrame(rows)
            print(df.groupby(FIJECT_DEFAULTS.LEGEND_TITLE_CLASS).describe())

            fig, ax = newFigAx(do.aspect_ratio)
            ax: plt.Axes

            # Format outliers (https://stackoverflow.com/a/35133139/9352077)
            flierprops = {
                # "markerfacecolor": '0.75',
                # "linestyle": 'none',
                "markersize": 0.1,
                "marker": "."
            }

            value_axis_label = do.value_axis_label
            if do.log and do.value_axis_label:
                value_axis_label = "$\log_{10}($" + value_axis_label + "$)$"

            if do.horizontal:
                sns.boxplot(df, x="value", y=FIJECT_DEFAULTS.LEGEND_TITLE_CLASS,
                            ax=ax, linewidth=0.5, flierprops=flierprops)
                ax.set_xlabel(value_axis_label)
                ax.set_ylabel(do.class_axis_label)
            else:
                sns.boxplot(df, x=FIJECT_DEFAULTS.LEGEND_TITLE_CLASS, y="value",
                            ax=ax, linewidth=0.5, flierprops=flierprops,
                            whis=do.iqr_limit, order=natsorted(self.data.keys()) if do.sort_classes else None)
                ax.set_xlabel(do.class_axis_label)
                ax.set_ylabel(value_axis_label)

                if do.value_tickspacing:
                    # Weird tick spacing hack that somehow works https://stackoverflow.com/a/44525175/9352077
                    import matplotlib.ticker as tck
                    ax.yaxis.set_major_locator(tck.MultipleLocator(do.value_tickspacing))
                    ax.yaxis.set_major_formatter(tck.ScalarFormatter())

            # if x_lims:
            #     ax.set_xlim(x_lims[0], x_lims[1])
            #
            # # ax.set_yticklabels(np.arange(0, max(self.x_values), ytickspacing, dtype=int))  # Don't do this. It literally overwrites existing ticks, rather than placing more of them, so the result is mathematically wrong.
            # ax.set_axisbelow(True)
            # ax.grid(True, axis="y")
            self.exportToPdf(fig, stem_suffix="_boxplot")


class Histogram(MultiHistogram):
    """
    Simplified interface for a histogram of only one variable.
    """

    def _load(self, saved_data: dict):
        super()._load(saved_data)
        if not (len(self.data) == 1 and "x_values" in self.data):
            raise ValueError("Histogram data corrupted.")

    def add(self, x_value: float):
        super().add("x_values", x_value)

    def addMany(self, values: Iterable[float]):
        super().addMany("x_values", values)

    def commit_boxplot(self, x_label="", aspect_ratio=None):
        fig, ax = newFigAx(aspect_ratio)
        df = self.toDataframe()

        sns.boxplot(df, x="value", showmeans=True, orient="h", ax=ax)

        ax.set_xlabel(x_label)
        ax.set_yticklabels([""])
        self.exportToPdf(fig, stem_suffix="_boxplot")

    def commit_violin(self, x_label="", y_label="", aspect_ratio=None):
        fig, ax = newFigAx(aspect_ratio)
        df = self.toDataframe()

        sns.violinplot(df, x="value", orient="h", ax=ax)

        ax.set_xlabel(x_label)
        ax.set_yticklabels([y_label], rotation=90, va='center')
        self.exportToPdf(fig, stem_suffix="_violinplot")

    def commit_qqplot(self, random_variable: scipy.stats.rv_continuous, tickspacing: float=None):  # TODO: Possibly, you need some new limits/tickspacing math for normal/chi²/... distributions.
        # Can be done with my own ScatterPlot class:
        values = self.data["x_values"]
        quantiles = random_variable.ppf((np.arange(1,len(values)+1) - 0.5)/len(values))

        graph = ScatterPlot(name=self.name)
        graph.addPointsToFamily("", quantiles, sorted(values))
        fig, ax = graph.commitWithArgs(
            ScatterPlot.ArgsGlobal(
                aspect_ratio=(3.25, 3.25),
                x_label="Theoretical quantiles",
                y_label="Empirical quantiles",
                legend=False,
                grid_x=True,
                grid_y=True,
                x_tickspacing=tickspacing,
                y_tickspacing=tickspacing
            ),
            ScatterPlot.ArgsPerFamily(size=15),
            export_mode=ExportMode.RETURN_ONLY
        )
        ax.axline(xy1=(0,0), slope=1.0, color="red", zorder=1, alpha=1.0, linewidth=0.75)
        self.exportToPdf(fig, stem_suffix="_qqplot")

        # Doing it with scipy is more complicated because you need to fit your data to your random variable first
        # (which gives you a FitResult object that has a .plot("qq") method) ... so I'm not going to do that!


@dataclass
class BinSpec:

    min: float  # Always defined
    width: float  # Always defined

    max: float=None
    amount: int=None

    @staticmethod
    def halfopen(minimum: float, width: float):
        return BinSpec(
            min=minimum,
            width=width
        )

    @staticmethod
    def closedFromWidth(minimum: float, maximum: float, width: float):
        n_bins_float = (maximum - minimum) / width
        n_bins_int   = round(n_bins_float)
        if abs(n_bins_float - n_bins_int) > 1e-6:
            raise ValueError(f"Cannot divide a range of size {maximum - minimum} into an integer amount of bins of width {width}.")

        return BinSpec(
            min=minimum,
            max=maximum,
            width=width,
            amount=n_bins_int
        )

    @staticmethod
    def closedFromAmount(minimum: float, maximum: float, amount: int):
        return BinSpec(
            min=minimum,
            max=maximum,
            width=(maximum - minimum)/amount,
            amount=amount
        )

    def isClosed(self) -> bool:
        return self.max is not None


class BinOverlapMode(Enum):
    OVERLAY = 1
    STACK = 2
    SIDE_BY_SIDE = 3

    def toString(self) -> str:
        if self == BinOverlapMode.OVERLAY:
            return "layer"
        elif self == BinOverlapMode.STACK:
            return "stack"
        elif self == BinOverlapMode.SIDE_BY_SIDE:
            return "dodge"
        else:
            raise RuntimeError()


class _PrecomputedMultiHistogram(Diagram):

    @dataclass
    class ArgsGlobal:
        aspect_ratio: AspectRatio=None
        x_tickspacing: float=None
        x_label: str=""
        x_center_ticks: bool=False
        y_tickspacing: float=None
        y_label: str=""

        fill_colour: str=None
        border_colour: str=None
        histo_overlapping: BinOverlapMode = BinOverlapMode.OVERLAY

        relative_counts: bool=False
        average_over_bin: bool=False

        # do_kde: bool = True
        # kde_smoothing: bool = True
        # do_hatch: bool = False

        x_lims: Optional[Tuple[Optional[float], Optional[float]]]=None
        # log_x: bool = False
        # log_y: bool = False

    def _commitGivenBars(self, global_args: ArgsGlobal, bar_left_edges: List[float], bar_heights: Dict[str,List[float]], closed_bin_spec: BinSpec,
                         disable_memory_safety: bool=False, **seaborn_args):
        if not closed_bin_spec.isClosed():
            raise ValueError("Bin specification must be closed, but was open.")  # Only user-facing methods have to be exception-safe.
        if not disable_memory_safety and closed_bin_spec.amount > 10**9:
            raise ValueError(f"Requested drawing {closed_bin_spec.amount} bins, which is likely too many without crashing your machine.\nEither reduce the amount of bins or disable memory safety at your own risk.")
        if not bar_left_edges:
            raise ValueError("No bar edges were given.")
        if not bar_heights:
            raise ValueError(f"No bars were given, even though bar edges were: {bar_left_edges}")
        if not all(len(bar_left_edges) == len(bars) for bars in bar_heights.values()):
            raise ValueError(f"Mismatch between amount of bar edges ({len(bar_left_edges)}) and amount of bars heights ({tuple(len(bars) for bars in bar_heights.values())}) given.")

        fig, ax = newFigAx(global_args.aspect_ratio)

        if global_args.relative_counts:
            if global_args.average_over_bin:
                mode = "density"  # Total area is 1.
            else:
                mode = "percent"  # Total area is 100.
        else:
            if global_args.average_over_bin:
                mode = "frequency"
            else:
                mode = "count"

        classes = list(bar_heights.keys())
        sns.histplot(
            # Serialise the given data.
            data={
                "x": bar_left_edges*len(classes),
                "h": cat(bar_heights[c] for c in classes),
                FIJECT_DEFAULTS.LEGEND_TITLE_CLASS: cat([c]*len(bar_heights[c]) for c in classes)
            }, x="x", weights="h", hue=FIJECT_DEFAULTS.LEGEND_TITLE_CLASS if len(bar_heights) > 1 else None,

            # Configure bin computation.
            binwidth=closed_bin_spec.width, binrange=(closed_bin_spec.min, closed_bin_spec.max), ax=ax,
            stat=mode, discrete=global_args.x_center_ticks, common_norm=False,

            # Visual parameters.
            color=global_args.fill_colour, edgecolor=global_args.border_colour,
            multiple=global_args.histo_overlapping.toString(), shrink=1 - 0.1*global_args.x_center_ticks,
            **seaborn_args
        )

        if len(bar_heights) > 1:
            ax.get_legend().set_title(None)  # Legend title is unnecessary clutter.

        ax.set_xlabel(global_args.x_label)
        ax.set_ylabel(global_args.y_label + r" [\%]" * (mode == "percent" and global_args.y_label != ""))

        # TODO: Actually, it's not obvious that you want to scale the gap to the edges by the bar width. Big bars cause way too large gaps and analogous for small bars.
        if global_args.x_center_ticks:
            ax.set_xlim(closed_bin_spec.min - closed_bin_spec.width*0.995, closed_bin_spec.max - 0.005*closed_bin_spec.width)
        else:
            ax.set_xlim(closed_bin_spec.min - closed_bin_spec.width/2, closed_bin_spec.max + closed_bin_spec.width/2)

        # Finally, if the user has specified custom x limits, set those. (Setting None leaves a bound unchanged.)
        if global_args.x_lims:
            ax.set_xlim(left=global_args.x_lims[0], right=global_args.x_lims[1])

        if global_args.x_tickspacing:
            ax.xaxis.set_major_locator(tkr.MultipleLocator(global_args.x_tickspacing))
            ax.xaxis.set_major_formatter(tkr.ScalarFormatter())
        if global_args.y_tickspacing:
            ax.yaxis.set_major_locator(tkr.MultipleLocator(global_args.y_tickspacing))
            ax.yaxis.set_major_formatter(tkr.ScalarFormatter())

        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linewidth=FIJECT_DEFAULTS.GRIDWIDTH)
        self.exportToPdf(fig)


class StreamingMultiHistogram(_PrecomputedMultiHistogram):
    """
    MultiHistogram that has the buckets baked in from the start.
    Say you have 1 billion data points that contribute to the histogram distribution. You can't store these in a list,
    because it will have length 1 billion. Yet, you can still easily turn them into a histogram by having a small amount
    of buckets N and just storing N integer counters that can become very large numbers which take a trivial amount of bits to store.

    Cannot be made more fine-grained. Can be made more coarse-grained (10 bins can be turned into 5 bins unambiguously).

    Not only supports big amounts of samples, but also big amounts of bins if their usage is sparse.
    Also tracks the exact mean and standard deviation, despite not remembering the sample values.
    """

    @dataclass
    class ArgsGlobal(_PrecomputedMultiHistogram.ArgsGlobal):
        combine_buckets: int = 1

    def __init__(self, name: str, binspec: BinSpec,
                 caching: CacheMode=CacheMode.NONE, overwriting: bool=False):
        self.bins = binspec
        super().__init__(name=name, caching=caching, overwriting=overwriting)

    def clear(self):
        super().clear()
        self.data["$summaries"] = dict()

    def _load(self, saved_data: dict):
        for class_name, counts in saved_data.items():
            if class_name == "$summaries":
                self.data[class_name] = counts
                continue

            self.data[class_name] = dict()
            for bin_index,count in counts.items():
                self.data[class_name][int(bin_index)] = count

        assert "$summaries" not in self.data or set(self.data["$summaries"]) == set(filter(lambda name: name != "$summaries", self.data.keys()))

    def add(self, class_name: str, value: float, weight: float=1.0):
        if class_name not in self.data:
            self.data[class_name] = dict()
            self.data["$summaries"][class_name] = [0,0,0]

        i = self._getBinIndex(value)
        if i not in self.data[class_name]:
            self.data[class_name][i] = 0

        self.data[class_name][i] += weight
        self.data["$summaries"][class_name][0] += weight
        self.data["$summaries"][class_name][1] += weight*value
        self.data["$summaries"][class_name][2] += weight*value**2

    def _getBinIndex(self, value: float) -> int:
        if value < self.bins.min or (self.bins.isClosed() and value > self.bins.max):
            warnings.warn(f"Value {value} is not within the histogram range [{self.bins.min};{self.bins.max}] so it was dropped.")
            return -1

        if self.bins.isClosed() and value == self.bins.max:
            return self.bins.amount-1

        value -= self.bins.min
        return int(value/self.bins.width)

    def getSummaries(self, mode_as_middle: bool=False) -> Dict[str,Tuple[float,float,float]]:
        """
        Returns the exact mean, the approximate mode, and the exact standard deviation of the histogram of each class.
        The exact estimators are taken from the BIRCH clustering algorithm.
        The mode is obtained by finding the leftmost bin with the highest amount of samples.

        :param mode_as_middle: Whether to give the mode as the middle of the highest bar, not the left boundary.
        """
        results = dict()
        for class_name, (n,ls,ss) in self.data["$summaries"].items():
            mean = ls/n
            std  = sqrt(1/(n-1) * (ss - n*(ls/n)**2))  if n > 1 else  0.0

            counts = self.data[class_name]
            max_count = max(counts.values())
            mode = None
            for bin_idx, count in counts.items():
                if count == max_count:
                    mode = self.bins.min + self.bins.width*(bin_idx + 0.5*mode_as_middle)
                    break

            results[class_name] = (mean, mode, std)

        return results

    def commit(self, global_args: ArgsGlobal, bin_reweighting: Dict[int,float]=None, **seaborn_args):
        """
        :param bin_reweighting: multipliers to apply to the counts in the original bins.
                                Normally, in histograms, it is samples that are weighted, but since a StreamingHistogram
                                is made to throw away samples, you can only weight bins (equivalent to weighting all the
                                samples in that bin by the same weight).
        """
        with ProtectedData(self):
            if bin_reweighting is None:
                bin_reweighting = dict()

            # First of all: find the last non-empty bin. Even if the binspec says the bin set is half-open, you have to close it when visualising!
            if not self.bins.isClosed():
                if self.isEmpty():
                    warnings.warn("Your histogram has no upper bound and no data to deduce it from. Nothing will be drawn.")
                    return

                n_original_bins = max(max(bin_counts.keys()) for class_name,bin_counts in self.data.items() if class_name != "$summaries") + 1  # if n is the index of the last bin, you have bin 0, 1, 2, 3, ..., n-1, n, which is n+1 bins.
                n_actual_bins = 1 + (n_original_bins-1) // global_args.combine_buckets
            else:  # For closed intervals, you cannot pretend there are more bins to the right to make groups of equal size.
                # TODO: What you could do, however, is check that max(self.data["counts"]) is far enough from the end of the interval to pretend that the interval is actually a bit shorter.
                if self.bins.amount % global_args.combine_buckets != 0:
                    raise ValueError(f"Cannot group {self.bins.amount} buckets in groups of {global_args.combine_buckets} without unfair treatment of the last bucket.")
                n_actual_bins = self.bins.amount // global_args.combine_buckets

            new_bins = BinSpec.closedFromAmount(minimum=self.bins.min, maximum=self.bins.min + n_actual_bins*global_args.combine_buckets*self.bins.width, amount=n_actual_bins)

            # Compute counts in these new bins. Since we don't yet know which bins are used, do this in an unordered dictionary.
            data_in_compressed_bins = dict()
            nonzero_bins = set()
            for class_name, counts in self.data.items():
                if class_name == "$summaries":
                    continue

                data_in_compressed_bins[class_name] = defaultdict(float)
                for original_bin_index, count in counts.items():
                    new_bin_index = original_bin_index // global_args.combine_buckets
                    data_in_compressed_bins[class_name][new_bin_index] += count * bin_reweighting.get(original_bin_index, 1)
                    nonzero_bins.add(new_bin_index)

            # Convert the data to lists with a definite order.
            bin_borders = []
            serial_data = {class_name: [] for class_name in data_in_compressed_bins}
            for new_bin_index in sorted(nonzero_bins):
                bin_borders.append(new_bins.min + new_bins.width*new_bin_index)
                for class_name, counts in data_in_compressed_bins.items():
                    serial_data[class_name].append(counts[new_bin_index])  # Because 'counts' is a defaultdict, it will give you 0 when the bin is not present for this class.

            self._commitGivenBars(global_args, bin_borders, serial_data, closed_bin_spec=new_bins, **seaborn_args)


class StreamingVariableGranularityHistogram(StreamingMultiHistogram):

    def __init__(self, name: str, binspec: BinSpec,
                 caching: CacheMode=CacheMode.NONE, overwriting: bool=False):
        if not binspec.isClosed():
            raise ValueError("Cannot instantiate a StreamingVariableGranularityHistogram without a closed bin domain!")
        super().__init__(name=name, binspec=binspec, caching=caching, overwriting=overwriting)

    def add(self, i: int, n: int, class_name: str, weight: float=1.0):
        """
        Add a sample i from a domain {0, 1, 2, ..., n-1} and spread it across the corresponding 1/n'th of the bins of the histogram.
        """
        if class_name not in self.data:
            self.data[class_name] = dict()
            self.data["$summaries"][class_name] = [0,0,0]

        # This whole part is different from StreamingMultiHistogram, which normally just finds one bin and does so by
        # applying the (value - min)/(max - min) transform. Instead, we do not apply the transformation, and we spread
        # the weight across a span of bins.
        lower_bin = int(self.bins.amount *     i/n)  # At most N-1
        upper_bin = int(self.bins.amount * (i+1)/n)  # At most N, in the one case of i == n-1.
        if lower_bin == upper_bin:  # Happens when your data bins are smaller than your final bins.
            upper_bin += 1

        n_bins = upper_bin - lower_bin
        for b in range(n_bins):
            current_bin = lower_bin+b
            if current_bin not in self.data[class_name]:
                self.data[class_name][current_bin] = 0

            self.data[class_name][current_bin] += weight/n_bins

        # Update summaries, NOT about the '1/n_bins' we added to the different bins,
        # but instead of the i/(n-1) value these bins are roughly centred around, which is what the histogram represents.
        fraction_0_to_1 = i/(n-1)  if n > 1 else  1  # 0/0 becomes 1
        point_value = self.bins.min + fraction_0_to_1*(self.bins.max - self.bins.min)
        self.data["$summaries"][class_name][0] += weight
        self.data["$summaries"][class_name][1] += weight*point_value
        self.data["$summaries"][class_name][2] += weight*point_value**2


class VariableGranularityHistogram(_PrecomputedMultiHistogram):
    """
    Histogram used for combining measurements made for multiple discrete domains that have the same range but a different
    granularity.

    For example, let's say you have a problem where you are given trees with a variable depth, and then you select one particular
    node in each tree. You now want to visualise the depth distribution of these nodes, with the top node having depth 0.0 and
    the furthest descendant of your selected node having depth 1.0. Even though every selected node has a depth between 0.0 and 1.0
    this way, depths are still discretised, and that's a problem: in a tree with the longest root-to-leaf path having 3 nodes, the
    only depth values you can have are 0.0, 0.5 and 1.0. Meanwhile, if the longest path had 4 nodes, the depth values on it would be
    0.0, 0.33, 0.67, and 1.0. For 6 nodes, it would be 0.0, 0.20, 0.40, 0.60, 0.80, 1.0. If you were to use a small bin width,
    the 3-high trees could only ever add to the first bin, the last bin, and the middle bin, and never contribute anything to
    all the other bins. Now imagine that for all trees (no matter the size), the frequency of being selected goes down with a node's
    depth. What you would like to see is exactly that taper in the histogram. Say that the 3-high trees appear most often, then you
    will NOT see a nice taper, but instead, the histograms would have three big spikes with smaller bars in between, completely
    losing the tapered shape.

    Instead, one sample of 0.5 in a 3-high tree should be SPREAD equally across all bins between 0.5-1/6 and 0.5+1/6. More
    formally: given that we want a histogram of N bins in the range [0.0, 1.0] and that the next sample comes from n possible
    values {0, 1, ..., n-2, n-1}, then one sampled value i contributes a uniform weight that sums to 1 to bin indices
    floor(i/n*N)th ... floor((i+1)/n*N)th (exclusive bound, otherwise the latter bin is included for i as the last and for i+1 as the first bucket).
    For example: if n == 3 and N == 20, then i == 1 contributes equally to bins i/n*N == 1/3*20 == 6 ... (i+1)/n*N == 2/3*20 == 13 exc,
    which is 7 bins total.
    Another example: if n == 50 and N = 3, then i == 49 contributes equally to bins floor(49/50*3) == floor(2.94) == 2 to 3 (exclusive),
    which is only 1 bin, namely the last one.
    """

    @dataclass
    class ArgsGlobal(_PrecomputedMultiHistogram.ArgsGlobal):
        x_min: float=0.0
        x_max: float=1.0
        n_bins: int=10

    def clear(self):
        super().clear()
        self.data["domains"] = dict()

    def _load(self, saved_data: dict):
        for n,xs in saved_data["domains"].items():
            self.data[int(n)] = xs

    def add(self, i: int, n: int, weight: float=1.0):
        """
        Add a sample i from a domain {0, 1, 2, ..., n-1}.
        """
        if n not in self.data["domains"]:
            self.data["domains"][n] = []

        self.data["domains"][n].append((i,weight))

    def commit(self, global_args: ArgsGlobal, **seaborn_args):
        with ProtectedData(self):
            CLASS_NAME = ""  # TODO: This class should become multi-class.
            bins = BinSpec.closedFromAmount(minimum=global_args.x_min, maximum=global_args.x_max, amount=global_args.n_bins)

            # The data stored in this class will now, in one go, be parsed by a streaming implementation.
            histo_builder = StreamingVariableGranularityHistogram(name=self.name, binspec=bins)
            for n,xs in self.data["domains"].items():
                for i,weight in xs:
                    histo_builder.add(i, n, class_name=CLASS_NAME, weight=weight)

            # Convert the N bins in [0,1] to shifted and scaled horizontal coordinates.
            xs = []
            ys = []
            for bin_index, count in histo_builder.data[CLASS_NAME].items():
                xs.append(bins.min + bins.width*bin_index)
                ys.append(count)

            # Commit
            self._commitGivenBars(global_args, xs, {CLASS_NAME: ys}, closed_bin_spec=bins, **seaborn_args)

            # Fill cache with summary statistics
            self.cache = histo_builder.getSummaries()[""]

    def getSummary(self) -> Tuple[float,float,float]:
        """
        After a commit, returns mean of the variable-granularity fractions, mode of the histogram, and standard deviation
        of those fractions.
        
        Note: the center of mass of the histogram is not the same as the mean of the data inside it.
        For example, if you only sample from the domain with 4 possible values (equivalent to 0/3, 1/3, 2/3, 3/3), then
        if you only ever added 0, the mean should be 0 whereas the histogram would suggest a mean of 1/8.
        """
        return self.cache
