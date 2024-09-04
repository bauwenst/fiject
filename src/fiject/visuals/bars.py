from dataclasses import dataclass
import matplotlib.ticker as tkr

from ..general import *


class Bars(Diagram):
    """
    Multi-bar chart. Produces a chart with groups of bars on them: each group has the same amount of bars and the
    colours of the bars are in the same order per group.

    All the bars of the same colour are considered the same family. All the families must have the same amount of
    bars, which is equal to the amount of groups.
    """

    @dataclass
    class ArgsGlobal:
        bar_width: float
        group_spacing: float

        group_names: Sequence[str]
        diagonal_labels=True

        aspect_ratio: Tuple[float,float]=None
        y_tickspacing: float=None
        log_y: bool=False
        y_label: str=""

    def _load(self, saved_data: dict):
        self.data = saved_data  # TODO: Needs more sanity checks

    def add(self, bar_slice_family: str, height: float):
        if bar_slice_family not in self.data:
            self.data[bar_slice_family] = []
        self.data[bar_slice_family].append(height)

    def addMany(self, bar_slice_family: str, heights: Sequence[float]):
        if bar_slice_family not in self.data:
            self.data[bar_slice_family] = []
        self.data[bar_slice_family].extend(heights)

    def commit(self, group_names: Sequence[str], bar_width: float, group_spacing: float, y_label: str="",
               diagonal_labels=True, aspect_ratio=None,
               y_tickspacing: float=None, log_y: bool=False):
        """
        The reason that group names are not given beforehand is because they are much like an x_label.
        Compare this to the family names, which are in the legend just as with LineGraph and MultiHistogram.
        """
        with ProtectedData(self):
            fig, main_ax = newFigAx(aspect_ratio)
            main_ax: plt.Axes

            colours = cycleNiceColours()
            group_locations = None
            for i, (bar_slice_family, slice_heights) in enumerate(self.data.items()):
                group_locations = group_spacing * np.arange(len(slice_heights))
                main_ax.bar(group_locations + bar_width*i, slice_heights, color=next(colours), width=bar_width,
                            label=bar_slice_family)

            # X-axis
            main_ax.set_xticks(group_locations + bar_width * len(self.data) / 2 - bar_width / 2)
            main_ax.set_xticklabels(group_names, rotation=45*diagonal_labels, ha="right" if diagonal_labels else "center")
            # main_ax.set_yticks(np.arange(0, 1.1, 0.1))

            # Y-axis
            main_ax.set_ylabel(y_label)
            # main_ax.set_ylim(0, 1)
            if log_y:
                main_ax.set_yscale("log")

            # Grid
            main_ax.set_axisbelow(True)  # Put grid behind the bars.
            main_ax.grid(True, axis="y", linewidth=FIJECT_DEFAULTS.GRIDWIDTH)
            main_ax.legend()

            self.exportToPdf(fig)

    def commitWithArgs(self, global_args: ArgsGlobal):
        return self.commit(
            bar_width=global_args.bar_width,
            group_spacing=global_args.group_spacing,

            group_names=global_args.group_names,
            diagonal_labels=global_args.diagonal_labels,

            aspect_ratio=global_args.aspect_ratio,
            y_tickspacing=global_args.y_tickspacing,
            log_y=global_args.log_y,
            y_label=global_args.y_label
        )


class HistoBars(Diagram):
    """
    Bars with a numerical domain. Has two common usages:
        - List plot: you have a list of values and want to plot them as bar heights, possibly in ascending/descending
                     order. The input explicitly doesn't matter.
        - Probability functions: you know the real->real probability mass or density mapping, and you want to plot it as
                                 disconnected bars rather than a connected line graph. To produce the same result with a
                                 histogram, you'd have to sample millions of points from the probability function.

    Unlike a histogram, you get exactly one bar per unique input value (because inputs are "mathematically meaningless"
    like in a categorical bar plot and hence aren't binned).
    """

    @dataclass
    class ArgsGlobal:
        sort_keyless_data: bool = False
        small_to_big: bool = True

        bar_width: float = 1
        center_ticks: bool = True

        aspect_ratio: Tuple[float,float] = None
        grid: bool = True
        x_tickspacing: float = None
        x_label: str = ""
        y_tickspacing: float = None
        y_label: str = ""
        log_y: bool = False

    def add(self, family: str, x: float, y: float):
        if family not in self.data:
            self.data[family] = dict()
        elif isinstance(self.data[family], list):
            raise ValueError(f"The bar family {family} contains keyless data, whilst you tried adding a key-value pair ({x},{y}).")

        self.data[family][x] = y

    def append(self, family: str, y: float):
        if family not in self.data:
            self.data[family] = []
        elif isinstance(self.data[family], dict):
            raise ValueError(f"The bar family {family} contains key-value data, whilst you tried adding a keyless data point {y}.")

        self.data[family].append(y)

    def commit(self, global_args: ArgsGlobal):
        """
        Based on
        https://stackoverflow.com/questions/62021334/barplot-in-seaborn-with-height-based-on-an-array
        and
        https://stackoverflow.com/questions/58963320/bar-plot-with-irregular-spacing
        """
        with ProtectedData(self):
            fig, main_ax = newFigAx(global_args.aspect_ratio)

            # TODO: would be nice to have opacity as in
            #     https://seaborn.pydata.org/generated/seaborn.objects.Bars.html
            colours = cycleNiceColours()
            for family, data in self.data.items():
                if isinstance(data, list):
                    x_values = range(len(data))
                    y_values = data if not global_args.sort_keyless_data else sorted(data, reverse=not global_args.small_to_big)
                elif isinstance(data, dict):
                    x_values, y_values = zip(*data.items())

                main_ax.bar(x_values, y_values, width=global_args.bar_width, align="center" if global_args.center_ticks else "edge", color=next(colours),
                           label=family)  # For the legend

            if global_args.x_tickspacing:  # set_xticks needs a minimum, which we don't want as usual. Also, bars have their own way of setting ticks: https://stackoverflow.com/questions/64196514/bar-plot-only-shows-the-last-tick-label
                main_ax.xaxis.set_major_locator(tkr.MultipleLocator(global_args.x_tickspacing))
                main_ax.xaxis.set_major_formatter(tkr.ScalarFormatter())

            if global_args.y_tickspacing:
                main_ax.yaxis.set_major_locator(tkr.MultipleLocator(global_args.y_tickspacing))
                main_ax.yaxis.set_major_formatter(tkr.ScalarFormatter())

            main_ax.set_xlabel(global_args.x_label)
            main_ax.set_ylabel(global_args.y_label)
            if global_args.log_y:
                main_ax.set_yscale("log")

            # Grid
            if global_args.grid:
                main_ax.set_axisbelow(True)  # Put grid behind the bars.
                main_ax.grid(True, axis="y", linewidth=FIJECT_DEFAULTS.GRIDWIDTH)
            main_ax.legend()

            self.exportToPdf(fig)
