from ..general import *

from dataclasses import dataclass
import itertools

import numpy as np
import matplotlib.ticker as tkr



class ScatterPlot(Diagram):

    def _load(self, saved_data: dict):
        self.data = saved_data  # FIXME: Needs more sanity checks obviously

    def copy(self, new_name: str):
        new_plot = ScatterPlot(new_name, caching=CacheMode.NONE)
        for name, values in self.data.items():
            new_plot.addPointsToFamily(name, values[0].copy(), values[1].copy())
        return new_plot

    def addPointsToFamily(self, family_name: str, xs: Iterable[float], ys: Iterable[float]):
        """
        Unlike the other diagram types, it seems justified to add scatterplot points in bulk.
        Neither axis is likely to represent time, so you'll probably have many points available at once.
        """
        if family_name not in self.data:
            self.data[family_name] = ([], [])
        self.data[family_name][0].extend(xs)
        self.data[family_name][1].extend(ys)

    def precommit(self) -> Dict[str,Tuple[List[float],List[float]]]:
        """
        Return a transformed version of the dataset before committing.
        """
        return self.data

    @dataclass
    class ArgsGlobal:
        aspect_ratio: Tuple[float, float] = None
        x_label: str = ""
        y_label: str = ""
        x_lims: Tuple[float, float] = None
        y_lims: Tuple[float, float] = None
        logx: bool = False
        logy: bool = False
        legend: bool = False
        x_tickspacing: float = None
        y_tickspacing: float = None
        grid_x: bool = False
        grid_y: bool = False

        default_markers_change: bool = False
        default_colours_rainbow: bool = False

    @dataclass
    class ArgsPerFamily:
        colour: str = None
        marker: str = None
        size: float = 35

    def commitWithArgs(self, diagram_options: ArgsGlobal, default_family_options: ArgsPerFamily, extra_family_options: Dict[str,ArgsPerFamily]=None,
                       only_for_return=False):
        do = diagram_options
        with ProtectedData(self):
            if extra_family_options is None:
                extra_family_options = dict()

            fig, ax = newFigAx(do.aspect_ratio)
            ax: plt.Axes

            data = self.precommit()

            if do.logx:
                ax.set_xscale("log")  # Needed for a log scatterplot. https://stackoverflow.com/a/52573929/9352077
                ax.xaxis.set_major_locator(tkr.LogLocator(base=10, numticks=999))  # See comment under https://stackoverflow.com/q/76285293/9352077
                ax.xaxis.set_major_formatter(tkr.LogFormatterSciNotation())
            elif do.x_tickspacing:
                ax.xaxis.set_major_locator(tkr.MultipleLocator(do.x_tickspacing))
                ax.xaxis.set_major_formatter(tkr.ScalarFormatter())

            if do.logy:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(tkr.LogLocator(base=10, numticks=999))
                ax.yaxis.set_major_formatter(tkr.LogFormatterSciNotation())
            elif do.y_tickspacing:
                ax.xaxis.set_major_locator(tkr.MultipleLocator(do.y_tickspacing))
                ax.xaxis.set_major_formatter(tkr.ScalarFormatter())

            if do.logx and do.logy:  # Otherwise you have a skewed view of horizontal vs. vertical distances.
                ax.set_aspect("equal")

            markers = itertools.cycle([".", "^", "+", "s"]) if do.default_markers_change else itertools.cycle(["."])
            cols    = cycleRainbowColours(len(data))        if do.default_colours_rainbow else cycleNiceColours()
            scatters = []
            names    = []
            for name, family in sorted(data.items(), reverse=True):
                options = extra_family_options.get(name, default_family_options)

                marker = options.marker if options.marker is not None else next(markers)
                colour = options.colour if options.colour is not None else next(cols)
                size   = options.size
                result = ax.scatter(family[0], family[1], marker=marker, linewidths=0.05, color=colour, s=size)
                scatters.append(result)
                names.append(name)

            if do.x_lims:
                ax.set_xlim(do.x_lims[0], do.x_lims[1])
            if do.y_lims:
                ax.set_ylim(do.y_lims[0], do.y_lims[1])

            if do.x_label:
                ax.set_xlabel(do.x_label)
            if do.y_label:
                ax.set_ylabel(do.y_label)

            if do.grid_x or do.grid_y:
                ax.set_axisbelow(True)
                ax.grid(axis="x" if do.grid_x and not do.grid_y else "y" if do.grid_y and not do.grid_x else "both",
                        linewidth=FIJECT_DEFAULTS.GRIDWIDTH)

            if do.legend:
                ax.legend(scatters, names, loc='upper left', markerscale=10, ncol=2)  # https://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend

            if not only_for_return:
                self.exportToPdf(fig)
            return fig, ax


class ScatterPlot_DiscreteContinuous(ScatterPlot):
    """
    Scatterplot (i.e. visual of bivariate non-functional data), but the first variable is assumed to be discrete
    (and specifically, to have INTEGER values, a special kind of discrete).

    With N discrete values, that means any pair of points has 1/N chance of matching exactly in one of the coordinate
    dimensions, and if their other coordinate follows any kind of distribution, they have a good chance of overlapping
    there too.

    Hence, we jitter around the discrete values to keep them visually discrete, but have less overlap probability.
    """

    def precommit(self) -> Dict[str,Tuple[List[float],List[float]]]:
        import numpy.random as npr
        MAX_JITTER = 0.25   # Due to the assumed integer spacing of the x values, this means you have an equal amount of air as you will bar between each number (0.25 bar 1, 0.5 air, 0.25 bar 2, 0.25 bar 2, 0.5 air, 0.25 bar 3 ...)

        jittered_data = dict()
        for family, (xs, ys) in self.data.items():
            new_xs = [x + MAX_JITTER * (2*(npr.rand()-0.5)) for x in xs]
            jittered_data[family] = (new_xs, ys)

        return jittered_data