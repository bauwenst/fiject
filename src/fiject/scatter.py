from .general import *

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

    def commit(self, aspect_ratio: Tuple[float,float]=DEFAULT_ASPECT_RATIO, x_label="", y_label="", legend=False,
               x_lims=None, y_lims=None, logx=False, logy=False, x_tickspacing=None, y_tickspacing=None, grid=False,
               family_colours=None, family_sizes=None, randomise_markers=False, only_for_return=False):
        with ProtectedData(self):
            fig, ax = newFigAx(aspect_ratio)
            ax: plt.Axes

            if logx:
                ax.set_xscale("log")  # Needed for a log scatterplot. https://stackoverflow.com/a/52573929/9352077
                ax.xaxis.set_major_locator(tkr.LogLocator(base=10, numticks=999))  # See comment under https://stackoverflow.com/q/76285293/9352077
                ax.xaxis.set_major_formatter(tkr.LogFormatterSciNotation())
            elif x_tickspacing:
                    ax.xaxis.set_major_locator(tkr.MultipleLocator(x_tickspacing))
                    ax.xaxis.set_major_formatter(tkr.ScalarFormatter())

            if logy:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(tkr.LogLocator(base=10, numticks=999))
                ax.yaxis.set_major_formatter(tkr.LogFormatterSciNotation())
            elif y_tickspacing:
                ax.xaxis.set_major_locator(tkr.MultipleLocator(y_tickspacing))
                ax.xaxis.set_major_formatter(tkr.ScalarFormatter())

            if logx and logy:  # Otherwise you have a skewed view of horizontal vs. vertical distances.
                ax.set_aspect("equal")

            if family_colours is None:
                family_colours = dict()
            if family_sizes is None:
                family_sizes = dict()

            markers = {".", "^", "+", "s"}
            # cols = getColours()
            cols = plt.cm.rainbow(np.linspace(0, 1, len(self.data)))  # Equally spaced rainbow colours.
            scatters = []
            names    = []
            for idx, tup in enumerate(sorted(self.data.items(), reverse=True)):
                name, family = tup
                m = markers.pop() if randomise_markers else "."
                c = family_colours.get(name, cols[idx])
                s = family_sizes.get(name, 35)
                result = ax.scatter(family[0], family[1], marker=m, linewidths=0.05, color=c, s=s)
                scatters.append(result)
                names.append(name)

            if x_lims:
                ax.set_xlim(x_lims[0], x_lims[1])
            if y_lims:
                ax.set_ylim(y_lims[0], y_lims[1])

            if x_label:
                ax.set_xlabel(x_label)
            if y_label:
                ax.set_ylabel(y_label)

            if grid:
                ax.set_axisbelow(True)
                ax.grid(True, linewidth=DEFAULT_GRIDWIDTH)

            if legend:
                ax.legend(scatters, names, loc='upper left', markerscale=10, ncol=2)  # https://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend

            if not only_for_return:
                self.exportToPdf(fig)
            return fig, ax
