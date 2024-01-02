from .general import *
from .scatter import ScatterPlot

import math
import pandas as pd
import scipy

import seaborn as sns
import matplotlib.legend as lgd  # Only for type-checking.
import matplotlib.ticker as tkr


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
        self.data[series_name].append(x_value)

    def addMany(self, series_name: str, values: Sequence[float]):
        if series_name not in self.data:
            self.data[series_name] = []
        self.data[series_name].extend(values)

    def toDataframe(self):
        # You would think all it takes is pd.Dataframe(dictionary), but that's the wrong way to do it.
        # If you do it that way, you are pairing up the i'th value of each family as if it belongs to one
        # object. This is not correct, and crashes if you have different sample amounts per family.
        rows = []  # For speed, instead of appending to a dataframe, make a list of rows as dicts. https://stackoverflow.com/a/17496530/9352077
        for name, x_values in self.data.items():
            for v in x_values:
                rows.append({"value": v, LEGEND_TITLE_CLASS: name})
        df = pd.DataFrame(rows)
        return df if len(self.data) != 1 else df.drop(columns=[LEGEND_TITLE_CLASS])

    def commit(self, width: float, x_label="", y_label="", aspect_ratio=DEFAULT_ASPECT_RATIO):
        with ProtectedData(self):
            fig, main_ax = newFigAx(aspect_ratio)
            for name,x_values in self.data.items():
                main_ax.hist(x_values, bins=int((max(x_values) - min(x_values)) // width))  # You could do this, but I don't think the ticks will then be easy to set.

            if x_label:
                main_ax.set_xlabel(x_label)
            if y_label:
                main_ax.set_ylabel(y_label)

            self.exportToPdf(fig)

    def commit_histplot(self, binwidth: float=1, log_x=False, log_y=False,
                        relative_counts: bool=False, average_over_bin: bool=False,
                        x_lims: Tuple[int,int]=None, aspect_ratio=DEFAULT_ASPECT_RATIO,
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
        with ProtectedData(self):
            if relative_counts:
                if average_over_bin:
                    mode = "density"  # Total area is 1.
                else:
                    mode = "percent"  # Total area is 100.
            else:
                if average_over_bin:
                    mode = "frequency"
                else:
                    mode = "count"

            df = self.toDataframe()
            if len(self.data) != 1:
                legend_title = LEGEND_TITLE_CLASS
                # print(df.groupby(LEGEND_TITLE_CLASS).describe())
            else:
                legend_title = None
                # print(df.value_counts())

            fig, ax = newFigAx(aspect_ratio)
            if not log_x:
                sns.histplot(df, ax=ax, x="value", hue=legend_title,  # x and hue args: https://seaborn.pydata.org/tutorial/distributions.html
                             binwidth=binwidth, binrange=(math.floor(df["value"].min()/binwidth)*binwidth,
                                                          math.ceil( df["value"].max()/binwidth)*binwidth),
                             discrete=center_ticks, stat=mode, common_norm=False,
                             kde=do_kde, kde_kws={"bw_adjust": 10} if kde_smoothing else seaborn_args.pop("kde_kws", None),  # Btw, do not use displot: https://stackoverflow.com/a/63895570/9352077
                             color=fill_colour, edgecolor=border_colour,
                             **seaborn_args)  # Do note use displot: https://stackoverflow.com/a/63895570/9352077
            else:
                sns.histplot(df, ax=ax, x="value", hue=legend_title,
                             log_scale=True,
                             discrete=center_ticks, stat=mode, common_norm=False,
                             color=fill_colour, edgecolor=border_colour,
                             **seaborn_args)  # Do note use displot: https://stackoverflow.com/a/63895570/9352077

            # Cross-hatching
            if do_hatch:
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
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label + r" [\%]" * (mode == "percent" and y_label != ""))
            if x_lims:
                if x_lims[0] is not None and x_lims[1] is not None:
                    ax.set_xlim(x_lims[0], x_lims[1])
                elif x_lims[0] is not None:
                    ax.set_xlim(left=x_lims[0])
                elif x_lims[1] is not None:
                    ax.set_xlim(right=x_lims[1])
                else:  # You passed (None,None)...
                    pass

            # Weird tick spacing hack that somehow works https://stackoverflow.com/a/44525175/9352077
            if not log_x:
                ax.xaxis.set_major_locator(tkr.MultipleLocator(x_tickspacing))
                ax.xaxis.set_major_formatter(tkr.ScalarFormatter())

            if not log_y:
                if y_tickspacing:
                    ax.yaxis.set_major_locator(tkr.MultipleLocator(y_tickspacing))
                    ax.yaxis.set_major_formatter(tkr.ScalarFormatter())
            else:
                ax.set_yscale("log")

            # ax.set_yticklabels(np.arange(0, max(self.x_values), ytickspacing, dtype=int))  # Don't do this. It literally overwrites existing ticks, rather than placing more of them, so the result is mathematically wrong.
            ax.set_axisbelow(True)
            ax.grid(True, axis="y", linewidth=DEFAULT_GRIDWIDTH)
            self.exportToPdf(fig, stem_suffix="_histplot")

    def commit_boxplot(self, value_axis_label: str= "", class_axis_label: str= "",
                       aspect_ratio=DEFAULT_ASPECT_RATIO,
                       log=False, horizontal=False, iqr_limit=1.5,
                       value_tickspacing=None):
        """
        Draws multiple boxplots side-by-side.
        Note: the "log" option doesn't just stretch the value axis, because in
        that case you will get a boxplot on skewed data and then stretch that bad
        boxplot. Instead, this method applies log10 to the values, then computes
        the boxplot, and plots it on a regular axis.
        """
        with ProtectedData(self):
            rows = []
            for name, x_values in self.data.items():
                for v in x_values:
                    if log:
                        rows.append({"value": np.log10(v), LEGEND_TITLE_CLASS: name})
                    else:
                        rows.append({"value": v, LEGEND_TITLE_CLASS: name})
            df = pd.DataFrame(rows)
            print(df.groupby(LEGEND_TITLE_CLASS).describe())

            fig, ax = newFigAx(aspect_ratio)
            ax: plt.Axes

            # Format outliers (https://stackoverflow.com/a/35133139/9352077)
            flierprops = {
                # "markerfacecolor": '0.75',
                # "linestyle": 'none',
                "markersize": 0.1,
                "marker": "."
            }

            if log and value_axis_label:
                value_axis_label = "$\log_{10}($" + value_axis_label + "$)$"

            if horizontal:
                sns.boxplot(df, x="value", y=LEGEND_TITLE_CLASS,
                            ax=ax, linewidth=0.5, flierprops=flierprops)
                ax.set_xlabel(value_axis_label)
                ax.set_ylabel(class_axis_label)
            else:
                sns.boxplot(df, x=LEGEND_TITLE_CLASS, y="value",
                            ax=ax, linewidth=0.5, flierprops=flierprops,
                            whis=iqr_limit)
                ax.set_xlabel(class_axis_label)
                ax.set_ylabel(value_axis_label)

                if value_tickspacing:
                    # Weird tick spacing hack that somehow works https://stackoverflow.com/a/44525175/9352077
                    import matplotlib.ticker as tck
                    ax.yaxis.set_major_locator(tck.MultipleLocator(value_tickspacing))
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

    def addMany(self, values: Sequence[float]):
        super().addMany("x_values", values)

    def commit_boxplot(self, x_label="", aspect_ratio=DEFAULT_ASPECT_RATIO):
        fig, ax = newFigAx(aspect_ratio)
        df = self.toDataframe()

        sns.boxplot(df, x="value", showmeans=True, orient="h", ax=ax)

        ax.set_xlabel(x_label)
        ax.set_yticklabels([""])
        self.exportToPdf(fig, stem_suffix="_boxplot")

    def commit_violin(self, x_label="", y_label="", aspect_ratio=DEFAULT_ASPECT_RATIO):
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
        fig, ax = graph.commit(aspect_ratio=(3.25,3.25), x_label="Theoretical quantiles", y_label="Empirical quantiles",
                               family_sizes={"": 15}, only_for_return=True, legend=False,
                               grid=True, x_tickspacing=tickspacing, y_tickspacing=tickspacing)
        ax.axline(xy1=(0,0), slope=1.0, color="red", zorder=1, alpha=1.0, linewidth=0.75)
        self.exportToPdf(fig, stem_suffix="_qqplot")

        # Doing it with scipy is more complicated because you need to fit your data to your random variable first
        # (which gives you a FitResult object that has a .plot("qq") method) ... so I'm not going to do that!
