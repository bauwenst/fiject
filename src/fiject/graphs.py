from .general import *

import itertools


class LineGraph(Diagram):
    """
    2D line graph. Plots the relationship between TWO variables, given in paired observations (x,y), and connects them.
    For the name, see: https://trends.google.com/trends/explore?date=all&q=line%20graph,line%20chart,line%20plot
    """

    def _load(self, saved_data: dict):
        """
        Restore a file by extending the series of this object.
        """
        for name, tup in saved_data.items():
            if len(tup) != 2 or len(tup[0]) != len(tup[1]):
                raise ValueError("Graph data corrupted: either there aren't two tracks for a series, or they don't match in length.")

            self.addMany(name, tup[0], tup[1])

    def initSeries(self, series_name: str):
        self.data[series_name] = ([], [])

    def add(self, series_name: str, x, y):
        """
        Add a single datapoint to the series (line) with the given label.
        """
        if type(x) == str or type(y) == str:
            print("WARNING: You are trying to use a string as x or y data. The datapoint was discarded, because this causes nonsensical graphs.")
            return

        if series_name not in self.data:
            self.initSeries(series_name)
        self.data[series_name][0].append(x)
        self.data[series_name][1].append(y)

    def addMany(self, series_name: str, xs: Sequence, ys: Sequence):
        if series_name not in self.data:
            self.initSeries(series_name)
        self.data[series_name][0].extend(xs)
        self.data[series_name][1].extend(ys)

    def commit(self, aspect_ratio=DEFAULT_ASPECT_RATIO, x_label="", y_label="", legend_position="lower right",
               do_points=True, initial_style_idx=0,
               grid_linewidth=DEFAULT_GRIDWIDTH, curve_linewidth=1, optional_line_at_y=None,
               y_lims=None, x_tickspacing=None, y_tickspacing=None, logx=False, logy=False,
               only_for_return=False, existing_figax: tuple=None):
        """
        Render a figure based on the added data.
        Also stores the data to a JSON file (see save()).

        Since figure rendering can error due to the LaTeX compiler (for example, because your axis labels use unicode
        instead of LaTeX commands), the entire implementation is wrapped in a try-except.
        Yes, I had to find out the hard way by losing a 6-hour render.
        """
        with ProtectedData(self):
            # The graph style is a tuple (col, line, marker) that cycles from front to back:
            #   - red solid dot, blue solid dot, green solid dot
            #   - red dashed dot, blue dashed dot, green dashed dot
            #   - ...
            #   - red solid x, blue solid x, green solid x
            #   - ...
            colours = getColours()
            line_styles = ["-", "--", ":"]
            line_markers = [".", "x", "+"] if do_points else [""]
            line_styles = list(itertools.product(line_markers, line_styles, colours))

            if existing_figax is None:
                fig, main_ax = newFigAx(aspect_ratio)
            else:
                fig, main_ax = existing_figax
            main_ax.grid(True, which='both', linewidth=grid_linewidth)
            main_ax.axhline(y=0, color='k', lw=0.5)

            style_idx = initial_style_idx
            for name, samples in self.data.items():
                marker, line, colour = line_styles[style_idx % len(line_styles)]
                style = marker + line

                if logx and logy:
                    main_ax.loglog(samples[0], samples[1], style, c=colour, label=name, linewidth=curve_linewidth)
                elif logx:
                    main_ax.semilogx(samples[0], samples[1], style, c=colour, label=name, linewidth=curve_linewidth)
                elif logy:
                    main_ax.semilogy(samples[0], samples[1], style, c=colour, label=name, linewidth=curve_linewidth)
                else:
                    main_ax.plot(samples[0], samples[1], style, c=colour, label=name, linewidth=curve_linewidth)

                style_idx += 1

            if optional_line_at_y is not None:
                main_ax.hlines(optional_line_at_y,
                               min([min(tup[0]) for tup in self.data.values()]),
                               max([max(tup[0]) for tup in self.data.values()]), colors='b', linestyles='dotted')

            if x_label:
                main_ax.set_xlabel(x_label)
            if y_label:
                main_ax.set_ylabel(y_label)
            if legend_position:  # Can be None or "" to turn it off.
                main_ax.legend(loc=legend_position)

            if y_lims:
                main_ax.set_ylim(y_lims[0], y_lims[1])

            if x_tickspacing:
                x_min, x_max = main_ax.get_xlim()
                main_ax.set_xticks(np.arange(0, x_max, x_tickspacing))

            if y_tickspacing:
                y_min, y_max = main_ax.get_ylim()
                main_ax.set_yticks(np.arange(0, y_max, y_tickspacing))

            if y_lims:  # Yes, twice. Don't ask.
                main_ax.set_ylim(y_lims[0], y_lims[1])

            if not only_for_return:
                self.exportToPdf(fig)
            return fig, main_ax

    def merge_commit(self, fig, ax1, other_graph: "LineGraph", **second_commit_kwargs):
        """
        The signature is vague because this is the only way to abstract the process of having a twin x-axis.
        Basically, here are the options:
            1. The user commits graph 1 without saving, takes a twinx for the ax, passes that into the commit of graph 2,
               and then passes the fig,ax1,ax2 triplet into a third function that adds the legends and saves.
            2. Same first step, but do the rest in this function, sadly not having autocompletion for the second graph's
               commit arguments.
            3. Same, except copy all the options from .commit() and pick the ones to re-implement here (e.g. styling, y
               limits ...).
            4. Do everything inside this function by copying the signature from .commit() twice (once per graph).

        The last two are really bad design, and really tedious to maintain.
        """
        name = "merged_(" + self.name + ")_(" + other_graph.name + ")"

        # "Twin x" means they share the same figure and x axis, but the other's y axis will be on the right.
        ax2 = ax1.twinx()

        # Modify ax2 in-place.
        other_graph.commit(**second_commit_kwargs, existing_figax=(fig,ax2),
                           initial_style_idx=len(self.data), only_for_return=True)

        # Drawing the legends is slightly tricky, see https://stackoverflow.com/a/54631364/9352077
        legend_1 = ax1.legend(loc='upper right')
        legend_1.remove()
        ax2.legend(loc='lower right')
        ax2.add_artist(legend_1)

        # At last, save.
        Diagram.safeFigureWrite(name, ".pdf", fig)

    @staticmethod
    def qndLoadAndCommit(json_path: Path):
        """
        Quick-and-dirty method to load in the JSON data of a graph and commit it without
        axis formatting etc. Useful when rendering with commit() failed but your JSON was
        saved, and you want to get a rough idea of what came out of it.

        Strips the part of the stem at the end that starts with "_".
        """
        raw_name = json_path.stem
        g = LineGraph(raw_name[:raw_name.rfind("_")], caching=CacheMode.NONE)
        g.load(json_path)
        g.commit()


class MergedLineGraph(Diagram):
    """
    Merger of two line graphs.
    The x axis is shared, the first graph's y axis is on the left, and the second graph's y axis is on the right.
    """

    def __init__(self, g1: LineGraph, g2: LineGraph,
                 caching: CacheMode=CacheMode.NONE):
        self.g1 = g1
        self.g2 = g2
        super().__init__(name=self.makeName(), caching=caching)

    def makeName(self):
        return self.g1.name + "+" + self.g2.name

    def _save(self) -> dict:
        return {"G1": {"name": self.g1.name,
                       "data": self.g1._save()},
                "G2": {"name": self.g2.name,
                       "data": self.g2._save()}}

    def _load(self, saved_data: dict):
        # The KeyError thrown when one of these isn't present, is sufficient.
        name, data = saved_data["G1"]["name"], saved_data["G1"]["data"]
        self.g1.name = name
        self.g1._load(data)
        name, data = saved_data["G2"]["name"], saved_data["G2"]["data"]
        self.g2.name = name
        self.g2._load(data)

        if self.name != self.makeName():
            raise ValueError("Graph names corrupted: found", self.g1.name, "and", self.g2.name, "for merge", self.name)

    def commit(self, aspect_ratio=DEFAULT_ASPECT_RATIO, x_label="", y_label_left="", y_label_right=""):
        with ProtectedData(self):
            ######## ALREADY IN .COMMIT
            # First graph
            colours = getColours()
            fig, ax1 = newFigAx(aspect_ratio)
            ax1.grid(True, which='both')
            ax1.axhline(y=0, color='k', lw=0.5)

            for name, samples in self.g1.data.items():
                ax1.plot(samples[0], samples[1], c=colours.pop(0), marker=".", linestyle="-", label=name)
            ########

            # Second graph
            ax2 = ax1.twinx()  # "Twin x" means they share the same figure and x axis, but the other's y axis will be on the right.

            ######## ALREADY IN .COMMIT
            for name, samples in self.g2.data.items():
                ax2.plot(samples[0], samples[1], c=colours.pop(0), marker=".", linestyle="-", label=name)

            # Labels
            if x_label:
                ax1.set_xlabel(x_label)
            if y_label_left:
                ax1.set_ylabel(y_label_left)
            if y_label_right:
                ax2.set_ylabel(y_label_right)
            #########

            # Drawing the legends is slightly tricky, see https://stackoverflow.com/a/54631364/9352077
            legend_1 = ax1.legend(loc='upper right')
            legend_1.remove()
            ax2.legend(loc='lower right')
            ax2.add_artist(legend_1)

            self.exportToPdf(fig)
