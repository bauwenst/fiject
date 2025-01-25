import warnings
from dataclasses import dataclass
from math import sqrt
import scipy
import itertools
from collections import defaultdict

import matplotlib.ticker as tkr

from ..general import *
from ..util.functions import weightedMean, weightedVariance


class LineGraph(Visual):
    """
    2D line graph. Plots the relationship between TWO variables, given in paired observations (x,y), and connects them.
    For the name, see: https://trends.google.com/trends/explore?date=all&q=line%20graph,line%20chart,line%20plot
    """

    @dataclass
    class ArgsGlobal:
        aspect_ratio: Tuple[float,float] = None
        do_spines: bool=True
        y_lims: Tuple[Optional[float], Optional[float]] = None

        x_label: str = ""
        y_label: str = ""
        legend_position: str = "lower right"
        initial_style_idx: int = 0
        grid_linewidth: float = None
        curve_linewidth: float = 1
        optional_line_at_y: float = None

        x_gridspacing: float = None
        y_gridspacing: float = None
        x_tickspacing: float = None
        y_tickspacing: float = None
        x_ticks_hardcoded: list = None
        logx: bool = False
        logy: bool = False
        tick_scientific_notation: bool = False  # Not 10 000 but 1*10^4.
        tick_log_multiples: bool = False  # Instead of ticking 10^1, 10^2, 10^3... also tick 2*10^1, 3*10^1, 4*10^1...
        
        logx_becomes_linear_at: float = 0.0  # If this is changed to a non-zero number, a logarithmic x-axis will be cut off and mirrored around 0.

        functions: Dict[str,Callable[[float],float]] = None  # Functions you want to evaluate on-the-fly during a commit.
        function_samples: int = 100

    @dataclass
    class ArgsPerLine:
        show_points: bool = True
        show_line: bool = True
        # These should be left empty for the default style, so you can cycle through the offered styles.
        colour: str = None
        line_style: str = None
        point_marker: str = None

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

    def add(self, series_name: str, x: float, y: float):
        """
        Add a single datapoint to the series (line) with the given label.
        """
        if type(x) == str or type(y) == str:
            print("WARNING: You are trying to use a string as x or y data. The datapoint was discarded, because this causes nonsensical graphs.")
            return

        if series_name not in self.data:
            self.initSeries(series_name)
        self.data[series_name][0].append(float(x))
        self.data[series_name][1].append(float(y))

    def addMany(self, series_name: str, xs: Sequence, ys: Sequence):
        if series_name not in self.data:
            self.initSeries(series_name)
        self.data[series_name][0].extend(map(float,xs))
        self.data[series_name][1].extend(map(float,ys))

    # TODO: Should be phased out at some point.
    def commit(self, aspect_ratio=None, x_label="", y_label="", legend_position="lower right",
               do_points=True, do_lines=True, initial_style_idx=0,
               grid_linewidth=None, curve_linewidth=1, optional_line_at_y=None,
               y_lims=None, x_tickspacing=None, y_tickspacing=None, logx=False, logy=False,
               export_mode: ExportMode=ExportMode.SAVE_ONLY, existing_figax: tuple=None):
        self.commitWithArgs(
            global_options=LineGraph.ArgsGlobal(
                aspect_ratio=aspect_ratio,
                x_label=x_label,
                y_label=y_label,
                legend_position=legend_position,
                grid_linewidth=grid_linewidth,
                curve_linewidth=curve_linewidth,
                optional_line_at_y=optional_line_at_y,
                y_lims=y_lims,
                x_tickspacing=x_tickspacing,
                y_tickspacing=y_tickspacing,
                logx=logx,
                logy=logy,
                initial_style_idx=initial_style_idx
            ),
            default_line_options=LineGraph.ArgsPerLine(
                show_points=do_points,
                show_line=do_lines
            ),
            export_mode=export_mode,
            existing_figax=existing_figax
        )

    def commitWithArgs(self, global_options: ArgsGlobal, default_line_options: ArgsPerLine, extra_line_options: Dict[str,ArgsPerLine]=None,
                       export_mode: ExportMode=ExportMode.SAVE_ONLY, existing_figax: tuple=None):
        """
        Render a figure based on the added data.
        Also stores the data to a JSON file (see save()).

        Since figure rendering can error due to the LaTeX compiler (for example, because your axis labels use unicode
        instead of LaTeX commands), the entire implementation is wrapped in a try-except.
        Yes, I had to find out the hard way by losing a 6-hour render.
        """
        do = global_options
        with ProtectedData(self):
            if extra_line_options is None:
                extra_line_options = dict()
            if not do.functions:
                do.functions = dict()

            if existing_figax is None:
                fig, main_ax = LineGraph._newFigAx(do)
            else:
                fig, main_ax = existing_figax

            # Get all data, including procedurally generated data.
            try:
                min_x = min(min(xs) for _,(xs,_) in self.data.items())
                max_x = max(max(xs) for _,(xs,_) in self.data.items())
            except:  # TODO: If you ever add x_lims, those should definitely go here.
                min_x = -1
                max_x = +1

            all_series = [(name, xy) for name, xy in self.data.items()]
            if not do.logx:
                xs = np.linspace(min_x, max_x, num=do.function_samples)
            else:
                xs = np.logspace(np.log10(min_x), np.log10(max_x), num=do.function_samples)
            for name, f in do.functions.items():
                ys = [f(x) for x in xs]
                all_series.append((name, (xs,ys)))

            # Plotting
            styles = LineGraph._makeLineStyleGenerator(advance_by=do.initial_style_idx)
            for name, (xs,ys) in all_series:
                # Get style options
                is_function = name in do.functions
                marker, line, colour = LineGraph._resolveLineStyle(name, is_function, default_line_options, extra_line_options, styles)

                # Draw lines
                style = marker + line
                if not style:  # No point adding air to the legend.
                    continue

                xs, ys = zip(*sorted(zip(xs,ys)))
                if do.logx and do.logy:
                    main_ax.loglog(  xs, ys, style, c=colour, label=name, linewidth=do.curve_linewidth)
                elif do.logx:
                    main_ax.semilogx(xs, ys, style, c=colour, label=name, linewidth=do.curve_linewidth)
                elif do.logy:
                    main_ax.semilogy(xs, ys, style, c=colour, label=name, linewidth=do.curve_linewidth)
                else:
                    main_ax.plot(    xs, ys, style, c=colour, label=name, linewidth=do.curve_linewidth)

            if do.optional_line_at_y is not None:
                main_ax.hlines(do.optional_line_at_y, min_x, max_x,
                               colors='b', linestyles='dotted')

            if do.x_label:
                main_ax.set_xlabel(do.x_label)
            if do.y_label:
                main_ax.set_ylabel(do.y_label)
            if do.legend_position:  # Can be None or "" to turn it off.
                main_ax.legend(loc=do.legend_position)

            if do.y_lims:
                main_ax.set_ylim(do.y_lims[0], do.y_lims[1])

            # FIXME: Known issue: you can't make 5 x 10^1 show as a tick label.
            #        I even found a SO post where the plot has exactly that issue: https://stackoverflow.com/q/49750107/9352077
            if do.logx:
                if not do.logx_becomes_linear_at:
                    main_ax.set_xscale("log")
                    main_ax.xaxis.set_major_locator(tkr.LogLocator(base=10, numticks=999, subs=list(range(1, 10)) if do.tick_log_multiples else [1]))
                    main_ax.xaxis.set_major_formatter(tkr.LogFormatterSciNotation() if do.tick_scientific_notation else tkr.ScalarFormatter())
                else:
                    main_ax.set_xscale("symlog", linthresh=do.logx_becomes_linear_at, linscale=0.5)
            elif do.x_ticks_hardcoded:
                main_ax.xaxis.set_ticks(do.x_ticks_hardcoded, minor=False)
            elif do.x_tickspacing:
                main_ax.xaxis.set_major_locator(tkr.MultipleLocator(do.x_tickspacing))
                main_ax.xaxis.set_major_formatter(tkr.LogFormatterSciNotation() if do.tick_scientific_notation else tkr.ScalarFormatter())

            if do.logy:
                main_ax.set_yscale("log")
                main_ax.yaxis.set_major_locator(tkr.LogLocator(base=10, numticks=999, subs=list(range(1,10)) if do.tick_log_multiples else [1]))
                main_ax.yaxis.set_major_formatter(tkr.LogFormatterSciNotation() if do.tick_scientific_notation else tkr.ScalarFormatter())
                # main_ax.yaxis.set_minor_locator(tkr.LogLocator(base=10, subs="all"))
                # main_ax.yaxis.set_minor_formatter(tkr.LogFormatterSciNotation())
            elif do.y_tickspacing:
                main_ax.yaxis.set_major_locator(tkr.MultipleLocator(do.y_tickspacing))
                main_ax.yaxis.set_major_formatter(tkr.LogFormatterSciNotation() if do.tick_scientific_notation else tkr.ScalarFormatter())

            if do.y_lims:  # Yes, twice. Don't ask.
                main_ax.set_ylim(do.y_lims[0], do.y_lims[1])

            if do.x_gridspacing or do.y_gridspacing:
                if do.x_gridspacing:
                    main_ax.xaxis.set_minor_locator(tkr.MultipleLocator(do.x_gridspacing))
                if do.y_gridspacing:
                    main_ax.yaxis.set_minor_locator(tkr.MultipleLocator(do.x_gridspacing))
                main_ax.grid(True, which='both', linewidth=do.grid_linewidth)

            if not do.do_spines:
                main_ax.spines['top'].set_visible(False)
                main_ax.spines['right'].set_visible(False)

            self.exportToPdf(fig, export_mode)
            if export_mode != ExportMode.SAVE_ONLY:
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
                           initial_style_idx=len(self.data), export_mode=ExportMode.RETURN_ONLY)

        # Drawing the legends is slightly tricky, see https://stackoverflow.com/a/54631364/9352077
        legend_1 = ax1.legend(loc='upper right')
        legend_1.remove()
        ax2.legend(loc='lower right')
        ax2.add_artist(legend_1)

        # At last, save.
        Visual.writeFigure(fig, name, "." + FIJECT_DEFAULTS.RENDERING_FORMAT, overwrite_if_possible=self.overwrite)

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

    ##########################

    @staticmethod
    def _newFigAx(global_options: ArgsGlobal):
        fig, main_ax = newFigAx(global_options.aspect_ratio)
        main_ax.grid(True, which='major', linewidth=global_options.grid_linewidth if global_options.grid_linewidth is not None else FIJECT_DEFAULTS.GRIDWIDTH)
        main_ax.axhline(y=0, color='k', lw=0.5)
        return fig, main_ax

    @staticmethod
    def _makeLineStyleGenerator(advance_by: int=0):
        """
        The graph style is a tuple (col, line, marker) that cycles from front to back:
          - red solid dot, blue solid dot, green solid dot
          - red dashed dot, blue dashed dot, green dashed dot
          - ...
          - red solid x, blue solid x, green solid x
          - ...
        """
        colours = niceColours()
        lines   = ["-", "--", ":"]
        markers = [".", "x", "+"]
        styles = itertools.cycle(itertools.product(markers, lines, colours))  # Note: itertools.product consumes its arguments, which is why you can't use an infinite cycle for it.
        for _ in range(advance_by):
            next(styles)  # Advance the styles by the given amount (0 by default)
        return styles

    @staticmethod
    def _resolveLineStyle(name: str, is_function: bool, default_style: ArgsPerLine, extra_styles: Dict[str, ArgsPerLine], impute_generator) -> Tuple[str, str, str]:
        options = extra_styles.get(name, default_style)
        marker = "" if not options.show_points or is_function else options.point_marker
        line   = "" if not options.show_line else options.line_style
        colour = options.colour
        if marker is None or line is None or colour is None:
            def_marker, def_line, def_colour = next(impute_generator)
            marker = marker if marker is not None else def_marker  # Note that you can't use "marker = marker or def_marker" because an empty string also triggers tha or!
            line   = line   if line   is not None else def_line
            colour = colour if colour is not None else def_colour
        return marker, line, colour


class MergedLineGraph(Visual):
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

    def commit(self, aspect_ratio=None, x_label="", y_label_left="", y_label_right=""):
        with ProtectedData(self):
            ######## ALREADY IN .COMMIT
            # First graph
            colours = cycleNiceColours()
            fig, ax1 = newFigAx(aspect_ratio)
            ax1.grid(True, which='both')
            ax1.axhline(y=0, color='k', lw=0.5)

            for name, samples in self.g1.data.items():
                ax1.plot(samples[0], samples[1], c=next(colours), marker=".", linestyle="-", label=name)
            ########

            # Second graph
            ax2 = ax1.twinx()  # "Twin x" means they share the same figure and x axis, but the other's y axis will be on the right.

            ######## ALREADY IN .COMMIT
            for name, samples in self.g2.data.items():
                ax2.plot(samples[0], samples[1], c=next(colours), marker=".", linestyle="-", label=name)

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


class _PrecomputedStochasticLineGraph(Visual):

    @dataclass
    class ArgsGlobal(LineGraph.ArgsGlobal):
        uncertainty_opacity: float = 0.0
        twosided_ci_percentage: float = None

    def _commit(self, global_options: ArgsGlobal, default_line_options: LineGraph.ArgsPerLine, extra_line_options: Dict[str,LineGraph.ArgsPerLine],
                series_to_x_to_n_w_mu_sigma: Dict[str,Dict[float,Tuple[int,float,float,float]]],
                export_mode: ExportMode=ExportMode.SAVE_ONLY, existing_figax: tuple=None):
        if extra_line_options is None:
            extra_line_options = dict()

        if existing_figax is None:
            fig, main_ax = LineGraph._newFigAx(global_options)
        else:
            fig, main_ax = existing_figax

        overlay_graph = LineGraph("-", caching=CacheMode.NONE)
        new_line_options = dict()
        styles = LineGraph._makeLineStyleGenerator(advance_by=global_options.initial_style_idx)
        for name, samples in series_to_x_to_n_w_mu_sigma.items():
            # Get style options
            marker, line, colour = LineGraph._resolveLineStyle(name, False, default_line_options, extra_line_options, styles)
            options = extra_line_options.get(name, default_line_options)
            new_line_options[name] = LineGraph.ArgsPerLine(
                show_line=options.show_line, show_points=options.show_points,
                point_marker=marker, line_style=line, colour=colour
            )

            # Turn samples into a plottable line.
            sorted_input = []
            average_line = []
            upper_deviation_line = []
            for x, (_, w, mu, sigma) in sorted(samples.items()):
                if global_options.twosided_ci_percentage:
                    remainder_percentage = 100 - global_options.twosided_ci_percentage  # E.g. 5% for 95% CI
                    one_side_remainder = remainder_percentage / 2  # => Each side outside the CI captures 2.5%
                    alpha = (100 - one_side_remainder) / 100  # => Alpha is 0.975.
                    distribution: scipy.stats.rv_continuous = scipy.stats.t(w-1)
                    quantile = distribution.ppf(alpha)
                    deviation = quantile * sigma/sqrt(w)
                else:
                    deviation = sigma

                sorted_input.append(x)
                average_line.append(mu)
                upper_deviation_line.append(deviation)

            sorted_input, average_line, upper_deviation_line = np.array(sorted_input), np.array(average_line), np.array(upper_deviation_line)

            # Plotting
            if options.show_line:
                main_ax.fill_between(sorted_input, average_line + upper_deviation_line,
                                     average_line - upper_deviation_line,
                                     color=colour, alpha=global_options.uncertainty_opacity)
            if options.show_points:
                main_ax.errorbar(sorted_input, average_line, yerr=upper_deviation_line, color=colour, fmt='none',
                                 elinewidth=0.5, capthick=0.5, capsize=0.75, alpha=1.0)
            overlay_graph.addMany(name, sorted_input, average_line)

        # Above, the extra_line_options were already consulted to construct the new_line_options. This loop is for lines that weren't drawn yet (e.g. functions).
        for name, options in extra_line_options.items():
            if name not in new_line_options:
                new_line_options[name] = options

        fig, main_ax = overlay_graph.commitWithArgs(global_options, default_line_options, new_line_options,
                                                    export_mode=ExportMode.RETURN_ONLY, existing_figax=(fig, main_ax))

        self.exportToPdf(fig, export_mode)
        if export_mode != ExportMode.SAVE_ONLY:
            return fig, main_ax


class StochasticLineGraph(_PrecomputedStochasticLineGraph):
    """
    Same as a line graph except each line is now a collection of sequences whose average is plotted.
    Allows plotting uncertainty intervals.
    """

    def _load(self, saved_data: dict):
        for series_name, samples in saved_data.items():
            if not isinstance(samples, list) \
                    or any(len(tup) != 3 for tup in samples) \
                    or any(not isinstance(ys, list) or not isinstance(ws, list)
                           or len(ys) == 0 or len(ws) == 0
                           or len(ys) != len(ws) for _,ys,ws in samples):
                raise ValueError("Graph data corrupted: either samples aren't stored as a dictionary, or at least one input has zero output samples.")

            for x, ys, ws in samples:
                for y,w in zip(ys, ws):
                    self.addSample(series_name, x, y, weight=w)

    def addSample(self, series_name: str, x: float, y: float, weight: float=1.0):
        if type(x) == str or type(y) == str:
            warnings.warn("WARNING: You are trying to use a string as x or y data. The datapoint was discarded, because this causes nonsensical graphs.")
            return
        if weight % 1 != 0.0:
            warnings.warn(f"WARNING: You are adding a value to this stochastic graph with a non-integer weight ({weight}). This will mean standard deviation is not computed correctly, see https://stats.stackexchange.com/q/6534/360389.")

        if series_name not in self.data:
            self.cache[series_name] = dict()  # {input value} -> index in data list
            self.data[series_name] = []

        if x not in self.cache[series_name]:
            self.cache[series_name][x] = len(self.data[series_name])
            self.data[series_name].append( (x,[],[]) )

        index_of_x = self.cache[series_name][x]
        self.data[series_name][index_of_x][1].append(y)
        self.data[series_name][index_of_x][2].append(weight)

    def commit(self, global_options: "StochasticLineGraph.ArgsGlobal", default_line_options: LineGraph.ArgsPerLine, extra_line_options: Dict[str,LineGraph.ArgsPerLine]=None,
               export_mode: ExportMode=ExportMode.SAVE_ONLY, existing_figax: tuple=None):
        with ProtectedData(self):
            series_to_x_to_n_w_mu_sigma: Dict[str,Dict[float,Tuple[int,float,float,float]]] = defaultdict(dict)
            for name, samples in self.data.items():
                # Turn the samples into a plottable line.
                for x, ys, ws in samples:
                    n      = len(ys)
                    w      = sum(ws)
                    Ybar_n = weightedMean(ys, ws)
                    S_n    = sqrt(weightedVariance(ys, ws, ddof=1))  # S_n² is an unbiased estimator of the variance.

                    series_to_x_to_n_w_mu_sigma[name][x] = (n,w,Ybar_n,S_n)

            self._commit(
                global_options=global_options, default_line_options=default_line_options, extra_line_options=extra_line_options,
                series_to_x_to_n_w_mu_sigma=series_to_x_to_n_w_mu_sigma, export_mode=export_mode, existing_figax=existing_figax
            )


class StreamingStochasticLineGraph(_PrecomputedStochasticLineGraph):

    def _load(self, saved_data: dict):
        for series_name, samples in saved_data.items():
            if not isinstance(samples, list) \
                    or any(len(tup) != 5 for tup in samples) \
                    or any(not isinstance(x,  (int,float)) or
                           not isinstance(n,   int) or
                           not isinstance(w,  (int,float)) or
                           not isinstance(ls, (int,float)) or
                           not isinstance(ss, (int,float)) for x,n,w,ls,ss in samples):
                raise ValueError("Graph data corrupted. Check the assertions above this line to see what could be wrong.")

            self.cache[series_name] = {x: i for i,(x,_,_,_,_) in enumerate(samples)}

        self.data = saved_data

    def addSample(self, series_name: str, x: float, y: float, weight: float=1.0):
        if type(x) == str or type(y) == str:
            print("WARNING: You are trying to use a string as x or y data. The datapoint was discarded, because this causes nonsensical graphs.")
            return
        if weight % 1 != 0.0:
            warnings.warn(f"WARNING: You are adding a value to this stochastic graph with a non-integer weight ({weight}). This will mean standard deviation is not computed correctly, see https://stats.stackexchange.com/q/6534/360389.")

        if series_name not in self.data:
            self.cache[series_name] = dict()  # {input value} -> index in data list.
            self.data[series_name] = []

        if x not in self.cache[series_name]:
            self.cache[series_name][x] = len(self.data[series_name])
            self.data[series_name].append( [x,0,0,0,0] )

        index_of_x = self.cache[series_name][x]
        self.data[series_name][index_of_x][1] += 1
        self.data[series_name][index_of_x][2] += weight
        self.data[series_name][index_of_x][3] += weight*y
        self.data[series_name][index_of_x][4] += weight*y**2

    def commit(self, global_options: "StreamingStochasticLineGraph.ArgsGlobal", default_line_options: LineGraph.ArgsPerLine, extra_line_options: Dict[str,LineGraph.ArgsPerLine]=None,
               export_mode: ExportMode=ExportMode.SAVE_ONLY, existing_figax: tuple=None):
        with ProtectedData(self):
            series_to_x_to_n_w_mu_sigma: Dict[str,Dict[float,Tuple[int,float,float,float]]] = defaultdict(dict)
            for name, samples in self.data.items():
                for x, n, w, ls, ss in samples:
                    Ybar_n = ls/w
                    S_n    = sqrt(1/(w-1) * (ss - w*Ybar_n**2))  # Not correct when w is a float.
                    series_to_x_to_n_w_mu_sigma[name][x] = (n,w,Ybar_n,S_n)

            self._commit(
                global_options=global_options, default_line_options=default_line_options, extra_line_options=extra_line_options,
                series_to_x_to_n_w_mu_sigma=series_to_x_to_n_w_mu_sigma, export_mode=export_mode, existing_figax=existing_figax
            )
