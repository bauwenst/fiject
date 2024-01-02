from .general import *


class Bars(Diagram):
    """
    Multi-bar chart. Produces a chart with groups of bars on them: each group has the same amount of bars and the
    colours of the bars are in the same order per group.

    All the bars of the same colour are considered the same family. All the families must have the same amount of
    bars, which is equal to the amount of groups.
    """

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
               diagonal_labels=True, aspect_ratio=DEFAULT_ASPECT_RATIO,
               y_tickspacing: float=None, log_y: bool=False):
        """
        The reason that group names are not given beforehand is because they are much like an x_label.
        Compare this to the family names, which are in the legend just as with LineGraph and MultiHistogram.
        """
        with ProtectedData(self):
            fig, main_ax = newFigAx(aspect_ratio)
            main_ax: plt.Axes

            colours = getColours()
            group_locations = None
            for i, (bar_slice_family, slice_heights) in enumerate(self.data.items()):
                group_locations = group_spacing * np.arange(len(slice_heights))
                main_ax.bar(group_locations + bar_width*i, slice_heights, color=colours.pop(0), width=bar_width,
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
            main_ax.grid(True, axis="y", linewidth=DEFAULT_GRIDWIDTH)
            main_ax.legend()

            self.exportToPdf(fig)
