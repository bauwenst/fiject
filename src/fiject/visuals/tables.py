from ..general import *
from ..util.printing import lprint

import dataclasses
from dataclasses import dataclass
from typing import TypeVar, Generic, Any
LeafContent = TypeVar("LeafContent")


@dataclass
class NamedTree(Generic[LeafContent]):
    name: str
    children: List["NamedTree"]
    content: LeafContent=None  # Should be mutually exclusive with children; you either have children or you have data.

    @staticmethod
    def fromDict(asdict: dict):  # inverse of dataclasses.asdict
        assert "name" in asdict and "children" in asdict and "content" in asdict
        assert isinstance(asdict["children"], list)
        return NamedTree(asdict["name"], [NamedTree.fromDict(d) for d in asdict["children"]], asdict["content"])

    def isLeaf(self):
        return len(self.children) == 0

    def width(self) -> int:
        if self.isLeaf():
            return 1
        else:
            return sum([col.width() for col in self.children])

    def height(self) -> int:
        if self.isLeaf():
            return 1
        else:
            return 1 + max([col.height() for col in self.children])

    def getLeaves(self) -> List["NamedTree"]:
        if self.isLeaf():
            return [self]
        else:
            leaves = []
            for col in self.children:
                leaves.extend(col.getLeaves())
            return leaves

    def getPaths(self) -> List[List["NamedTree"]]:  # For each leaf returns the path from the root to it.
        if self.isLeaf():
            return [[self]]
        else:
            subpaths = []
            for child in self.children:
                subpaths.extend(child.getPaths())
            for subpath in subpaths:
                subpath.insert(0, self)
            return subpaths

    def setdefault(self, name_path: List[str], content_if_missing: LeafContent) -> "NamedTree":
        current_node = self
        was_missing = False
        for name in name_path:
            for child in current_node.children:
                if child.name == name:
                    current_node = child
                    break
            else:
                new_child = NamedTree(name, [], None)
                current_node.children.append(new_child)
                current_node = new_child
                was_missing = True

        if was_missing:
            current_node.content = content_if_missing

        return current_node

    def renameBranch(self, old_branch: List[str], new_branch: List[str]):
        nodes = [self]
        for name in old_branch:
            for child in nodes[-1].children:
                if child.name == name:
                    nodes.append(child)
                    break
            else:
                print("No such tree path exists:", old_branch)
                return

        for node, name in zip(nodes, new_branch):  # If proposed name is too long, only the first names will be applied. If it is too short, only the first nodes will be renamed.
            node.name = name

    def __repr__(self):
        return "Column('" + self.name + "')"


TableRow    = NamedTree[int]
TableColumn = NamedTree[Dict[int, Any]]


@dataclass
class ColumnStyle:
    # Column-wide
    alignment: str="c"
    aggregate_at_rowlevel: int=-1  # Follows the same indexing standard as row borders. -1 computes extrema across all rows.
    do_bold_maximum: bool=False  # This is applied AFTER the cell functions and BEFORE rounding.
    do_bold_minimum: bool=False  # idem
    do_deltas: bool=False  # If true, will output the first row of the group as-is, and for the others, the difference with that row.

    # Cellwise. E.g.: to format a tokeniser's vocabulary size, you'd use function=lambda x: x/1000, digits=1, suffix="k"
    cell_prefix: str=""
    cell_function: Callable[[float], float] = lambda x: x   # E.g. x/1000
    digits: int=2  # This option might seem redundant given that we allow applying any function, but it takes the burden off the user to apply either round() (which drops zeroes) or something like f"{x:.2f}".
    cell_suffix: str=""

RowGroupKey = Tuple[str,...]

@dataclass
class RowGroupInColumn:
    # Bolding
    min: float
    max: float

    # Deltas
    id_of_first: int     # identifier of the row that appear first in the group (so you don't compute a delta for it)
    value_of_first: int  # its value (cached so that you're not re-applying the cell function for every delta)


class Table(Diagram):
    """
    Structure with named rows and infinitely many nested named columns, AND with an order for all columns and all rows.
    A good example of this kind of table is the morphemic-lexemic unweighted-weighted Pr-Re-F1 tables in my thesis.

    You could choose to either store a dictionary from row to a nested dictionary of all columns it has a value at (and
    that value), or store a nested dictionary of all columns containing as leaves a dictionary from row to value.
    The former is more intuitive when indexing (you start with the row), but actually, it makes way more sense to
    only store the structure of the table once.

    With nested rows, here's the format of the table:
        - One dictionary stores the tree of row names, where leftmost in the tree is topmost in the table.
          The leaves contain a unique integer identifier for the path that leads there.
        - Another dictionary stores the tree of column names. The leaves contain a dictionary from row ID to a value.
    The identifier system allows inserting rows out of their desired order, without having to rename all column content.
    """

    def clear(self):
        self.data = {
            "rows": 0,
            "column-tree": TableColumn("", [], dict()),
            "row-tree": TableRow("", [], None)
        }

    def getAsColumn(self) -> TableColumn:
        return self.data["column-tree"]

    def getRowTree(self) -> TableRow:
        return self.data["row-tree"]

    def set(self, value: float, row_path: List[str], column_path: List[str]):
        if not column_path:
            raise ValueError("Column path needs at least one column.")
        if not self.data:
            self.clear()

        # Get row identifier
        row_leaf = self.getRowTree().setdefault(row_path, -1)
        if row_leaf.content == -1:
            self.data["rows"] += 1
            row_leaf.content = self.data["rows"]

        # Get column leaf
        col_leaf = self.getAsColumn().setdefault(column_path, dict())
        col_leaf.content[row_leaf.content] = value

    def renameRow(self, old_rowname: List[str], new_rowname: List[str]):
        self.getRowTree().renameBranch(old_rowname, new_rowname)

    def renameColumn(self, old_colname: List[str], new_colname: List[str]):
        self.getAsColumn().renameBranch(old_colname, new_colname)

    def _save(self) -> dict:
        return {
            "rows": self.data["rows"],
            "column-tree": dataclasses.asdict(self.getAsColumn()),
            "row-tree":    dataclasses.asdict(self.getRowTree())
        }

    def _load(self, saved_data: dict):
        self.data = {
            "rows": saved_data["rows"],
            "column-tree": TableColumn.fromDict(saved_data["column-tree"]),
            "row-tree":    TableRow.fromDict(saved_data["row-tree"])
        }
        # Note: JSON converts the integer keys that index a column's content into strings. We need to convert back.
        for leaf_column in self.data["column-tree"].getLeaves():
            leaf_column.content = {int(key): value for key,value in leaf_column.content.items()}

    def commit(self, rowname_alignment="l",
               borders_between_columns_of_level: List[int]=None, borders_between_rows_of_level: List[int]=None,
               default_column_style: ColumnStyle=None, alternate_column_styles: Dict[Tuple[str,...], ColumnStyle]=None,
               do_hhline_syntax=True):  # TODO: Needs an option to align &s. Also needs to replace any & in col/row names by \&.
        """
        :param rowname_alignment: How to align row names (choose between "l", "c" and "r").
        :param borders_between_columns_of_level: List of layer indices that cause vertical lines to be drawn in the table
                                                 when a new column starts at that layer of the table header.
                                                 The top layer is layer 0, the under it is layer 1, etc.
        :param borders_between_rows_of_level: Same but for horizontal lines drawn when a new row of a certain layer starts.
                                              The leftmost layer is layer 0.
        :param default_column_style: The style to apply to all columns.
        :param alternate_column_styles: Specifies specific columns to which a different style should be applied.
        """
        with ProtectedData(self):
            table = self.getAsColumn()
            header_height = table.height() - 1
            margin_depth  = self.getRowTree().height() - 1

            # Style imputation
            if default_column_style is None:
                default_column_style = ColumnStyle()
            if alternate_column_styles is None:
                alternate_column_styles = dict()
            if borders_between_columns_of_level is None:
                borders_between_columns_of_level = []
            elif len(borders_between_columns_of_level) > 0 and (min(borders_between_columns_of_level) < 0 or max(borders_between_columns_of_level) >= header_height):
                raise ValueError(f"This table has {header_height} header levels, with identifiers 0 to {header_height-1}. You gave {borders_between_columns_of_level}.")
            if borders_between_rows_of_level is None:
                borders_between_rows_of_level = []
            elif len(borders_between_rows_of_level) > 0 and (min(borders_between_rows_of_level) < 0 or max(borders_between_rows_of_level) >= margin_depth):
                raise ValueError(f"This table has {margin_depth} row levels, with identifiers 0 to {margin_depth-1}. You gave {borders_between_rows_of_level}.")

            # STEP 1: Make first line. Note that there are no default borders (indicated with | normally). Everything is regulated by multicolumn below.
            first_line = r"\begin{tabular}{" + rowname_alignment*margin_depth + "||"
            for path in table.getPaths():
                identifier = tuple(node.name for node in path[1:])
                style = alternate_column_styles.get(identifier, default_column_style)
                first_line += style.alignment
            # for top_level_column in table.children:
            #     first_line += default_column_style.alignment*top_level_column.width()
            first_line += "}"

            # STEP 2: Get all header lines and where the borders are at each header level
            header_lines = []

            level_has_edge_after_ncols = []
            frontier = table.children
            for header_line_idx in range(header_height):  # Vertical iteration
                line = "&"*(margin_depth-1)
                level_has_edge_after_ncols.append([0])
                cumulative_width = 0
                new_frontier = []
                for frontier_idx, col in enumerate(frontier):  # Horizontal iteration
                    line += " & "
                    width = col.width()
                    cumulative_width += width
                    if col.height() >= header_height-header_line_idx:  # This is where you enter all columns on the same header level. Very useful.
                        new_frontier.extend(col.children)

                        # Is this level one with borders, or does it have a border for a previous level, or neither?
                        right_border = False
                        left_border  = False
                        if header_line_idx in borders_between_columns_of_level:
                            left_border  = frontier_idx != 0 and level_has_edge_after_ncols[-1][-1] != 0  # No left border at the start and also not if a right border was just placed in that position.
                            right_border = frontier_idx != len(frontier)-1  # No right border at the end of the table.
                        elif frontier_idx != len(frontier)-1:  # In this case, you may still inherit a border.
                            for level in borders_between_columns_of_level:
                                if level >= header_line_idx:  # Only take into account levels strictly smaller than this one
                                    continue
                                if cumulative_width in level_has_edge_after_ncols[level]:
                                    right_border = True
                                    break

                        if left_border:  # We know that the cell to the left is empty. Go back and change it to an empty cell with a right border. (What you cannot do is add a left border to the current cell, despite multicolumn allowing this (e.g. |c| instead of c|). The reason is that a right border in cell x and a left border in cell x+1 are offset by 1 pixel.)
                            line = line[:line.rfind("&")] + r"\multicolumn{1}{c|}{}  &"

                        # Render content
                        if width == 1 and not right_border:  # Simple
                            line += col.name
                        else:  # Multicolumn width and/or border
                            line += r"\multicolumn{" + str(width) + "}{c" + "|"*right_border + "}{" + col.name + "}"

                        # Border math
                        if level_has_edge_after_ncols[-1][-1] != 0:
                            level_has_edge_after_ncols[-1].append(width)
                        else:
                            level_has_edge_after_ncols[-1][-1] = width
                        level_has_edge_after_ncols[-1].append(0)
                    else:  # Column starts lower in the table. Re-schedule it for rendering later.
                        new_frontier.append(col)

                        line += " & "*(width-1)
                        level_has_edge_after_ncols[-1][-1] += width
                line += r" \\"
                header_lines.append(line)

                level_has_edge_after_ncols[-1] = level_has_edge_after_ncols[-1][:-2]  # Trim off last 0 and also last column since we don't want the edge of the table to have a border.
                level_has_edge_after_ncols[-1] = [sum(level_has_edge_after_ncols[-1][:i+1]) for i in range(len(level_has_edge_after_ncols[-1]))]  # cumsum
                frontier = new_frontier
            header_lines[-1] += r"\hline\hline" if not do_hhline_syntax else \
                                r"\hhline{*{" + str(margin_depth+table.width()) + r"}{=}}"

            # STEP 3: Find maximal and minimal values per column, possibly per row group (which differs per column!)
            aggregates_per_column: List[Dict[RowGroupKey, RowGroupInColumn]] = []  # List over all columns, dict over all group keys.
            groupkeys_per_columns: List[Dict[int,RowGroupKey]] = []  # List over all columns, dict over all row identifiers.
            for column_path in table.getPaths():
                col_path_names = tuple(node.name for node in column_path[1:])
                style = alternate_column_styles.get(col_path_names, default_column_style)
                content_node = column_path[-1]

                # This overlaps in work with step 4, but I don't really have a decent alternative.
                aggregates_per_column.append(dict())
                groupkeys_per_columns.append(dict())
                for row_path in self.getRowTree().getPaths():
                    # Determine value of this row in the current column
                    identifier = row_path[-1].content
                    if identifier not in content_node.content:
                        continue
                    else:
                        cell_value = content_node.content[identifier]
                        if isinstance(cell_value, (int, float)):
                            cell_value = style.cell_function(cell_value)

                    # Get row's group key in this column
                    row_path_names = tuple(node.name for node in row_path[1:])
                    group_key = row_path_names[:style.aggregate_at_rowlevel+1]
                    groupkeys_per_columns[-1][identifier] = group_key

                    # Update aggregates
                    if group_key not in aggregates_per_column[-1]:  # Note: I'm not using the classic approach of using .get(key, float(inf)) because a table can also contain strings, which can be compared but not with floats.
                        aggregates_per_column[-1][group_key] = RowGroupInColumn(
                            min=cell_value,
                            max=cell_value,
                            id_of_first=identifier,
                            value_of_first=cell_value
                        )
                    else:
                        groupdata = aggregates_per_column[-1][group_key]
                        groupdata.min = min(cell_value, groupdata.min)
                        groupdata.max = max(cell_value, groupdata.max)

            # STEP 4: Make rows
            body_lines = []
            prev_names = ["" for _ in range(margin_depth)]
            for row_idx, row_path in enumerate(self.getRowTree().getPaths()):  # Vertical iteration: for row in rows
                line = ""

                # 4.1: Row name.
                row_path = [None for _ in range(margin_depth-len(row_path)+1)] + row_path[1:]
                row_path_changed = False  # Has to become True at some point
                cline_start = None
                for row_depth_idx, node in enumerate(row_path):  # Horizontal iteration: for namepart in row
                    if row_depth_idx != 0:
                        line += " & "

                    name = node.name if node is not None else None
                    if prev_names[row_depth_idx] != name:
                        row_path_changed = True
                        prev_names[row_depth_idx] = name

                    if row_path_changed and node is not None:  # Reprint every on the path if a parent changed, even if it hasn't changed since the row above.
                        width = node.width()
                        if width > 1:
                            line += r"\multirow{" + str(width) + "}{*}{" + node.name + "}"
                        else:
                            line += node.name

                    if row_path_changed and cline_start is None and row_depth_idx in borders_between_rows_of_level:  # "Start the border on the earliest depth where a change occurred and that needs a border"
                        cline_start = row_depth_idx+1  # \cline is 1-based

                if row_idx != 0 and cline_start is not None:
                    body_lines[-1] += r"\cline{" + f"{cline_start}-{margin_depth+table.width()}" + "}" if not do_hhline_syntax else \
                                      r"\hhline{" + "~"*(cline_start-1) + r"*{" + str(margin_depth+table.width()-cline_start+1) + r"}{-}}"

                # 4.2: Row body.
                for col_idx, col_path in enumerate(table.getPaths()):
                    column_content = col_path[-1].content

                    # Is there a border here?
                    right_border = False
                    for level in borders_between_columns_of_level:
                        if col_idx+1 in level_has_edge_after_ncols[level]:
                            right_border = True
                            break

                    # Get column style
                    column_path_names = tuple(node.name for node in col_path[1:])
                    style = alternate_column_styles.get(column_path_names, default_column_style)

                    row_identifier = row_path[-1].content
                    if row_identifier in column_content:
                        # Get the cell value and its group aggregates
                        cell_value       = column_content[row_identifier]
                        group_aggregates = aggregates_per_column[col_idx][groupkeys_per_columns[col_idx][row_identifier]]

                        # Process value: apply cell function, subtract reference (optionally), and round.
                        is_relative = style.do_deltas and group_aggregates.id_of_first != row_identifier
                        if isinstance(cell_value, (int, float)):
                            # Compute value
                            cell_value = style.cell_function(cell_value)
                            if is_relative:
                                cell_value -= group_aggregates.value_of_first
                            # Format value
                            cell_string = f"{cell_value:.{style.digits}f}"
                            if is_relative and cell_value >= 0:
                                cell_string = "+" + cell_string
                        else:
                            cell_string = str(cell_value)

                        # Compare value
                        bolded = (style.do_bold_minimum and cell_value == group_aggregates.min) or \
                                 (style.do_bold_maximum and cell_value == group_aggregates.max)

                        cell_content = r"\bfseries"*bolded + style.cell_prefix + cell_string + style.cell_suffix
                    else:
                        cell_content = ""

                    if not right_border:
                        line += " & " + cell_content  # No alignment needed, since it is set in the table header.
                    else:
                        line += " & " + r"\multicolumn{1}{" + style.alignment + "|}{" + cell_content + "}"
                line += r" \\"
                body_lines.append(line)
            body_lines[-1] = body_lines[-1][:-2]  # Strip off the \\ at the end.

            # Last line
            last_line = r"\end{tabular}"

            # Construct table
            header_prefix = max(line.find("&") for line in body_lines)
            lines = [first_line] + \
                    ["\t" + " "*header_prefix + line for line in header_lines] + \
                    ["\t" + line for line in body_lines] + \
                    [last_line]

            print(f"Writing .tex {self.name} ...")
            with open(PathHandling.getSafePath(PathHandling.getProductionFolder(), self.name, ".tex"), "w") as file:
                file.write("\n".join(lines))

            lprint(lines)

    @staticmethod
    def getLaTeXpreamble(include_cellgradients: bool=False) -> str:
        """
        :param include_cellgradients: includes code that defines a "table gradient" (tgrad) macro, which, when put
                                      around a number in a table cell, will colour that cell using a gradient.
        """
        return r"""
        \usepackage{multirow}
        \usepackage[table]{xcolor}
        \usepackage{arydshln}
        \usepackage{hhline}  % There's a fatal flaw in arydshln's partial table lines, \cline and \cdashline, namely that they don't move cells down to make a gap into which to insert their line, unlike \hline. As a result, coloured cells cover those lines (see https://tex.stackexchange.com/a/603623/203081). A solution is using \hhline syntax (https://tex.stackexchange.com/a/121477/203081).
        """ + \
        include_cellgradients*\
        r"""
        \usepackage{etoolbox}
        \usepackage{pgf}
        \usepackage{xargs}
        
        % Colours
        \definecolor{high}{HTML}{03AC13}
        \definecolor{mid}{HTML}{F7E379}
        \definecolor{low}{HTML}{ec462e}
        \newcommand*{\opacity}{80}
        
        % Cell command
        \newcommandx{\tgrad}[4][1=0.0, 2=0.5, 3=1.0]{%
            \ifdim #4 pt > #2 pt%
                \pgfmathparse{max(min(100.0*(#4-#2)/(#3-#2),100.0),0)}%
                \xdef\PercentColor{\pgfmathresult}%
                \cellcolor{high!\PercentColor!mid!\opacity}#4%
            \else
                \pgfmathparse{max(min(100.0*(#2-#4)/(#2-#1),100.0),0)}%
                \xdef\PercentColor{\pgfmathresult}%
                \cellcolor{low!\PercentColor!mid!\opacity}#4%
            \fi
        }
        """