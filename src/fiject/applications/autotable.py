"""
Rather than manually setting the individual cells of a Fiject Table, this interface allows overriding only a few
methods in order to parse any file/folder into a Fiject Table automatically.

For example, if you have a bunch of experimental results from a Weights & Biases CSV export, you write methods that
extract which table row each CSV row belongs to (many CSV rows usually map to one table row), what columns the results
go in, and how to format that row and column in the final table.
"""
from typing import TypeVar, Generic, Iterator, Iterable, List, Tuple, Dict, Optional, Hashable, Sequence
from pathlib import Path
from abc import ABC, abstractmethod

from collections import defaultdict

from ..visuals.tables import Table


Instance = TypeVar("Instance")
RawRowKeys = TypeVar("RawRowKeys", bound=Hashable)
RawColKeys = TypeVar("RawColKeys", bound=Hashable)
SortableRowKeys = TypeVar("SortableRowKeys", bound=tuple)  # The "sortable" applies intra-key, not inter-key.
SortableColKeys = TypeVar("SortableColKeys", bound=tuple)

FormattedKeys = List[str]
Permutation = List[int]
AnySequence = TypeVar("AnySequence", bound=Sequence)


class AutoTable(ABC, Generic[Instance,RawRowKeys,RawColKeys,SortableRowKeys,SortableColKeys]):
    """
    Provides an implicit, subtractive approach to turning a set of results into a table.
    Basically: filter out all the results you don't want, sort what's left, and add to a table. That table cannot be any
    other table than the table you want.

    The alternative approach would be additive, where you select the results you do want by defining just literally the
    table's hierarchy using dictionaries, only ever querying what the explicitly known table structure requires.
    """

    @abstractmethod
    def _generateInstances(self, path: Path) -> Iterator[Instance]:
        """Uses some kind of path to generate the objects from which both input and output can be extracted."""
        pass

    @abstractmethod
    def _extractRowKey(self, raw: Instance) -> RawRowKeys:
        """Extracts a key from the instance which will be used both for sorting and for formatting of rows."""
        pass

    @abstractmethod
    def _extractColResults(self, raw: Instance) -> Dict[RawColKeys, float]:
        """Extracts key-value pairs where the keys will be used for sorting and formatting of columns."""
        pass

    @abstractmethod
    def _filterColResults(self, raw: Instance, parsed_results: Dict[RawColKeys, float]) -> Dict[RawColKeys, float]:
        """Produces a reduced set of relevant column results. It is expected that some filtering will have already
           been done in _extractColResults(), but there is a difference between unparsable and unwanted values."""
        pass

    @abstractmethod
    def _to_sortkey_row(self, key: RawRowKeys) -> SortableRowKeys:
        pass

    @abstractmethod
    def _to_sortkey_col(self, key: RawColKeys) -> SortableColKeys:
        pass

    @abstractmethod
    def _format_row(self, key: RawRowKeys) -> FormattedKeys:
        pass

    @abstractmethod
    def _format_col(self, key: RawColKeys) -> FormattedKeys:
        pass

    ####################################################################################################################

    def _permute_tuple(self, t: AnySequence, p: Optional[Permutation]) -> AnySequence:
        if p is None:
            return t
        elif len(t) != len(p):
            raise ValueError(f"Cannot permute tuple of length {len(t)} with permutation of length {len(p)}.")
        else:
            return t.__class__(t[i] for i in p)

    def _permute_sortkey_row(self, key: SortableRowKeys, level_permutation: Optional[Permutation]) -> SortableRowKeys:
        return self._permute_tuple(key, level_permutation)

    def _permute_sortkey_col(self, key: SortableColKeys, level_permutation: Optional[Permutation]) -> SortableColKeys:
        return self._permute_tuple(key, level_permutation)

    def _permute_formatted_row(self, key: FormattedKeys, level_permutation: Optional[Permutation]) -> FormattedKeys:
        return self._permute_tuple(key, level_permutation)

    def _permute_formatted_col(self, key: FormattedKeys, level_permutation: Optional[Permutation]) -> FormattedKeys:
        return self._permute_tuple(key, level_permutation)

    def _sortAndFormatRows(self, keys: Iterable[RawRowKeys], level_permutation: Optional[Permutation]) -> Iterator[Tuple[RawRowKeys,FormattedKeys]]:
        for raw_key in sorted(keys, key=lambda k: self._permute_sortkey_row(self._to_sortkey_row(k), level_permutation)):
            yield raw_key, self._permute_formatted_row(self._format_row(raw_key), level_permutation)  # TODO: Not necessarily true that len(sortkey) == len(formattedkey), so the permutation should be a separate one. However, what's the point of having a sortkey with more/fewer levels than the amount of row levels in the final table?

    def _sortAndFormatCols(self, keys: Iterable[RawColKeys], level_permutation: Optional[Permutation]) -> Iterator[Tuple[RawColKeys,FormattedKeys]]:
        for raw_key in sorted(keys, key=lambda k: self._permute_sortkey_col(self._to_sortkey_col(k), level_permutation)):
            yield raw_key, self._permute_formatted_col(self._format_col(raw_key), level_permutation)

    def _getResultsFromPath(self, path: Path) -> Dict[RawRowKeys, Dict[RawColKeys, float]]:
        results = defaultdict(dict)
        for raw in self._generateInstances(path):
            key  = self._extractRowKey(raw)
            cols = self._extractColResults(raw)
            cols = self._filterColResults(raw,cols)

            if key in results:
                columns_with_existing_result = list(set(results[key]) & set(cols))
                if columns_with_existing_result:
                    print("\nDuplicate data found for key:", key)
                    print("\tOld values (preserved):")
                    for col in columns_with_existing_result:
                        print(f"\t\t{col}: {results[key][col]}")
                    print("\tNew values (discarded):")
                    for col in columns_with_existing_result:
                        print(f"\t\t{col}: {cols[col]}")
                    for col in columns_with_existing_result:
                        cols.pop(col)
            results[key] |= cols
        return results

    def _tabulateResults(self, results: Dict[RawRowKeys, Dict[RawColKeys, float]], name: str,
                         row_level_permutation: Optional[Permutation], col_level_permutation: Optional[Permutation]) -> Table:
        table = Table(name, overwriting=True)
        for row_key, row_formatted in self._sortAndFormatRows(results.keys(), row_level_permutation):
            for col_key, col_formatted in self._sortAndFormatCols(results[row_key].keys(), col_level_permutation):
                result = results[row_key][col_key]
                table.set(result, row_formatted, col_formatted)
        return table

    def run(self, path: Path, stem_suffix: str="", row_level_permutation: Permutation=None, col_level_permutation: Permutation=None) -> Table:
        return self._tabulateResults(self._getResultsFromPath(path), path.stem + stem_suffix,
                                     row_level_permutation=row_level_permutation, col_level_permutation=col_level_permutation)


from csv import DictReader

class AutoTableFromCsv(AutoTable[dict,RawRowKeys,RawColKeys,SortableRowKeys,SortableColKeys]):
    def _generateInstances(self, path: Path) -> Iterator[Instance]:
        assert path.suffix == ".csv"
        with open(path, "r", encoding="utf-8") as handle:
            yield from DictReader(handle)
