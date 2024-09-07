from typing import List, TypeVar, Iterable

T = TypeVar("T")


def cat(lists: Iterable[List[T]]) -> List[T]:
    return sum(lists, start=[])
