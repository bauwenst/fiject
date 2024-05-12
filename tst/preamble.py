from fiject import setFijectOutputFolder
from pathlib import Path

PATH_PREAMBLE = Path(__file__).resolve()
PATH_TESTS    = PATH_PREAMBLE.parent
PATH_ROOT     = PATH_TESTS.parent
PATH_DATA     = PATH_ROOT / "data"
PATH_DATA_OUT = PATH_DATA / "out"
PATH_DATA_OUT.mkdir(exist_ok=True, parents=True)

setFijectOutputFolder(PATH_DATA_OUT)
