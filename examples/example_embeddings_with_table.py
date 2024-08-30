"""
Generate LaTeX code for random coloured embedding vectors,
using fiject.visuals.table's body_only rendering.
"""
import numpy.random as npr
from fiject import Table, ColumnStyle, ExportMode

RNG = npr.default_rng(3)


min_val = 0
max_val = 1

print(r"""
\definecolor{high}{HTML}{0080FF}
\definecolor{mid}{HTML}{FFFFFF}
\definecolor{low}{HTML}{EE8F35}

\setlength{\fboxsep}{0pt}
\renewcommand{\arraystretch}{1.5}
""")

t = Table("slide-embeddings")
for _ in range(5):
	t.clear()
	embedding = min_val + (max_val-min_val)*RNG.random(size=10)
	for i, e in enumerate(embedding):
		t.set(e, row_path=[0], column_path=[i])

	render = t.commit(body_only=True, export_mode=ExportMode.RETURN_ONLY,
					  default_column_style=ColumnStyle(cell_prefix=r"\tgrad" + f"[{min_val}][{min_val + 0.5*(max_val - min_val)}][{max_val}]" + "{", cell_suffix="}", digits=1))
	print(r"\fbox{\hspace{-4pt}")
	print(render)
	print(r"\hspace{-4pt}}")
	print()
	print(r"\vspace{1em}")
	print()
