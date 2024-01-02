# Fiject
Object-oriented, two-stage PDF figure generators in Python.

Gives an answer to *"How can I change the look of a figure without re-computing its data?"* which isn't possible in
`matplotlib` nor `seaborn`.

## Features
- Two-stage figure generation:
  - **Addition stage:** Incrementally add data to figure objects, rather than needing to do it all at once along with
                        visual parameters. Data caching allows skipping this step when a figure needs to be reformatted but not recomputed.
  - **Commit stage:** Format the data stored in a figure object, with a simple interface that hides the complexities of `matplotlib` and `seaborn`.
- Supported figure types:
  - `LineGraph`: points connected by lines.
  - `Bars`: bar plot on a categorical axis.
  - `MultiHistogram`: one or more histograms on the same numerical axis. Can also be committed to box plots.
  - `ScatterPlot`: unconnected points.
  - `Table`: LaTeX tables with hierarchical rows and hierarchical columns, column stylisation (e.g. rounding, min/max bolding, ...) and borders where you need them.

## Example
Let's say you have a machine learning project where a classifier called `model` is trained and then evaluated on precision (Pr), 
recall (Re) and F1 score, for 5 different values of a hyperparameter *h*. You would do your experiments as follows:
```python
from fiject import LineGraph, CacheMode

g = LineGraph("project-results", caching=CacheMode.IF_MISSING)
if g.needs_computation:
    h_values = [0.1, 0.25, 0.5, 0.7, 1.0]
    for h in h_values:
        # ...
        # model.trainModel(h)  # Takes hours to compute. We don't want to repeat it just to reformat the graph!
        # pr, re, f1 = model.evaluateYourModel()
        # ...
        g.add("Pr",    h, pr)
        g.add("Re",    h, re)
        g.add("$F_1$", h, f1)
              
g.commit(aspect_ratio=(4,3), y_lims=(0, 100), x_tickspacing=0.1, y_tickspacing=10,
         x_label="Hyperparameter value", y_label="Binary classification performance [\\%]")
```
A PDF `project-results_0.pdf` will appear, and the data will be cached in a file `project-results_0.json`.

Notice that the `CacheMode` along with the check `if g.needs_computation` will ensure that you don't
have to redo your computation if you don't like the way your figure came out the first time. You can
just change the parameters to `g.commit()` and re-run *the same code* to get a new PDF `project-results_1.pdf`
instantly.

## Installation
You can install `fiject` as any other package, or as a developer if you want to tinker with the source yourself.

### Normal install
Open a terminal and run:
```commandline
pip install git+https://github.com/bauwenst/fiject.git
```

### Developer install
Open a terminal and, instead of the above, run:
```commandline
git clone https://github.com/bauwenst/fiject.git
cd fiject
pip install -e .
```
The last command will detect the `pyproject.toml` file (`pip install`), look for the `fiject/__init__.py` file in the 
current directory (`.`), and put a symlink to this folder in Python's `site-packages` folder (`-e`). This means that
when you `import fiject`, it is imported from the current folder and hence any changes you make here are applied immediately.

## Credit
This package was developed over the span of multiple years (2021-2023) and across multiple research papers at university.
If you produce figures for your own reports with this package, please be a kind human and acknowledge my work by crediting
this repository in a footnote. For example, in LaTeX: 
```latex
\footnote{All figures were made using Fiject (\url{https://github.com/bauwenst/fiject}), 
          a Python package by ir.\ Thomas Bauwens.}
```

## Showcase
A collage of all the figures I have drawn with this code across many university projects.

### Line graphs
<p align="middle" >
  <img src="https://github.com/bauwenst/fiject/assets/145220868/59e49cab-55ad-466b-b6bb-a681f18088d8" width="40%" />⠀
  <img src="https://github.com/bauwenst/fiject/assets/145220868/23269b46-6786-40ae-a8c9-1b199939846f" width="52%" />
  <br>
  <br>
  <img src="https://github.com/bauwenst/fiject/assets/145220868/c520a27a-a8bd-483b-98a6-269648f5baff" height="300" />
</p>

### Histograms
<p align="middle">
  <img src="https://github.com/bauwenst/fiject/assets/145220868/97b31ed8-729e-4fcc-9ac6-396636a4f2eb" height="300" />
</p>

### Bar plots
<p align="middle">
  <img src="https://github.com/bauwenst/fiject/assets/145220868/837cb644-263c-4cae-a301-dff5736cf10f" height="400" />
</p>

### Scatterplots
<p align="middle">
  <img src="https://github.com/bauwenst/fiject/assets/145220868/119eba35-3116-469c-922a-f8d56d530cd0" height="400" />
</p>

### Tables
<p align="middle">
  <img src="https://github.com/bauwenst/fiject/assets/145220868/ce488b65-ddd5-4a82-9899-380431d7a5dc" height="300" />
</p>
