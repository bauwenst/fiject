# Fiject design notes
## Caching system
Ideally, we could design experiment functions like this:
```python
@figure(g=Graph("where-to-look"), use_cache=True)
def myexperiment(some, arguments, here):
    if not decorator.use_cache or not decorator.cacheExists():
        (...stuff executed only when use_cache is False or there is no cache...)
    decorator.g.commit(...)
```
The purpose would be to avoid having to write separate functions for CALCULATING and for FORMATTING
because .commit already formats and hence you would need to keep two .commit calls synchronised with each other
(every time you change one's arguments, you have to change the other's).

The alternative is to use a .commitData and then have a separate function call to format.
So all your experiments would look like
```python
if not existsdata("where-to-look"):
    myexperiment()  # runs Graph("where-to-look") and then .commitData()...
g = Graph.load("where-to-look")
g.format(...)
```
which still isn't nice because 1. all experiments use this same pattern (except for the visual arguments) and 2.
you need to keep the file name synchronised in at least two places.
You could simplify that setup though:
```python
    g = myexperiment()
    g.format(...)
```
with a decorator
```python
@figure(t=Graph, name="where-to-look", use_cache=True)
def myexperiment(some, arguments) -> Graph:
    g = Graph("where-to-look")
    ...
    return g
```
and the decorator itself would do the existence check. Note that above, the decorator was used to switch INSIDE the
function body, whilst here, it is used to decide whether or not to RUN the function.

Problem though: what if there are multiple figures in one call? Or what if you generate figures in a loop?

What you want to avoid is the following:
    - Doing work when a figure already exists.
    - Having two .commit calls in your code for the same data (one for an "initial commit", one for formatting).
One way to do this could be with a "with" clause:
```python
with CreateFigure(t=Graph, name="where-to-look", use_cache=True) as f:
    ...only entered when f doesn't exist yet...
f.commit(...)
```
f is indeed available after the "with". https://stackoverflow.com/a/52992404/9352077
Sadly we can't do this, since unlike a decorator, "with" bodies are always executed unless you edit the callstack.
https://stackoverflow.com/a/12594323/9352077
I'm also not sure if it would work anyway, since
```python
with figure_that_exists as f1, figure_that_doesn't_exist as f2
```
should execute the whole body (since you need to do all the work for f2), but might skip the body.

Really, what we want is a code block that can do something like this:
```python
g = {type}({arg})
if use_cache and g.exists():
    g.load()
else:
    {body of the block}
```
and then run g.commit(). The closest is with a decorator, but a decorator needs a whole function, not just a body,
which is a problem if you need TWO figures to be generated from ONE execution.

I guess you could indeed just use an 'if' with a method.
```python
g = Graph("name", use_cache=True)
if g.unavailable():
    ...
```
and for two graphs
```python
g1 = Graph("name", use_cache=True)
g2 = Graph("othername", use_cache=True)
if g1.unavailable() or g2.unavailable():
    ...
```
would have been nice to have this all in one statement. What wouldn't work is a walrus operator:
```python
if (g1 := Graph("name", use_cache=True)) is None or ...
```
because the question is not whether g1 is a Graph object. The question is whether it could be initialised with data
during construction.
Another thing you could do is to have the preloading take place outside of the constructor, which has some nice OOP
properties because you FIRST initialise the superclass attributes, THEN the subclass attributes, and only THEN does
the loader get called. If you call the loader in the superclass, the subclass properties haven't been set yet and
hence the loader can't access them. It's also just cleaner.
```python
g = Graph("name", use_cache=True)
if not g.attemptPreload():
    ...
```
In hindsight, it probably is best to just have a method and not have big wrappers or 'with' constructors. Imagine
you want to cache these graphs:
```python
g1 = Graph("name1", use_cache=True)
g2 = Graph("name2", use_cache=True)
doHardWorkNeededByBoth()
doHardWorkForG1()
g1.set(...)
doHardWorkForG2()
g2.set(...)
```
You want to avoid the per-graph hard work if unnecessary, so you need THREE statements for skipping:
```python
g1 = Graph("name1", use_cache=True)
g2 = Graph("name2", use_cache=True)
if g1.unavailable() or g2.unavailable():  # <-----
    doHardWorkNeededByBoth()
    if g1.unavailable():  # <-----
        doHardWorkForG1()
        g1.set(...)
    if g2.unavailable():  # <-----
        doHardWorkForG2()
        g2.set(...)
```
