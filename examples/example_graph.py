from fiject import LineGraph, CacheMode

def example_linegraph():
    graph = LineGraph("test", caching=CacheMode.NONE)
    graph.addMany("a", [1,2,3,4,5,6,7,8,9,10], [5,4,8,3,7,9,5,4,8,6])
    graph.addMany("b", [1,2,3,4,5,6,7,8,9,10], [1,8,5,3,1,4,7,5,3,8])
    graph.commit()
