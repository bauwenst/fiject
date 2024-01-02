from fiject import Table, CacheMode


def example_table():
    table = Table("test", caching=CacheMode.NONE)
    table.set(3.14,["Dutch", "BPE", "base"], ["sizes", "$|V|$"])
    table.set(15,  ["Dutch", "BPE", "base"], ["sizes", "$|M|$"])

    table.set(92,  ["Dutch", "BPE", "base"], ["morphemes", "unweighted", "Pr"])
    table.set(6.5, ["Dutch", "BPE", "base"], ["morphemes", "unweighted", "Re"])
    table.set(35,  ["Dutch", "BPE", "base"], ["morphemes", "unweighted", "$F_1$"])
    table.set(8.9, ["Dutch", "BPE", "base"], ["morphemes", "weighted", "Pr"])
    table.set(79,  ["Dutch", "BPE", "base"], ["morphemes", "weighted", "Re"])
    table.set(3.2, ["Dutch", "BPE", "base"], ["morphemes", "weighted", "$F_1$"])

    table.set(3.8, ["Dutch", "BPE", "base"], ["inbetween", "left"])
    table.set(46,  ["Dutch", "BPE", "knockout"], ["inbetween", "right"])

    table.set(26,  ["English", "BPE", "base"], ["lexemes", "Pr"])
    table.set(4.3, ["English", "BPE", "base"], ["lexemes", "Re"])
    table.set(38,  ["English", "BPE", "base"], ["lexemes", "$F_1$"])
    table.set(3.2, ["English", "BPE", "base"], ["lexemes", "weighted", "Pr"])
    table.set(79,  ["English", "BPE", "knockout"], ["lexemes", "weighted", "Re"])
    table.set(5.0, ["English", "BPE", "knockout"], ["lexemes", "weighted", "$F_1$"])

    table.set(0.1, ["ULM", "base"], ["morphemes", "weighted", "Re"])
    table.set(0.2, ["ULM", "yuk"], ["morphemes", "weighted", "Pr"])

    # print(table.getAsColumn().getPaths())

    table.commit(borders_between_columns_of_level=[0,1])
