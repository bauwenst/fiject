import matplotlib.pyplot as plt


########################
### TO-BE-INTEGRATED ###
########################
def arrow(ax: plt.Axes, start_point, end_point):  # FIXME: I want TikZ's stealth arrows, but this only seems possible in Matplotlib's legacy .arrow() interface (which doesn't keep its head shape properly): https://stackoverflow.com/a/43379608/9352077
    """
    Matplotlib's arrow interface is impossibly complicated. This simplifies that.
    Based on:
    https://stackoverflow.com/a/52613154/9352077
    """
    prop = {
        "arrowstyle": "->",
        # "overhang": 0.2,
        # "headwidth": 0.4,
        # "headlength": 0.8,
        "shrinkA": 0,
        "shrinkB": 0,
        "linewidth": 0.4
    }
    ax.annotate("", xy=end_point, xytext=start_point, arrowprops=prop)


def simpleTable(TP, FN, FP, TN):
    tp_len = len(str(TP))
    tn_len = len(str(TN))
    fp_len = len(str(FP))
    fn_len = len(str(FN))

    column_1_size = max(tp_len, fp_len)
    column_2_size = max(fn_len, tn_len)

    s = "TP: " + " " * (column_1_size - tp_len) + f"{TP} | FN: " + " " * (column_2_size - fn_len) + f"{FN}\n" + \
        "FP: " + " " * (column_1_size - fp_len) + f"{FP} | TN: " + " " * (column_2_size - tn_len) + f"{TN}"
    return s


def latexTable(name, TP, FN, FP, TN, F1frac, MSEfrac, MSE_label: str = "MSE"):
    s = r"\begin{tabular}{rrcc}" + "\n" + \
        r"                     &                          & \multicolumn{2}{c}{$f(\vec X)$} \\" + "\n" + \
        r"      \multicolumn{2}{r}{" + name + r"}                   & \multicolumn{1}{|c|}{$P$}  & $N$ \\ \cline{2-4} " + "\n" + \
        r"      \multirow{2}{*}{$Y$} & \multicolumn{1}{r|}{$P$} & " + str(TP) + " & " + str(
        FN) + r"   \\ \cline{2-2}" + "\n" + \
        r"                           & \multicolumn{1}{r|}{$N$} & " + str(FP) + " & " + str(
        TN) + r"\\ \cline{2-4}" + "\n" + \
        r"                           & \multicolumn{3}{l}{$F_1 = " + str(
        round(F1frac * 100, 2)) + r"\%$}                        \\" + "\n" + \
        r"                           & \multicolumn{3}{l}{${\scalebox{0.75}{$" + MSE_label + r"$}} = " + str(
        round(MSEfrac * 100, 2)) + r"\%$}" + "\n" + \
        r"\end{tabular}"

    return s
