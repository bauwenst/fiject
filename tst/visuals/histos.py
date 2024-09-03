from tst.preamble import *

from fiject.visuals.histos import StreamingHistogram, BinSpec, VariableGranularityHistogram


def test_streaming():
    samples = [1,4,9,16,25,36,49,64,81,100]  # Histogram: 4, 2, 1, 1, 2

    histo = StreamingHistogram("test-histo_streaming", BinSpec.halfopen(minimum=0, width=20))
    for sample in samples:
        histo.add(sample)

    print(histo.data)

    histo.commit(StreamingHistogram.ArgsGlobal(
        y_tickspacing=1,
        combine_buckets=1
    ))
    histo.commit(StreamingHistogram.ArgsGlobal(
        combine_buckets=2
    ))


def test_granularity():
    histo = VariableGranularityHistogram("test-histo_vg")

    histo.add(1,3)
    histo.commit(VariableGranularityHistogram.ArgsGlobal(
        n_bins=20
    ))

    histo.add(0,3)
    histo.commit(VariableGranularityHistogram.ArgsGlobal(
        n_bins=20
    ))

    histo.add(1,4)
    histo.commit(VariableGranularityHistogram.ArgsGlobal(
        n_bins=20
    ))

    histo.commit(VariableGranularityHistogram.ArgsGlobal(
        n_bins=24
    ))


if __name__ == "__main__":
    # test_streaming()
    test_granularity()