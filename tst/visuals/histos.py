from tst.preamble import *
import time

from fiject.visuals.histos import StreamingMultiHistogram, BinSpec, VariableGranularityHistogram, StreamingVariableGranularityHistogram


def tst_streaming():
    samples = [1,4,9,16,25,36,49,64,81,100]  # Histogram if closed: 4, 2, 1, 1, 2. Histogram if open: 4, 2, 1, 1, 1, 1.

    histo = StreamingMultiHistogram("test-histo_streaming_" + time.strftime("%H%M%S"), BinSpec.halfopen(minimum=0, width=20))
    for sample in samples:
        histo.add("series 1", sample)

    print(histo.data)

    histo.commit(StreamingMultiHistogram.ArgsGlobal(
        y_tickspacing=1,
        combine_buckets=1
    ))
    histo.commit(StreamingMultiHistogram.ArgsGlobal(
        combine_buckets=2
    ))


def tst_granularity():
    histo = VariableGranularityHistogram("test-histo_vg_" + time.strftime("%H%M%S"))

    histo.add(1,3)
    histo.commit(VariableGranularityHistogram.ArgsGlobal(n_bins=20))

    histo.add(0,3)
    histo.commit(VariableGranularityHistogram.ArgsGlobal(n_bins=20))

    histo.add(1,4)
    histo.commit(VariableGranularityHistogram.ArgsGlobal(n_bins=20))

    histo.commit(VariableGranularityHistogram.ArgsGlobal(n_bins=24))

    # Same thing now, but now with streaming.

    histo = StreamingVariableGranularityHistogram("test-histo_vgstream_" + time.strftime("%H%M%S"), binspec=BinSpec.closedFromAmount(0,1,20))

    histo.add(1,3, class_name="a")
    histo.commit(StreamingVariableGranularityHistogram.ArgsGlobal())

    histo.add(0,3, class_name="a")
    histo.commit(StreamingVariableGranularityHistogram.ArgsGlobal())

    histo.add(1,4, class_name="a")
    histo.commit(StreamingVariableGranularityHistogram.ArgsGlobal())

    histo = StreamingVariableGranularityHistogram(histo.name, binspec=BinSpec.closedFromAmount(0,1,24))
    histo.add(1,3, class_name="a")
    histo.add(0,3, class_name="a")
    histo.add(1,4, class_name="a")
    histo.commit(StreamingVariableGranularityHistogram.ArgsGlobal())



if __name__ == "__main__":
    # tst_streaming()
    tst_granularity()
