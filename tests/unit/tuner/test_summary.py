import os.path

from finetuner.tuner.summary import ScalarSequence, Summary
import numpy as np


def test_summary(tmpdir):
    s1 = ScalarSequence('s1')
    s2 = ScalarSequence('s2')
    s1 += np.random.random((10,))
    assert s1.floats()
    s2 += np.random.random((100,))
    assert s1.floats()

    s3 = ScalarSequence('empty')

    sm = Summary(s1, s2, s3)
    assert len(sm.dict()) == 2  #: empty record is not counted
    sm.plot()
    sm.plot(max_plot_points=5)
    sm.plot(max_plot_points=1000)
    sm.plot(tmpdir / 'sm.png')
    assert os.path.exists(tmpdir / 'sm.png')
