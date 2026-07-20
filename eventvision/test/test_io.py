import numpy as np
from eventvision.io import standardize_events

def test_polarity_forced_to_0_1():
    fake_events = np.array(
        [(0, 0, 100, -1), (1, 1, 200, 1)],
        dtype=[('x', '<i8'), ('y', '<i8'), ('t', '<i8'), ('p', '<i8')]
    )
    result = standardize_events(fake_events)
    assert set(result[:, 3]) <= {0, 1}
