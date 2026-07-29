import numpy as np
from eventvisioncnn.benchmark import drop_events, shrink_window


def test_drop_events_removes_expected_fraction():
    events = np.zeros((1000, 4))
    rng = np.random.default_rng(seed=0)
    result = drop_events(events, drop_fraction=0.5, rng=rng)
    assert 400 <= len(result) <= 600


def test_shrink_window_keeps_only_early_events():
    events = np.array([
        [0, 0, 0, 1],
        [0, 0, 50, 1],
        [0, 0, 100, 1],
    ], dtype=np.float64)

    result = shrink_window(events, keep_fraction=0.5)

    assert len(result) == 2
    assert result[:, 2].max() <= 50