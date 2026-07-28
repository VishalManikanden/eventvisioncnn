import numpy as np
from eventvisioncnn.baseline import events_to_conventional_frame


def test_conventional_frame_discards_polarity():
    events = np.array([
        [0, 0, 100, 1],  # ON
        [0, 0, 200, 0],  # OFF, same pixel
        [1, 1, 150, 1],  # ON
    ], dtype=np.float64)

    frame = events_to_conventional_frame(events, sensor_size=(2, 2))

    assert frame.shape == (2, 2, 1)
    assert frame[0, 0, 0] == 2   # both events at (0,0) counted, regardless of polarity
    assert frame[1, 1, 0] == 1