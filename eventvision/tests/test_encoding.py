import numpy as np
from eventvision.encoding import events_to_frame
import matplotlib.pyplot as plt

from eventvision.io import load_sample, nmnist_train


def test_fixed_time_counts_repeated_pixel():
    events = np.array([
        [0, 0, 100, 1],
        [0, 0, 200, 1],
        [1, 1, 150, 0],
    ], dtype=np.float64)

    frame = events_to_frame(events, sensor_size=(2, 2), strategy='fixed_time')

    assert frame[0, 0, 1] == 2   # ON channel, pixel (0,0), fired twice
    assert frame[1, 1, 0] == 1   # OFF channel, pixel (1,1), fired once


def test_voxel_produces_correct_channel_count():
    events = np.array([
        [0, 0, 0, 1],
        [0, 0, 500, 0],
    ], dtype=np.float64)

    frame = events_to_frame(events, sensor_size=(2, 2), strategy='voxel', num_bins=5)

    assert frame.shape == (2, 2, 10)   # num_bins * 2 channels

