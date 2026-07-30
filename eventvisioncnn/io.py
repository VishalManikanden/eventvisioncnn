import numpy as np


def standardize_events(events):
    """
    Converts a tonic structured event array of fields [x, y, t, p] into a
    plain [num_events, 4] array with a guaranteed dtype and polarity
    convention, regardless of which dataset it came from.
    """
    x = events['x'].astype(np.int32)
    y = events['y'].astype(np.int32)
    t = events['t'].astype(np.int64)
    p = events['p'].astype(np.int8)

    # force everything to the 0/1 convention
    if p.min() == -1:
        p = (p + 1) // 2

    return np.stack([x, y, t, p], axis=1)

def load_sample(dataset, index):
    """
    Loads one sample from any tonic-style event dataset and returns it in
    a common format, works identically for NMNIST and DVS128 Gesture.

    Returns:
    events: (num_events, 4) array, columns = [x, y, t, p]
    label: actual number
    sensor_size: (width, height)
    """
    events, label = dataset[index]
    events = standardize_events(events)
    sensor_size = dataset.sensor_size[:2]  # tonic stores (W, H, num_polarity_channels)
    return events, label, sensor_size

