import numpy as np

def _accumulate_counts(events, sensor_size, weights=None):
    """
    events: (N, 4) array, columns = [x, y, t, p]
    sensor_size: (width, height)
    weights: optional (N,) array of per-event weights; defaults to 1 per event

    Returns a (height, width, 2) array: channel 0 = OFF accumulation,
    channel 1 = ON accumulation.
    """
    width, height = sensor_size
    frame = np.zeros((height, width, 2), dtype=np.float32)

    if len(events) == 0:
        return frame

    xs = events[:, 0].astype(np.int64)
    ys = events[:, 1].astype(np.int64)
    ps = events[:, 3].astype(np.int64)

    if weights is None:
        weights = np.ones(len(events), dtype=np.float32)

    np.add.at(frame, (ys, xs, ps), weights)
    return frame