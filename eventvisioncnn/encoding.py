import numpy as np


def accumulate_counts(events, sensor_size, weights=None):
    """
    Returns a (height, width, 2) array: channel 0 = OFF accumulation,
    channel 1 = ON accumulation.

    events: (N, 4) array, columns = [x, y, t, p]
    sensor_size: (width, height)
    weights: optional (N,) array of per-event weights; defaults to 1 per event
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


def events_to_frame_fixed_time(events, sensor_size, t_start=None, t_end=None):
    """
    Accumulates all events within a fixed time window into one frame.

    t_start, t_end: window bounds in the same units as the events' timestamps,
    defaults to the full time range of the given events if not specified.
    """

    # picking the lowest/highest times if no window is specified
    if t_start is None:
        t_start = events[:, 2].min()
    if t_end is None:
        t_end = events[:, 2].max()

    mask = (events[:, 2] >= t_start) & (events[:, 2] <= t_end)
    return accumulate_counts(events[mask], sensor_size)


def events_to_frame_fixed_count(events, sensor_size, start_idx=0, count=500):
    """
    Accumulates a fixed number of consecutive events into one frame.

    Assumes events are already sorted by timestamp, so a plain index
    slice is equal to slicing by time-order. Window duration self-adjusts
    to motion speed, since faster motion produces the same event count in less time.
    """
    window = events[start_idx:start_idx + count]
    return accumulate_counts(window, sensor_size)


def events_to_frame_decay(events, sensor_size, tau=5000.0, t_ref=None):
    """
    Accumulates events with an exponentially decaying weight based on
    how long ago each event occurred relative to t_ref.

    tau: decay time constant (same units as timestamps): smaller tau
    means older events fade out faster.
    t_ref: reference "now" moment, defaults to the latest timestamp
    in the given events.
    """
    if t_ref is None:
        t_ref = events[:, 2].max()

    ts = events[:, 2].astype(np.float64)
    weights = np.exp(-(t_ref - ts) / tau)
    return accumulate_counts(events, sensor_size, weights=weights)


def events_to_frame_voxel(events, sensor_size, num_bins=5):
    """
    Splits the event time range into num_bins equal intervals and
    accumulates each into its own ON/OFF channel pair, producing a
    (height, width, num_bins * 2) tensor — analogous to stacking
    multiple time slices the way RGB stacks color channels.
    """
    width, height = sensor_size
    t_min, t_max = events[:, 2].min(), events[:, 2].max()
    bin_edges = np.linspace(t_min, t_max, num_bins + 1)

    frame = np.zeros((height, width, num_bins * 2), dtype=np.float32)

    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b < num_bins - 1:
            mask = (events[:, 2] >= lo) & (events[:, 2] < hi)
        else:
            mask = (events[:, 2] >= lo) & (events[:, 2] <= hi)

        channel_slice = frame[:, :, b * 2:(b + 1) * 2]
        counts = accumulate_counts(events[mask], sensor_size)
        channel_slice += counts

    return frame


def events_to_frame(events, sensor_size, strategy='fixed_time', **kwargs):
    """
    Dispatches to one of the accumulation strategies above by name.

    strategy: one of 'fixed_time', 'fixed_count', 'decay', 'voxel'
    **kwargs: forwarded to the selected strategy function
      (e.g. tau=3000 for 'decay', num_bins=8 for 'voxel')
    """
    strategies = {
        'fixed_time': events_to_frame_fixed_time,
        'fixed_count': events_to_frame_fixed_count,
        'decay': events_to_frame_decay,
        'voxel': events_to_frame_voxel,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy '{strategy}', choose from {list(strategies)}")

    return strategies[strategy](events, sensor_size, **kwargs)
