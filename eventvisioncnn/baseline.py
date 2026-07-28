from eventvisioncnn.encoding import accumulate_counts


def events_to_conventional_frame(events, sensor_size, t_start=None, t_end=None):
    """
    Simulates what a single conventional-camera exposure would capture
    given the same physical motion the event camera recorded.

    Reuses the exact same accumulation math as events_to_frame_fixed_time,
    but collapses the on/off channels into one since a standard intensity
    sensor has no way to distinguish "got brighter" from "got darker" the
    way each DVS pixel's own circuit does, and it only reports total light.
    Defaults to the full recording as one window, mirroring a low-frame-rate
    camera trying to capture the whole event in a single shot.
    """
    if t_start is None:
        t_start = events[:, 2].min()
    if t_end is None:
        t_end = events[:, 2].max()

    mask = (events[:, 2] >= t_start) & (events[:, 2] <= t_end)
    frame = accumulate_counts(events[mask], sensor_size)   # (H, W, 2)
    return frame.sum(axis=2, keepdims=True)                  # (H, W, 1)