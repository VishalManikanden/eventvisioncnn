import numpy as np
import tensorflow as tf
from eventvisioncnn.io import load_sample
from eventvisioncnn.encoding import events_to_frame


def get_frame_shape(source_dataset, strategy='fixed_time', **encoding_kwargs):
    events, label, sensor_size = load_sample(source_dataset, 0)
    frame = events_to_frame(events, sensor_size, strategy=strategy, **encoding_kwargs)
    return frame.shape


def sample_generator(source_dataset, strategy='fixed_time', shuffle_indices=False, **encoding_kwargs):
    """
    Creates (frame, label) pairs, one at a time, for every sample in a
    tonic-style event dataset, converting raw events to a dense frame while shuffling the indices
    if specified
    """

    indices = np.arange(len(source_dataset))
    if shuffle_indices:
        np.random.shuffle(indices)

    for index in indices:
        events, label, sensor_size = load_sample(source_dataset, index)
        frame = events_to_frame(events, sensor_size, strategy=strategy, **encoding_kwargs)
        yield frame, label


def make_dataset(source_dataset, strategy='fixed_time', batch_size=32,
                  shuffle=True, shuffle_buffer=1000, **encoding_kwargs):
    frame_shape = get_frame_shape(source_dataset, strategy=strategy, **encoding_kwargs)

    output_signature = (
        tf.TensorSpec(shape=frame_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: sample_generator(source_dataset, strategy=strategy,
                                   shuffle_indices=shuffle, **encoding_kwargs),
        output_signature=output_signature
    )

    # keeps a rolling buffer of shuffle_buffer samples and picks randomly from within as it goes.
    # default 1000 (to ensure that epochs don't repeat in the exact same order)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    ds = ds.batch(batch_size)

    # prepares the next batch while the current batch is training
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def normalize_frame(frame):
    """Compresses the wide dynamic range of raw event counts."""
    return np.log1p(frame)


def precompute_frames(source_dataset, strategy='fixed_time', **encoding_kwargs):
    """
    Encodes every sample in source_dataset up front into plain NumPy arrays.
    Only use this when the full encoded dataset comfortably fits in memory.
    Check the size estimate before choosing this over make_dataset().
    """
    frames = []
    labels = []
    for index in range(len(source_dataset)):
        events, label, sensor_size = load_sample(source_dataset, index)
        frame = events_to_frame(events, sensor_size, strategy=strategy, **encoding_kwargs)
        frames.append(normalize_frame(frame))
        labels.append(label)

    return np.stack(frames), np.array(labels, dtype=np.int64)


def augment(frame, label):
    max_shift = 4
    dy = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
    dx = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
    frame = tf.roll(frame, shift=[dy, dx], axis=[0, 1])
    return frame, label


def make_dataset_precomputed(source_dataset, strategy='fixed_time', batch_size=16, shuffle=True, shuffle_buffer=1000, augment_data=False,
                              **encoding_kwargs):
    frames, labels = precompute_frames(source_dataset, strategy=strategy, **encoding_kwargs)
    ds = tf.data.Dataset.from_tensor_slices((frames, labels))

    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    if augment_data:
        ds = ds.map(augment)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
