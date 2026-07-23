import numpy as np
import tensorflow as tf

from eventvision.io import load_sample, nmnist_train
from eventvision.encoding import events_to_frame


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

    # keeps a rolling buffer of shuffle_buffer samples and picks randomly from within as it goes
    # default 1000 (to ensure that epochs don't repeat in the exact same order)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    ds = ds.batch(batch_size)

    # prepares the next batch while the current batch is training
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
