"""
Typical usage:

    from functools import partial
    from eventvisioncnn.encoding import events_to_frame
    from eventvisioncnn.baseline import events_to_conventional_frame
    from eventvisioncnn.benchmark import (
        load_registry_models, build_test_datasets, summarize_accuracy,
        summarize_encoding_speed, run_robustness_sweep, plot_robustness
    )

    registry = {
        'baseline': dict(model_path='models/baseline.keras',
                          encode_function=partial(events_to_conventional_frame)),
        'fixed_time': dict(model_path='models/fixed_time.keras',
                            encode_function=partial(events_to_frame, strategy='fixed_time')),
    }

    models = load_registry_models(registry)
    test_datasets = build_test_datasets(registry, my_test_dataset)
    table = summarize_accuracy(models, test_datasets)
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import time
from eventvisioncnn.io import load_sample
from eventvisioncnn.datasets import make_dataset_precomputed, normalize_frame

def load_registry_models(registry):
    """
    Loads every model referenced in a registry dict.

    registry: dict mapping name: {'model_path': ..., 'encode_function': ...}
    Returns: dict mapping the same names to loaded keras model
    """
    return {name: tf.keras.models.load_model(cfg['model_path'])
            for name, cfg in registry.items()}

def build_test_datasets(registry, source_dataset, batch_size=16):
    """
    Builds one un-shuffled, un-augmented tf.data test set per registry
    entry, each encoded with that entry's own encode_function, to guarantee
    every model is evaluated on input encoded the same way it was trained on.
    """
    return {
        name: make_dataset_precomputed(
            source_dataset, cfg['encode_function'],
            batch_size=batch_size, shuffle=False, augment_data=False
        )
        for name, cfg in registry.items()
    }


def summarize_accuracy(models, test_datasets):
    """
    Evaluates each model on its matching test dataset.

    models, test_datasets: dicts sharing the same keys (such as from
    load_registry_models and build_test_datasets).
    Returns a list of dicts, one row per model, suitable for a table or CSV.
    """
    rows = []
    for name, model in models.items():
        loss, acc = model.evaluate(test_datasets[name], verbose=0)
        rows.append({
            'encoding': name,
            'test_accuracy': acc,
            'test_loss': loss,
            'params': model.count_params(),
        })
    return rows

# models = load_registry_models()
#
# accuracy_table = summarize_accuracy(models, test_datasets)
# for row in accuracy_table:
#     print(row)


def benchmark_encoding_speed(source_dataset, encode_function, num_samples=50):
    """Average wall-clock time (seconds) to encode one sample with encode_function."""

    n = min(num_samples, len(source_dataset))
    start = time.perf_counter()
    for index in range(n):
        events, label, sensor_size = load_sample(source_dataset, index)
        _ = encode_function(events, sensor_size)
    elapsed = time.perf_counter() - start
    return elapsed / n

def summarize_encoding_speed(registry, source_dataset, num_samples=50):
    """Runs benchmark_encoding_speed() for every entry in a registry."""
    return {
        name: benchmark_encoding_speed(source_dataset, cfg['encode_function'], num_samples=num_samples)
        for name, cfg in registry.items()
    }

def drop_events(events, drop_fraction, rng=None):
    """
    Randomly removes a fraction of events — simulating a dimmer scene,
    since fewer photons crossing each pixel's brightness-change threshold
    means fewer events fire in low light, even for identical motion.
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(events) == 0 or drop_fraction <= 0:
        return events
    keep_mask = rng.random(len(events)) >= drop_fraction
    return events[keep_mask]

def shrink_window(events, keep_fraction):
    """
    Keeps only the first keep_fraction of a recording's time span to
    simulate the same physical motion compressed into less time than
    the camera actually had to observe it
    """
    if len(events) == 0 or keep_fraction >= 1.0:
        return events
    t_min, t_max = events[:, 2].min(), events[:, 2].max()
    cutoff = t_min + keep_fraction * (t_max - t_min)
    return events[events[:, 2] <= cutoff]

def evaluate_under_degradation(model, source_dataset, encode_function, degrade_function, severities):
    """
    Re-encodes every sample in source_dataset at each severity level and
    reports accuracy, measures how gracefully a model degrades, not just
    its accuracy on clean data.

    degrade_function: a callable taking (events, severity) -> degraded events,
    like drop_events or shrink_window.
    severities: values to pass to degrade_function in turn
    Returns a dict mapping severity to accuracy
    """
    results = {}
    for severity in severities:
        frames, labels = [], []
        for index in range(len(source_dataset)):
            events, label, sensor_size = load_sample(source_dataset, index)
            degraded = degrade_function(events, severity)
            frame = encode_function(degraded, sensor_size)
            frames.append(normalize_frame(frame))
            labels.append(label)

        frames = np.stack(frames)
        labels = np.array(labels, dtype=np.int64)

        predictions = model.predict(frames, verbose=0)
        predicted = np.argmax(predictions, axis=1)
        results[severity] = float(np.mean(predicted == labels))

    return results

def run_robustness_sweep(models, registry, source_dataset, dropout_severities=(0.0, 0.25, 0.5, 0.75),
                         window_severities=(1.0, 0.75, 0.5, 0.25)):
    """
    Runs both degradation sweeps (simulated low light via drop_events,
    simulated fast motion via shrink_window) for every model in the registry.

    Returns (low_light_results, fast_motion_results), each a dict mapping
    model name to {severity: accuracy}.
    """
    low_light, fast_motion = {}, {}
    for name, model in models.items():
        encode_function = registry[name]['encode_function']
        low_light[name] = evaluate_under_degradation(
            model, source_dataset, encode_function, drop_events, list(dropout_severities)
        )
        fast_motion[name] = evaluate_under_degradation(
            model, source_dataset, encode_function, shrink_window, list(window_severities)
        )
    return low_light, fast_motion

def plot_robustness(low_light_results, fast_motion_results, dropout_severities=(0.0, 0.25, 0.5, 0.75),
                    window_severities=(1.0, 0.75, 0.5, 0.25)):
    """
    Plots both robustness sweeps side by side. Pass the same severities used in run_robustness_sweep so
    the x-axes match the actual data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, per_severity in low_light_results.items():
        accs = [per_severity[s] for s in dropout_severities]
        axes[0].plot(dropout_severities, accs, marker='o', label=name)
    axes[0].set_xlabel('fraction of events dropped')
    axes[0].set_ylabel('accuracy')
    axes[0].set_title('Accuracy under simulated low light')
    axes[0].legend()

    for name, per_severity in fast_motion_results.items():
        accs = [per_severity[s] for s in window_severities]
        axes[1].plot(window_severities, accs, marker='o', label=name)
    axes[1].set_xlabel('fraction of recording kept')
    axes[1].set_ylabel('accuracy')
    axes[1].set_title('Accuracy under simulated fast motion')
    axes[1].invert_xaxis()
    axes[1].legend()

    plt.tight_layout()
    plt.show()
