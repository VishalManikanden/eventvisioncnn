import numpy as np
from eventvisioncnn.datasets import make_dataset


class FakeDataset:
    """Used only for testing."""

    def __init__(self, n=8):
        self.n = n
        self.sensor_size = (4, 4, 2)  # tonic convention: (width, height, num_polarities)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        events = np.array(
            [(0, 0, index, 1), (1, 1, index + 1, 0)],
            dtype=[('x', '<i8'), ('y', '<i8'), ('t', '<i8'), ('p', '<i8')]
        )
        label = index % 2
        return events, label


def test_make_dataset_batch_shapes():
    fake = FakeDataset(n=8)
    ds = make_dataset(fake, strategy='fixed_time', batch_size=4, shuffle=False)

    frames, labels = next(iter(ds))

    assert frames.shape == (4, 4, 4, 2)
    assert labels.shape == (4,)