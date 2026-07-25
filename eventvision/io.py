import os
import certifi
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['SSL_CERT_DIR'] = os.path.dirname(certifi.where())

import tonic

nmnist_train = tonic.datasets.NMNIST(save_to='./data', train=True)
nmnist_test = tonic.datasets.NMNIST(save_to='./data', train=False)

# print(len(train_dataset))   # 60,000
# print(len(test_dataset))    # 10,000

# events, label = nmnist_train[50000]

# checking the NMNIST dataset
'''
print(label)               # the digit this sample represents, 0-9
# print(events.dtype)        # structured array — field names, e.g. ('x','y','t','p')
# print(events.shape)        # (num_events,) — how many events this one sample has
# print(events[:10])         # the first 10 raw events

print(events['x'].min(), events['x'].max())   # expect 0 to 33 (N-MNIST is 34x34 pixels)
print(events['y'].min(), events['y'].max())   # expect 0 to 33
print(events['t'].min(), events['t'].max())   # expect roughly 0 to ~300,000 (microseconds)
print(events['p'].min(), events['p'].max())   # expect 0 and 1 (or -1 and 1, depending on convention)
'''

# plotting events
'''
on_events = events[events['p'] == 1]
off_events = events[events['p'] == 0]

plt.scatter(on_events['x'], on_events['y'], s=1, c='red', label='ON')
plt.scatter(off_events['x'], off_events['y'], s=1, c='blue', label='OFF')
plt.gca().invert_yaxis()  # image coordinates: y increases downward
plt.legend()
plt.show()
'''

# print(tonic.datasets.__all__)

tonic.datasets.DVSGesture.train_url = "https://ndownloader.figshare.com/files/38022171"
tonic.datasets.DVSGesture.test_url = "https://ndownloader.figshare.com/files/38020584"

gesture_train = tonic.datasets.DVSGesture(save_to='./data', train=True)
gesture_test = tonic.datasets.DVSGesture(save_to='./data', train=False)

# print(len(gesture_train))

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

    #force everything to the 0/1 convention confirmed on N-MNIST
    if p.min() == -1:
        p = (p + 1) // 2

    return np.stack([x, y, t, p], axis=1)

def load_sample(dataset, index):
    """
    Loads one sample from any tonic-style event dataset and returns it in
    a common format, works identically for NMNIST and DVSGesture.

    Returns:
        events: (num_events, 4) array, columns = [x, y, t, p]
        label: int
        sensor_size: (width, height)
    """
    events, label = dataset[index]
    events = standardize_events(events)
    sensor_size = dataset.sensor_size[:2]  # tonic stores (W, H, num_polarity_channels)
    return events, label, sensor_size

g_events, g_label, g_sensor_size = load_sample(gesture_train, 0)

# print("label:", g_label)
# print("num events:", g_events.shape)
# print("sensor size:", g_sensor_size)
# print("t range:", g_events[:, 2].min(), "to", g_events[:, 2].max())

# all_labels = np.array([gesture_train[i][1] for i in range(len(gesture_train))])
# print("unique labels:", np.unique(all_labels))
# print("num classes:", len(np.unique(all_labels)))

