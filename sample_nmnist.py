"""
Example code for training a fixed time CNN on the NMNIST dataset
"""

import os
import certifi
from eventvisioncnn.datasets import make_dataset
from eventvisioncnn.datasets import get_frame_shape
from eventvisioncnn.encoding import events_to_frame
from eventvisioncnn.io import load_sample
from eventvisioncnn.models import build_cnn, compile_cnn
import matplotlib.pyplot as plt
import numpy as np

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['SSL_CERT_DIR'] = os.path.dirname(certifi.where())

import tonic

# build the train and test datasets
nmnist_train = tonic.datasets.NMNIST(save_to='./data', train=True)
nmnist_test = tonic.datasets.NMNIST(save_to='./data', train=False)
train_ds = make_dataset(nmnist_train, strategy='fixed_time', batch_size=32, shuffle=True)
test_ds = make_dataset(nmnist_test, strategy='fixed_time', batch_size=32, shuffle=False)

# build and compile the model with correct shape and class count
frame_shape = get_frame_shape(nmnist_train, strategy='fixed_time')  # (34, 34, 2)
num_classes = 10

cnn = build_cnn(input_shape=frame_shape, num_classes=num_classes)
cnn = compile_cnn(cnn)

# caching the dataset to avoid recomputing frames every epoch
train_ds = train_ds.cache()
test_ds = test_ds.cache()

# training the model
history = cnn.fit(x=train_ds, validation_data=test_ds, epochs=25)

# visualization of the training curves
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()

# evaluate on the test set
test_loss, test_accuracy = cnn.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.4f}")

# single sample prediction
events, true_label, sensor_size = load_sample(nmnist_test, 9000)
frame = events_to_frame(events, sensor_size, strategy='fixed_time')
frame_batch = np.expand_dims(frame, axis=0)   # model expects a batch dimension

prediction = cnn.predict(frame_batch)
predicted_label = np.argmax(prediction[0])

print("true label:", true_label)
print("predicted label:", predicted_label)
print("model's confidence per digit:", prediction[0])

# saving the model
os.makedirs('models', exist_ok=True)
cnn.save('models/nmnist_fixed_time_cnn.keras')