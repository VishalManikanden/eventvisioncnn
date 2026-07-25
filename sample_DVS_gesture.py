import os
import certifi
from eventvision.datasets import make_dataset_precomputed
from eventvision.datasets import get_frame_shape
from eventvision.encoding import events_to_frame
from eventvision.io import load_sample
from eventvision.models import build_cnn, compile_cnn
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['SSL_CERT_DIR'] = os.path.dirname(certifi.where())

import tonic

tonic.datasets.DVSGesture.train_url = "https://ndownloader.figshare.com/files/38022171"
tonic.datasets.DVSGesture.test_url = "https://ndownloader.figshare.com/files/38020584"

gesture_train = tonic.datasets.DVSGesture(save_to='./data', train=True)
gesture_test = tonic.datasets.DVSGesture(save_to='./data', train=False)

gesture_train_ds = make_dataset_precomputed(gesture_train, strategy='fixed_time', batch_size=16, shuffle=True, augment_data=True)
gesture_test_ds = make_dataset_precomputed(gesture_test, strategy='fixed_time', batch_size=16, shuffle=False, augment_data=False)

gesture_frame_shape = get_frame_shape(gesture_train, strategy='fixed_time')
all_labels = np.array([gesture_train[i][1] for i in range(len(gesture_train))])
num_classes = len(np.unique(all_labels))

gesture_cnn = build_cnn(
    input_shape=gesture_frame_shape,
    num_classes=num_classes,
    extra_conv_block=True,
    dropout_rate=0.2,
    l2_regularization=0.0
)
gesture_cnn = compile_cnn(gesture_cnn)
gesture_cnn.summary()

# early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# halves the learning rate whenever val_loss stopes improving for 6 epochs
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6)

gesture_history = gesture_cnn.fit(
    x=gesture_train_ds,
    validation_data=gesture_test_ds,
    epochs=50,
    callbacks=[early_stop] # reduce_lr
)

test_loss, test_accuracy = gesture_cnn.evaluate(gesture_test_ds)
print(f"Actual final test accuracy: {test_accuracy:.4f}")

plt.plot(gesture_history.history['accuracy'], label='train accuracy')
plt.plot(gesture_history.history['val_accuracy'], label='val accuracy')
plt.axhline(y=max(gesture_history.history['val_accuracy']), color='gray', linestyle='--', label='best val_accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

os.makedirs('models', exist_ok=True)
gesture_cnn.save('models/gesture_fixed_time_cnn.keras')
