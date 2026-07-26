import os
import certifi
from eventvisioncnn.datasets import make_dataset_precomputed
from eventvisioncnn.datasets import get_frame_shape
from eventvisioncnn.models import build_cnn, compile_cnn
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

num_bins = 8

gesture_train_ds_voxel = make_dataset_precomputed(gesture_train, strategy='voxel', num_bins=num_bins, batch_size=16,
                                                  shuffle=True, augment_data=True
)
gesture_test_ds_voxel = make_dataset_precomputed(gesture_test, strategy='voxel', num_bins=num_bins, batch_size=16,
                                                 shuffle=False, augment_data=False
)

voxel_frame_shape = gesture_train_ds_voxel.element_spec[0].shape[1:]

all_labels = np.array([gesture_train[i][1] for i in range(len(gesture_train))])
num_classes = len(np.unique(all_labels))
gesture_cnn_voxel = build_cnn(input_shape=voxel_frame_shape, num_classes=num_classes, extra_conv_block=True,
    dropout_rate=0.2, l2_regularization=0.0)

gesture_cnn_voxel = compile_cnn(gesture_cnn_voxel)

early_stop_voxel = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr_voxel = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

gesture_history_voxel = gesture_cnn_voxel.fit(x=gesture_train_ds_voxel, validation_data=gesture_test_ds_voxel,
                                              epochs=50, callbacks=[early_stop_voxel, reduce_lr_voxel])

test_loss_voxel, test_accuracy_voxel = gesture_cnn_voxel.evaluate(gesture_test_ds_voxel)
print(f"voxel test accuracy: {test_accuracy_voxel:.4f}")
# print(f"fixed_time test accuracy (previous best): {test_accuracy:.4f}")

plt.plot(gesture_history_voxel.history['accuracy'], label='train accuracy (voxel)')
plt.plot(gesture_history_voxel.history['val_accuracy'], label='val accuracy (voxel)')
plt.axhline(y=max(gesture_history_voxel.history['val_accuracy']), color='gray', linestyle='--', label='best val')
plt.xlabel('epoch')
plt.legend()
plt.show()

os.makedirs('models', exist_ok=True)
gesture_cnn_voxel.save('models/gesture_voxel_cnn_8bins.keras')