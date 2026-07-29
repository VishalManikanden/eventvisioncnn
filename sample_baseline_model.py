import os
from functools import partial
import certifi
import numpy as np
import matplotlib.pyplot as plt
from eventvisioncnn.baseline import events_to_conventional_frame
from eventvisioncnn.datasets import make_dataset_precomputed
from eventvisioncnn.models import build_cnn, compile_cnn
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['SSL_CERT_DIR'] = os.path.dirname(certifi.where())

import tonic

tonic.datasets.DVSGesture.train_url = "https://ndownloader.figshare.com/files/38022171"
tonic.datasets.DVSGesture.test_url = "https://ndownloader.figshare.com/files/38020584"

gesture_train = tonic.datasets.DVSGesture(save_to='./data', train=True)
gesture_test = tonic.datasets.DVSGesture(save_to='./data', train=False)

baseline_encode_function = partial(events_to_conventional_frame)

gesture_train_ds_baseline = make_dataset_precomputed(gesture_train, baseline_encode_function, batch_size=16, shuffle=True,
                                                     augment_data=True)
gesture_test_ds_baseline = make_dataset_precomputed(gesture_test, baseline_encode_function, batch_size=16, shuffle=False,
                                                    augment_data=False)

baseline_frame_shape = gesture_train_ds_baseline.element_spec[0].shape[1:]

all_labels = np.array([gesture_train[i][1] for i in range(len(gesture_train))])
num_classes = len(np.unique(all_labels))

gesture_cnn_baseline = build_cnn(input_shape=baseline_frame_shape, num_classes=num_classes, extra_conv_block=True,
                                 dropout_rate=0.4, l2_regularization=0.0)
gesture_cnn_baseline = compile_cnn(gesture_cnn_baseline)

early_stop_baseline = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr_baseline = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

baseline_history = gesture_cnn_baseline.fit(x=gesture_train_ds_baseline, validation_data=gesture_test_ds_baseline,
                                            epochs=50, callbacks=[early_stop_baseline])

test_loss_baseline, test_accuracy_baseline = gesture_cnn_baseline.evaluate(gesture_test_ds_baseline)

print(f"conventional (frame) baseline: {test_accuracy_baseline:.4f}")

plt.plot(baseline_history.history['accuracy'], label='train accuracy (baseline)')
plt.plot(baseline_history.history['val_accuracy'], label='val accuracy (baseline)')
plt.axhline(y=max(baseline_history.history['val_accuracy']), color='gray', linestyle='--', label='best val')
plt.xlabel('epoch')
plt.legend()
plt.show()

os.makedirs('models', exist_ok=True)
gesture_cnn_baseline.save('models/gesture_baseline_dropout04.keras')
