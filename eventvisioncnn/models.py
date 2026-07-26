import tensorflow as tf
from tensorflow.keras import regularizers


def build_cnn(input_shape, num_classes, extra_conv_block=False, dropout_rate=0.0, l2_regularization=0.0):
    """
    CNN classifier for event-frame inputs, generalized to accept any input
    shape and any number of output classes, and to work with both NMNIST and DVS Gesture

    dropout_rate: regularization technique, randomly turns off dropout_rate fraction of
    neurons in the layer that follows to force the network not to over-rely on a specific
    neuron's memorized pattern

    l2_regularization: penalizes large weight values during training to discourage the network
    from fitting to patterns specific to just the training dataset
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                                     kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                                     kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    if extra_conv_block:
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',
                                         kernel_regularizer=regularizers.l2(l2_regularization)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu',
                                    kernel_regularizer=regularizers.l2(l2_regularization)))

    if dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))
    return model


def compile_cnn(model, learning_rate=0.001):
    """
    Compiles the CNN model with the given learning rate for fine tuning
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
