import tensorflow as tf


def build_cnn(input_shape, num_classes):
    """
    CNN classifier for event-frame inputs, generalized to accept any input
    shape and any number of output classes.
    """
    model = tf.keras.models.Sequential()

    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
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
