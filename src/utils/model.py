import tensorflow as tf
from tensorflow import keras


def create_model(LOSS_FUNCTON, OPTIMIZER, METRICS, NUM_CLASSES):

    layers = [
            keras.layers.Flatten(input_shape=[28,28], name="input_layer"),
            keras.layers.Dense(300, activation='relu', name="hidden_layer_1"),
            keras.layers.Dense(100, activation='relu', name="hidden_layer_2"),
            keras.layers.Dense(NUM_CLASSES, activation='softmax', name="output_layer")
    ]

    model = keras.models.Sequential(layers)
    model.summary()

    model.compile(loss=LOSS_FUNCTON, optimizer=OPTIMIZER, metrics=METRICS)


    return model  ## <<< Untrained model

