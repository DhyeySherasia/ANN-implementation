import tensorflow as tf
from tensorflow import keras
import time
import os


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

    return model  # <<< Untrained model


def get_unique_filename(model_name):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{model_name}")
    return unique_filename


def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)

