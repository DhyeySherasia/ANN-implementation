import tensorflow as tf

def get_data(validation_data_size):
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_valid, x_train = x_train_full[:validation_data_size] / 255., x_train_full[validation_data_size:] / 255.
    y_valid, y_train = y_train_full[:validation_data_size], y_train_full[validation_data_size:]

    x_test = x_test / 255.

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)