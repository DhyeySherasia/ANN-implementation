import tensorflow as tf
import numpy as np
import os
import time


def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}"

    return unique_name


def get_callbacks(config, x_train):
    logs = config['logs']
    unique_dir_name = get_timestamp("tb_logs")
    tensorboard_root_log_dir = os.path.join(logs['logs_dir'], logs['tensorboard_root_log_dir'], unique_dir_name)

    os.makedirs(tensorboard_root_log_dir, exist_ok=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_root_log_dir)
    file_writer = tf.summary.create_file_writer(logdir=tensorboard_root_log_dir)
    with file_writer.as_default():
        images = np.reshape(x_train[10:30], (-1, 28, 28, 1))
        tf.summary.image("20 Handwritten digit samples", images, max_outputs=25, step=0)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    
    checkpt_dir = os.path.join(config['artifacts']['artifacts_dir'], config['artifacts']['checkpoint_dir'])
    os.makedirs(checkpt_dir, exist_ok=True)
    checkpt_path = os.path.join(checkpt_dir, "model_checkpt.h5")
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(checkpt_path, save_best_weights_only=True)

    return [tensorboard_cb, early_stopping_cb, checkpointing_cb]





