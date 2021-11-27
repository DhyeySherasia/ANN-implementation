# ../src will move one step up and then search for src
import sys
sys.path.insert(0, '../src')
from utils.common import read_config
from utils.data_mgnt import get_data
from utils.model import create_model

import argparse


def training(config_path):
    config = read_config(config_path)
    validation_data_size = config["params"]["validation_data_size"]
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_data(validation_data_size)

    LOSS_FUNCTON = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]
    model = create_model(LOSS_FUNCTON, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config["params"]["epochs"]
    VALIDATION = (x_valid, y_valid)

    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--config', '-c', default='config.yaml')

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)