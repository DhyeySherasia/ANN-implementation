# ../src will move one step up and then search for src
import sys
sys.path.insert(0, '../src')
from utils.common import read_config
from utils.data_mgnt import get_data

import argparse


def training(config_path):
    config = read_config(config_path)
    validation_data_size = config["params"]["validation_data_size"]
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_data(validation_data_size)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--config', '-c', default='config.yaml')

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)