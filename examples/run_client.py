'''This script is a non-runnable example - you need to provide your own model class and wandb info
'''
import logging
from nn_service import NNClient


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    client = NNClient('127.0.0.1', '5555', '/tmp/plasma')
    client.register(model_name, cache_dir, run_path, checkpoint, gpu)

    y = client.query(model_name, {'x': x})['y']
