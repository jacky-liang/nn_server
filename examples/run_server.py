'''This script is a non-runnable example - you need to provide your own model class and wandb info
'''
import logging
from nn_service import NNServer


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    server = NNServer('127.0.0.1', '5555', '/tmp/plasma', {model_name: model_class})
    server.serve()
