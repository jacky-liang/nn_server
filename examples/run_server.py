'''This script is a non-runnable example - you need to provide your own model class and wandb info
'''

from nn_service import NNServer


if __name__ == '__main__':
    server = NNServer('127.0.0.1', '5555', '/tmp/pyarrow', {model_cls.__name__: model_cls})
    server.serve()
