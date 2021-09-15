'''This script is a non-runnable example - you need to provide your own model class and wandb info
'''

from nn_service import NNClient


if __name__ == '__main__':
    client = NNClient('127.0.0.1', '5555', '/tmp/pyarrow')
    client.register(model_cls_name, cache_dir, run_path, checkpoint, gpu)

    y = client.query(model_cls_name, {'x': x})['y']
