import logging
from simple_zmq import SimpleZMQClient
import pyarrow.plasma as plasma

from .pa_utils import np_dict_to_pa_dict, pa_dict_to_np_dict

logger = logging.getLogger(__name__)


class NNClient:

    def __init__(self, ip, port, pyarrow_path):
        logger.info(f'Creating client on {ip}:{port}')
        self._client = SimpleZMQClient(ip, port)

        logger.info(f'Connecting to plasma on {pyarrow_path}')
        self._pyarrow_client = plasma.connect(pyarrow_path)

    def register(self, model_cls_name, cache_dir, run_path, checkpoint, gpu, model_init_kwargs={}):
        req = {
            'cmd': 'register',
            'content': {
                'model_cls_name': model_cls_name,
                'cache_dir': cache_dir,
                'run_path': run_path,
                'checkpoint': checkpoint,
                'gpu': gpu,
                'model_init_kwargs': model_init_kwargs
            }
        }
        rep = self._client.send(req)

        if rep['success']:
            return rep['content']
        raise Exception(rep['content'])

    def query(self, model_cls_name, inputs_np):
        inputs_pa = np_dict_to_pa_dict(inputs_np, self._pyarrow_client)
        req = {
            'cmd': 'query',
            'content': {
                'model_cls_name': model_cls_name,
                'inputs': inputs_pa
            }
        }
        rep = self._client.send(req)
        
        if rep['success']:
            outputs_pa = rep['content']
            outputs_np = pa_dict_to_np_dict(outputs_pa, self._pyarrow_client)
            return outputs_np
        raise Exception(rep['content'])