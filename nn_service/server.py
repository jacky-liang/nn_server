import logging
import torch
import wandb
from simple_zmq import SimpleZMQServer
from pathlib import Path
import pyarrow.plasma as plasma
from .pa_utils import pa_dict_to_np_dict, np_dict_to_pa_dict

logger = logging.getLogger(__name__)


class NNServer:

    def __init__(self, ip, port, plasma_path, model_classes):
        logger.info(f'Creating server on {ip}:{port}')
        self._server = SimpleZMQServer(ip, port)

        logger.info(f'Connecting to plasma on {plasma_path}')
        self._pyarrow_client = plasma.connect(plasma_path)

        self._model_classes = model_classes
        self._models = {}

    def serve(self):
        logger.info(f'Serving models: {list(self._model_classes.keys())}')
        while True:
            req = self._server.recv()
            cmd, content = req['cmd'], req['content']

            rep = {
                'success': False,
                'content': ''
            }

            if cmd == 'register':
                try:
                    model_cls_name = content['model_cls_name']
                    logger.info(f'Registering model: {content}')

                    model_cls = self._model_classes[model_cls_name]
                    self._models[model_cls_name] = load_wandb_model(
                        model_cls, 
                        content['cache_dir'],
                        content['run_path'],
                        content['checkpoint'],
                        content['gpu'],
                        model_init_kwargs=content['model_init_kwargs']
                    )
                    rep['success'] = True
                except Exception as e:
                    logger.error(e)
                    rep['content'] = str(e)
            elif cmd == 'query':
                try:
                    model_cls_name = content['model_cls_name']
                    logger.info(f'Querying model {model_cls_name}')

                    model = self._models[model_cls_name]
                    
                    inputs_np = pa_dict_to_np_dict(content['inputs'], self._pyarrow_client)
                    inputs_th = {k: torch.from_numpy(v).to(model.device) for k, v in inputs_np.items()}
                    
                    outputs_th = model(inputs_th)
                    outputs_np = {k: v.cpu().numpy() for k, v in outputs_th.items()}
                    rep['content'] = np_dict_to_pa_dict(outputs_np, self._pyarrow_client)
                    rep['success'] = True
                except Exception as e:
                    logger.error(e)
                    rep['content'] = e.message
            else:
                rep['content'] = f'Unknown cmd: {cmd}'

            logger.info('Done')
            self._server.rep(rep)


def load_wandb_model(model_cls, cache_dir, run_path, checkpoint, gpu, model_init_kwargs={}):
    root_path = Path(cache_dir) / run_path
    ckpt_file = wandb.restore(checkpoint, run_path=run_path, root=root_path)

    model = model_cls.load_from_checkpoint(ckpt_file.name, **model_init_kwargs).cuda(gpu)
    model.train(False)
    
    return model
