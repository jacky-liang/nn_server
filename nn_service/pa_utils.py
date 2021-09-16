from time import time
import numpy as np
import pyarrow.plasma as plasma


def pa_dict_to_np_dict(pa_dict, client):
    vals = client.get([object_id for object_id in pa_dict.values()])
    return {k : v for k, v in zip(pa_dict.keys(), vals)}


def np_dict_to_pa_dict(pa_dict, client):
    ids = [client.put(v, object_id=random_object_id()) for v in pa_dict.values()]
    return {k : v for k, v in zip(pa_dict.keys(), ids)}


def random_object_id():
    time_bytes = bytes(str(time()), 'utf8')
    return plasma.ObjectID(time_bytes + np.random.bytes(20 - len(time_bytes)))