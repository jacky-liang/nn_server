def pa_dict_to_np_dict(pa_dict, client):
    vals = client.get([object_id for object_id in pa_dict.values()])
    return {k : v for k, v in zip(pa_dict.keys(), vals)}


def np_dict_to_pa_dict(pa_dict, client):
    ids = [client.put(v) for v in pa_dict.values()]
    return {k : v for k, v in zip(pa_dict.keys(), ids)}
