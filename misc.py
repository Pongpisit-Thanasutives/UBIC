import h5py
import numpy as np
import yaml
from pathlib import Path
import pickle

def h5file(file_path, cdict={}, mode='r', return_dict=False):
    content = h5py.File(file_path, mode)
    if mode == 'r': print(list(content.keys()))
    elif mode == 'w': 
        if len(cdict) == 0: 
            print("There is nothing to write...")
            return
        for k in cdict:
            content.create_dataset(k, data=cdict[k])
    out = tuple([np.array(content.get(k)) for k in content.keys()])
    if return_dict:
        out = dict(zip(content.keys(), out))
    content.close()
    return out

def yaml_load(file_path):
    if Path(file_path).exists():
        with open(file_path, 'r') as stream_file:
            # contents = yaml.safe_load(stream_file)
            contents = yaml.load(stream_file, Loader=yaml.Loader)
        stream_file.close()
        print("Successfully loaded...")
        return contents

def yaml_save(file_path, cdict):
    if Path(file_path).exists():
        with open(file_path, 'w') as stream_file:
            yaml.dump(cdict, stream_file, default_flow_style=False)
        stream_file.close()
        print("Successfully wrote...")

def pickle_save(file_path, obj):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    f.close()
    print("pickle_save done")

def pickle_load(file_path):
    if Path(file_path).exists():
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
    print("pickle_load done")
    return obj
