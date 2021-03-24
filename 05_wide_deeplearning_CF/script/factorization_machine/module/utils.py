import pickle
import yaml
import os

def load_pickle(path):
    
    with open(path, 'rb') as f:

        return pickle.load(f)


def save_pickle(path, obj_):

    with open(path, 'wb') as f:

        pickle.dump(obj_, f, pickle.HIGHEST_PROTOCOL)


def load_yaml(path):

    with open(path, 'r') as f:

        return yaml.load(f, Loader=yaml.FullLoader)

def create_directory(dir_):

    if not os.path.isdir(dir_):
        os.makedirs(dir_)

        
