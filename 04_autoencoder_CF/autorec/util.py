import logging
import pickle

def set_logger(name: str, file_path: str)-> logging.RootLogger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(message)s')

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def save_pickle(path, obj):
    
    with open(path, 'wb') as f:
        
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):

    with open(path, 'rb') as f:

        return pickle.load(f)


