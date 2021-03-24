import pickle



def save_pickle(path, obj):
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):

    with open(path, 'rb') as f:

        return pickle.load(f)

if __name__ == '__main__':
    pass