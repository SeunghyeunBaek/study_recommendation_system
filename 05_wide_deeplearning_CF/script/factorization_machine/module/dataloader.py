from module.utils import load_pickle
import tensorflow as tf
import os

from sklearn.preprocessing import MinMaxScaler

class Dataset():

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.scaler = None
        self.load_data()

    def load_data(self):
        self.x = load_pickle(os.path.join(self.data_dir, 'x.pickle'))
        self.y = load_pickle(os.path.join(self.data_dir, 'y.pickle'))

    def set_scaler(self, scaler):
        self.scaler = scaler


