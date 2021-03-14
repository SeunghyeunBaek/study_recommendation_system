import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import math
from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm

DATA_DIR = '/workspace/recsys/03_nural_cf/item2vec/archive/kmrd/kmr_dataset/datafile/kmrd-small'

class DatasetLoader:


    def __init__(self, data_path):
        self.train_df, val_temp_df = read_data(data_path)

        self.min_rating = min(self.train_df.rate)
        self.max_rating = self.train_df.rate.max()

        self.unique_users = self.train_df.user.unique()
        self.num_users = len(self.unique_users)
        self.user_to_index = {original: idx for idx, original in enumerate(self.unique_users)}
        # 0 1 0 0 0 ... 0

        self.unique_movies = self.train_df.movie.unique()
        self.num_movies = len(self.unique_movies)
        self.movie_to_index = {original: idx for idx, original in enumerate(self.unique_movies)}

        self.val_df = val_temp_df[val_temp_df.user.isin(self.unique_users) & val_temp_df.movie.isin(self.unique_movies)]


    def generate_trainset(self):
        # user 0, 0, 0, 1,2, 3,3, -> movie: 0,0,0,0,0,0,
        X_train = pd.DataFrame({'user': self.train_df.user.map(self.user_to_index),
                                'movie': self.train_df.movie.map(self.movie_to_index)})
        y_train = self.train_df['rate'].astype(np.float32)

        return X_train, y_train


    def generate_valset(self):
        X_val = pd.DataFrame({'user': self.val_df.user.map(self.user_to_index),
                              'movie': self.val_df.movie.map(self.movie_to_index)})
        y_val = self.val_df['rate'].astype(np.float32)
        return X_val, y_val


def read_data(dir_):
    df = pd.read_csv(os.path.join(dir_, 'rates.csv'))
    train_df, val_df = train_test_split(df, test_size=.2, shuffle=True, random_state=42)

    return train_df, val_df


