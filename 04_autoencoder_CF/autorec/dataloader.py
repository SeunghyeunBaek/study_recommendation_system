import torch
from torch.utils.data import Dataset, DataLoader

import os
import sys
import numpy as np 
import pandas as pd
from util import set_logger, save_pickle
from sklearn.model_selection import train_test_split

PRJ_DIR = ''
DATA_DIR = '/workspace/study_recsys/data/kmrd/kmr_dataset/datafile/kmrd-small'
LOGGER_PATH = '/workspace/study_recsys/04_autoencoder/autorec/log/dataloader.log'
DATA_LOADER_DIR = '/workspace/study_recsys/04_autoencoder/autorec/data/'



class KMRDataset(Dataset):
    

    def __init__(self, df, user_to_index, movie_to_index, item_based=True):
        self.min_rating = min(df['rate'])
        self.max_rating = max(df['rate'])

        self.rating = df['rate'].values
        self.user = [user_to_index[user] for user in df['user'].values]
        self.movie = [movie_to_index[movie] for movie in df['movie'].values]
        
        output_tensor = torch.FloatTensor(self.rating)

        if item_based:
            input_tensor = torch.LongTensor([self.movie, self.user])
            size = torch.Size([len(movie_to_index), len(user_to_index)])
            self.data = torch.sparse.FloatTensor(input_tensor,
                                                 output_tensor,
                                                 size).to_dense()

        else:
            input_tensor = torch.LongTensor([self.user, self.movie])
            size = torch.Size([len(user_to_index ), len(movie_to_index)])
            self.data = torch.sparse.FloatTensor(input_tensor,
                                                 output_tensor,
                                                 size).to_dense()
          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def read_data(path, n_sample=10000):
    df = pd.read_csv(os.path.join(path, 'rates.csv'))[:n_sample]
    train_df, val_df = train_test_split(df, test_size=.2, random_state=42)
    user_to_index = {user: id for id, user in enumerate(df['user'].unique())}
    movie_to_index = {movie: id for id, movie in enumerate(df['movie'].unique())}
    
    return train_df, val_df, user_to_index, movie_to_index


train_df, val_df, user_to_index, movie_to_index = read_data(DATA_DIR)

if __name__ == '__main__':
    
    logger = set_logger(name='dataloader', file_path=LOGGER_PATH)
    train_df, val_df, user_to_index, movie_to_index = read_data(DATA_DIR)
    train_dataset = KMRDataset(train_df, user_to_index, movie_to_index)
    val_dateset = KMRDataset(val_df, user_to_index, movie_to_index)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dateset, batch_size=16, shuffle=True)

    logger.info(f"Load data from {DATA_DIR}")
    logger.info(f"Train_df: {train_df.shape}, {train_dataset.data[0].size()}")
    logger.info(f"Validaetion df: {val_df.shape}, {val_dateset.data[0].size()}")

    save_pickle(os.path.join(DATA_LOADER_DIR, 'dataloader_train.pickle'), train_dataloader)
    save_pickle(os.path.join(DATA_LOADER_DIR, 'dataloader_val.pickle'), val_dataloader)


    