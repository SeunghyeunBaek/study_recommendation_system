from dataloader import * 
from architecture import SimpleAutoEncoder
from util import *
from train import weights_init, MSEloss
import os

import torch.optim as optim


DATALOADER_DIR = '/workspace/study_recsys/04_autoencoder/autorec/data/'
LEARNING_RATE = 1e-3

if __name__ == '__main__':

    #* Load data1
    train_dataloader = load_pickle(os.path.join(DATALOADER_DIR, 'dataloader_train.pickle'))
    test_dataloader = load_pickle(os.path.join(DATALOADER_DIR, 'dataloader_val.pickle'))

    #* Train
    n_user, n_movie= len(user_to_index.keys()), len(movie_to_index.keys())
    simple_encoder_model = SimpleAutoEncoder(num_inputs=n_user, num_hiddens=100, kind='selu')
    optimizer = optim.Adam(simple_encoder_model.parameters(), lr=LEARNING_RATE)

    simple_encoder_model.apply(weights_init)
    simple_encoder_model.train()
    train_loss = 0
    
    for idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        pred = simple_encoder_model(batch)
        loss, num_rating = MSEloss(pred, batch)
        loss = torch.sqrt(loss/num_rating)
        train_loss += loss.item()
        optimizer.step()
        print(train_loss/(idx+1))

    simple_encoder_model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            pred = simple_encoder_model(batch)
            loss, num_ratings = MSEloss(pred, batch)
            loss = torch.sqrt(loss / num_ratings)
            val_loss += loss.item()

            print(val_loss/(idx+1))
