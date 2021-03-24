
from module.utils import load_yaml, save_pickle
from module.utils import load_pickle, create_directory
from module.plotter import plot_lines
from module.trainer import BatchTrainer
from module.dataloader import Dataset
from module.architecture import FM

from sklearn.model_selection import train_test_split
from datetime import datetime
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random
import os

conf_path = '/workspace/study_recsys/05_widedeepfm/FM/config.yml'
conf = load_yaml(conf_path)

PROCESSED_DATA_DIR = '/workspace/study_recsys/data/breast_cancer/processed/'

BATCH_SIZE = conf['TRAIN']['batch_size']
SEED = conf['TRAIN']['seed']
EPOCH = conf['TRAIN']['epoch'] 

NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
TRAIN_SERIAL = 'train' + '_' + NOW
RESULT_DIR = os.path.join(conf['DIRECTORY']['result'], TRAIN_SERIAL)

if __name__ == '__main__':

    # Set result directory
    create_directory(RESULT_DIR)

    # Set seed
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    tf.random.set_seed(SEED)

    # Load data
    x = load_pickle(os.path.join(PROCESSED_DATA_DIR, 'x.pickle'))
    y = load_pickle(os.path.join(PROCESSED_DATA_DIR, 'y.pickle'))
    scaler = load_pickle(os.path.join(PROCESSED_DATA_DIR, 'scaler.pickle'))

    x_train, x_test,  y_train, y_test = train_test_split(x, y, test_size=.2)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train, tf.float32), tf.cast(y_train, tf.float32))).shuffle(SEED).batch(BATCH_SIZE, drop_remainder=True)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_test, tf.float32), tf.cast(y_test, tf.float32))).shuffle(SEED).batch(BATCH_SIZE,  drop_remainder=True)

    # Load model 
    model = FM(n_feature=30, n_latent_feature=10)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
    loss_function = tf.keras.losses.binary_crossentropy

    # Initiate history
    train_loss_history = list()
    train_score_history = list()
    val_loss_history = list()
    val_score_history = list()

    # Train/Validate
    trainer = BatchTrainer(model, optimizer, metric, loss_function)
    best_loss = np.Inf

    for epoch_index in tqdm(range(EPOCH)):
        sum_loss, sum_score = 0, 0

        # Train
        for x, y in train_ds:

            loss, score = trainer.train(x, y)
            sum_loss += loss
            sum_score += score
        
        train_loss_history.append(sum_loss/len(train_ds))
        train_score_history.append(sum_score/len(train_ds))
        
        sum_loss, sum_score = 0, 0
        
        # Validation
        for x, y in test_ds:
            loss, score = trainer.validate(x, y)
            sum_loss += loss
            sum_score += score

        val_loss_history.append(sum_loss/len(test_ds))
        val_score_history.append(sum_score/len(test_ds))

        if best_loss > sum_loss/len(test_ds):
            
            model.save_weights(os.path.join(RESULT_DIR, 'model.h5'))
            best_loss = sum_loss/len(test_ds)


    # Save results
    result_dict = {
        'train_loss': train_loss_history,
        'train_score': train_score_history,
        'val_loss': val_loss_history,
        'val_score': val_score_history,
        'best_loss': best_loss,
        'config': conf
    }

    loss_plot = plot_lines(list(range(EPOCH)), train_loss_history, val_loss_history)
    score_plot = plot_lines(list(range(EPOCH)), train_score_history, val_score_history)

    loss_plot.savefig(os.path.join(RESULT_DIR, 'loss.jpg'))
    score_plot.savefig(os.path.join(RESULT_DIR, 'score.jpg'))
    save_pickle(os.path.join(RESULT_DIR, 'result.pickle'), result_dict)