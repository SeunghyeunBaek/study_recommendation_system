"""데이터 전처리

"""

import os, sys
sys.path.append('/workspace/study_recsys/05_widedeepfm/FM/')

from sklearn.preprocessing import MinMaxScaler
from module.dataloader import Dataset
from module.utils import load_yaml, save_pickle

conf_path = '/workspace/study_recsys/05_widedeepfm/FM/config.yml'
conf = load_yaml(conf_path)

ORIGINAL_DATA_DIR = os.path.join(conf['DIRECTORY']['original_data'])
PROCESSED_DATA_DIR = os.path.join(conf['DIRECTORY']['processed_data'])

if __name__ == '__main__':

    dataset = Dataset(ORIGINAL_DATA_DIR)
    x, y = dataset.x, dataset.y

    scaler = MinMaxScaler()
    dataset.x = scaler.fit_transform(x)
    dataset.set_scaler = scaler
    
    save_pickle(os.path.join(PROCESSED_DATA_DIR, 'scaler.pickle'), dataset.scaler)
    save_pickle(os.path.join(PROCESSED_DATA_DIR, 'x.pickle'), dataset.x)
    save_pickle(os.path.join(PROCESSED_DATA_DIR, 'y.pickle'), dataset.y)