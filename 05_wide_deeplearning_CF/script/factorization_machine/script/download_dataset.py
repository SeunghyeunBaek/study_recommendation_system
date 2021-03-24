
"""데이터 다운로드

"""

import os, sys
sys.path.append('/workspace/study_recsys/05_widedeepfm/FM/')

from sklearn.datasets import load_breast_cancer
from module.utils import load_yaml, save_pickle
import os

config_path = '/workspace/study_recsys/05_widedeepfm/FM/config.yml'
config = load_yaml(config_path)

DATA_DIR = os.path.join(config['DIRECTORY']['original_data'])

if __name__ == '__main__':

    x, y = load_breast_cancer(return_X_y=True, as_frame=False)
    save_pickle(os.path.join(DATA_DIR, 'x.pickle'), x)
    save_pickle(os.path.join(DATA_DIR, 'y.pickle'), y)