import pandas as pd
import pickle
import torch
import os

from sklearn.metrics import accuracy_score

from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep import Trainer

from utils import load_pickle, save_pickle

PRJ_DIR = os.path.dirname(__file__) 
TEST_DATA_PATH = os.path.join(PRJ_DIR, 'data/test.csv')

MODEL_DIR = os.path.join(PRJ_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pth')
WIDE_PROC_PATH = os.path.join(MODEL_DIR, 'wide_proc.pickle')
DEEP_PROC_PATH = os.path.join(MODEL_DIR, 'deep_proc.pickle')

if __name__ == '__main__':

    # 데이터 불러오기
    test_df = pd.read_csv(TEST_DATA_PATH)

    # Trinaer 불러오기
    trainer = Trainer(model=MODEL_PATH, objective='binary')
    trainer.batch_size = 256

    # Preprocessor 불러오기
    wide_processor = load_pickle(WIDE_PROC_PATH)
    deep_processor = load_pickle(DEEP_PROC_PATH)

    x_wide = wide_processor.transform(test_df)
    x_deep = deep_processor.transform(test_df)
    
    # 예측
    y_pred = trainer.predict(X_wide=x_wide, X_tab=x_deep)
    y = test_df['income_label']

    print(f"Accuracy: {accuracy_score(y, y_pred)}")