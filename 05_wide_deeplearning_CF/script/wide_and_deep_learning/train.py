
import pandas as pd
import numpy as np
import pickle
import os

from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep import Trainer

from utils import save_pickle

PRJ_DIR = os.path.dirname(__file__) 
TRAIN_DATA_PATH = os.path.join(PRJ_DIR, 'data/train.csv')
MODEL_DIR = os.path.join(PRJ_DIR, 'model')

MODEL_PATH = os.path.join(MODEL_DIR, 'model.pth')
WIDE_PROC_PATH = os.path.join(MODEL_DIR, 'wide_proc.pickle')
DEEP_PROC_PATH = os.path.join(MODEL_DIR, 'deep_proc.pickle')

if __name__ == '__main__':
    
    """
    데이터 불러오기
    """
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    """
    Wide, Deep 컬럼 나누기
    """

    # Wide
    wide_columns_list = [
        "education",
        "relationship",
        "workclass",
        "occupation",
        "native-country",
        "gender"]
    wide_cross_column_list = [
        ("education", "occupation"),
        ("native-country", "occupation")]
    
    # Deep
    deep_embedding_columns_list = [
        ("education", 16),
        ("workclass", 16),
        ("occupation", 16),
        ("native-country", 32)]
    deep_continuous_column_list = [
        "age",
        "hours-per-week"
    ]
    
    # Target
    target_column_list = [
        "income_label"
    ]
    
    target = train_df[target_column_list].values


    """
    Preprocessing
    """

    # Wide
    wide_preprocessor = WidePreprocessor(
        wide_cols=wide_columns_list,
        crossed_cols=wide_cross_column_list)

    x_wide = wide_preprocessor.fit_transform(train_df)

    # Deep
    tab_preprocessor = TabPreprocessor(
        embed_cols=deep_embedding_columns_list,
        continuous_cols=deep_continuous_column_list)
    x_deep = tab_preprocessor.fit_transform(train_df)


    """
    Model 구조 정의
    """
    # Model
    wide = Wide(wide_dim=np.unique(x_wide).shape[0], pred_dim=1)
    deeptabular = TabMlp(
        mlp_hidden_dims=[64, 32],
        column_idx=tab_preprocessor.column_idx,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=deep_continuous_column_list)
    model = WideDeep(wide=wide, deeptabular=deeptabular)

    """
    학습
    """
    trainer = Trainer(model, objective="binary", metrics=[Accuracy])
    trainer.fit(
        X_wide=x_wide,
        X_tab=x_deep,
        target=target,
        n_epochs=5,
        batch_size=256,
        val_split=0.1)
    
    trainer.save_model(MODEL_PATH)
    save_pickle(WIDE_PROC_PATH, wide_preprocessor)
    save_pickle(DEEP_PROC_PATH, tab_preprocessor)