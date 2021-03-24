from sklearn.model_selection import train_test_split
import pandas as pd
import os

PRJ_DIR = os.path.dirname(__file__) 
DATA_PATH = os.path.join(PRJ_DIR, 'data/adult.csv')
TRAIN_DATA_PATH = os.path.join(PRJ_DIR, 'data/train.csv')
TEST_DATA_PATH = os.path.join(PRJ_DIR, 'data/test.csv')
if __name__ == '__main__':

    df = pd.read_csv(DATA_PATH)

    # 라벨링: >50K: 1, <=50K: 0
    df['income_label'] = df['income'].apply(lambda x: 1 if ">50K" in x else 0)
    df.drop('income', axis=1, inplace=True)

    # Split
    train_df, test_df = train_test_split(
        df,
        test_size=.2,
        stratify=df['income_label'],
        random_state=42)
    
    train_df.to_csv(TRAIN_DATA_PATH)
    test_df.to_csv( TEST_DATA_PATH)