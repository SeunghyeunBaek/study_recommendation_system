# from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
import os
"""
https://heartbeat.fritz.ai/recommender-systems-with-python-part-iii-collaborative-filtering-singular-value-decomposition-5b5dcb3f242b
"""

def get_completed_matrix():

    # Load data
    cwd = os.path.dirname(__file__)
    data_dir = os.path.join(cwd, 'data/movie_lens_small/')
    rating_data_path = os.path.join(data_dir, 'ratings.csv')
    movie_data_path = os.path.join(data_dir, 'movies.csv')

    rating_df = pd.read_csv(rating_data_path)
    movie_df = pd.read_csv(movie_data_path)

    # Preprocessing
    rating_df.drop('timestamp', axis=1, inplace=True)
    movie_df.drop('genres', axis=1, inplace=True)

    # Make movie-user matrix
    merge_df = pd.merge(left=rating_df, right=movie_df, on='movieId', how='left')
    user_movie_matrix = merge_df.pivot_table(values='rating', index='userId', columns='title').fillna(0)
    # print(user_movie_matrix)
    # movie_user_matrix = user_movie_matrix.values.T

    # A = USV
    u, sigma, vt = svds(user_movie_matrix, k=50) # Number of latents
    sigma = np.diag(sigma)

    completed_matrix = np.dot(np.dot(u, sigma), vt) + user_movie_matrix.mean(axis=1).values.reshape(-1, 1)
    
    row_id = user_movie_matrix.index
    column_id = user_movie_matrix.columns
    
    completed_matrix = pd.DataFrame(completed_matrix, index=row_id, columns=column_id)

    return completed_matrix, user_movie_matrix
    
def get_prefer_movie(user_id, completed_matrix, ntop=5):

    movie_score_list = completed_matrix.loc[user_id, :].sort_values(ascending=False)
    top_movie = movie_score_list[:ntop]
    return top_movie

if __name__ == '__main__':
    
    user_id = 1
    completed_matrix, movie_user_matrix = get_completed_matrix()

    top_movie_infer = get_prefer_movie(user_id=user_id, completed_matrix=completed_matrix, ntop=10)
    top_movie_real = get_prefer_movie(user_id=user_id, completed_matrix=movie_user_matrix, ntop=10)
