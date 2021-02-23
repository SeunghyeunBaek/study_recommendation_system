# from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
import os
"""
https://heartbeat.fritz.ai/recommender-systems-with-python-part-iii-collaborative-filtering-singular-value-decomposition-5b5dcb3f242b
"""

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
movie_user_matrix = merge_df.pivot_table(values='rating', index='title', columns='userId').fillna(0)
# print(user_movie_matrix)
# movie_user_matrix = user_movie_matrix.values.T

# A = USV
u, sigma, vt = svds(movie_user_matrix, k=50) # Number of latents
sigma = np.diag(sigma)

completed_matrix = np.dot(np.dot(u, sigma), vt)

print(completed_matrix)


# def rec_movie(pre)