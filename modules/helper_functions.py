
import pandas as pd
import numpy as np

from sklearn.decomposition import NMF
import numpy as np

def build_rating_matrix(train_file):
    df = pd.read_csv(train_file)
    unique_users = sorted(df["userId"].unique())
    unique_movies = sorted(df["movieId"].unique())
    user_map = {uid: i for i, uid in enumerate(unique_users)}
    movie_map = {mid: j for j, mid in enumerate(unique_movies)}
    
    n_users = len(user_map)
    n_movies = len(movie_map)
    
    # Fill with GLOBAL mean, not user mean (less biased for NMF)
    global_mean = df["rating"].mean()
    Z = np.full((n_users, n_movies), global_mean, dtype=np.float32)
    mask = np.zeros((n_users, n_movies), dtype=bool)  # True = real rating
    
    for row in df.itertuples():
        i = user_map[row.userId]
        j = movie_map[row.movieId]
        Z[i, j] = row.rating
        mask[i, j] = True
    
    return Z, mask, user_map, movie_map

# def build_rating_matrix(train_file, weight_user=0.5):
#     df = pd.read_csv(train_file)

#     # Obliczamy średnie dla użytkowników i filmów
#     user_means_ser = df.groupby("userId")["rating"].mean()
#     movie_means_ser = df.groupby("movieId")["rating"].mean()

#     unique_users = sorted(df["userId"].unique())
#     unique_movies = sorted(df["movieId"].unique())

#     user_map = {uid: i for i, uid in enumerate(unique_users)}
#     movie_map = {mid: j for j, mid in enumerate(unique_movies)}

#     n_users = len(user_map)
#     n_movies = len(movie_map)

#     u_means = user_means_ser.reindex(unique_users).values
#     m_means = movie_means_ser.reindex(unique_movies).values

#     # Tworzymy macierz wypełnioną średnią ważoną
#     weight_movie = 1.0 - weight_user
#     Z = (weight_user * u_means[:, np.newaxis]) + (weight_movie * m_means[np.newaxis, :])
    
#     Z = Z.astype(np.float32)

#     # Nadpisujemy wartości tam, gdzie ocena faktycznie istnieje w zbiorze treningowym
#     for row in df.itertuples():
#         u = row.userId
#         m = row.movieId
#         rating = row.rating
#         i = user_map[u]
#         j = movie_map[m]
#         Z[i, j] = rating

#     return Z, user_map, movie_map