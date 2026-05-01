import pandas as pd
import numpy as np


def predict_nmf(test_file, model_data):
    df       = pd.read_csv(test_file)
    Z_approx = model_data["Z_approx"]
    user_map = model_data["user_map"]
    movie_map= model_data["movie_map"]
    predictions = []
    for row in df.itertuples():
        u, m = row.userId, row.movieId
        if u in user_map and m in movie_map:
            rating = Z_approx[user_map[u], movie_map[m]]
        else:
            rating = 3.5
        rating_rounded = round(rating * 2) / 2
        predictions.append({"userId": u, "movieId": m, "rating": rating_rounded})
    return predictions