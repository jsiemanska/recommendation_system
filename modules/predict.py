import pandas as pd
import numpy as np


def predict(test_file, model_data):
    df        = pd.read_csv(test_file)
    Z_approx  = model_data["Z_approx"]
    user_map  = model_data["user_map"]
    movie_map = model_data["movie_map"]
    user_means  = model_data.get("user_means", {})   
    movie_means = model_data.get("movie_means", {})  
    global_mean = model_data.get("global_mean", 3.5)

    predictions = []
    for row in df.itertuples():
        u, m = row.userId, row.movieId

        if u in user_map and m in movie_map:
            # Both known — use matrix
            rating = Z_approx[user_map[u], movie_map[m]]

        elif u not in user_map and m in movie_map:
            # New user, known movie → use movie mean
            rating = movie_means.get(m, global_mean)

        elif u in user_map and m not in movie_map:
            # Known user, new movie → use user mean
            rating = user_means.get(u, global_mean)

        else:
            # Both unknown → global mean
            rating = global_mean

        rating = float(np.clip(rating, 1, 5))
        rating_rounded = round(rating * 2) / 2
        predictions.append({"userId": u, "movieId": m, "rating": rating_rounded})

    return predictions
