import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from modules.helper_functions import build_rating_matrix
from modules.train import find_optimal_r, train_nmf_model   

df = pd.read_csv("C:\\Users\\d-het\\MCaDR\\data\\ratings.csv")
df.drop(columns=["timestamp"], inplace=True)
# Z, user_map, movie_map, _= build_rating_matrix("C:\\Users\\d-het\\MCaDR\\data\\ratings.csv")
# print(Z.shape)
ratings_train, ratings_test = train_test_split(df, test_size=0.15, random_state=338681)
ratings_valid, ratings_test = train_test_split(ratings_test, test_size=0.35, random_state=338681)
ratings_valid_no_ratings = ratings_valid.drop(columns=["rating"])
ratings_test_no_ratings = ratings_test.drop(columns=["rating"])
# # Save the test file (no ratings, so integers are fine here for IDs)
np.savetxt("C:\\Users\\d-het\\MCaDR\\data\\ratings_valid_no_ratings.csv", ratings_valid_no_ratings.values, delimiter=",", fmt='%d', header="userId,movieId", comments="")
np.savetxt("C:\\Users\\d-het\\MCaDR\\data\\ratings_valid.csv", ratings_valid, delimiter=",", fmt='%d,%d,%.1f', header="userId,movieId,rating", comments="")
np.savetxt("C:\\Users\\d-het\\MCaDR\\data\\ratings_test_no_ratings.csv", ratings_test_no_ratings.values, delimiter=",", fmt='%d', header="userId,movieId", comments="")   
np.savetxt("C:\\Users\\d-het\\MCaDR\\data\\ratings_train.csv", ratings_train, delimiter=",", fmt='%d,%d,%.1f', header="userId,movieId,rating", comments="")
np.savetxt("C:\\Users\\d-het\\MCaDR\\data\\ratings_test.csv", ratings_test, delimiter=",", fmt='%d,%d,%.1f', header="userId,movieId,rating", comments="")