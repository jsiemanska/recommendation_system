import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def check_ratings(ratings_file):
    df = pd.read_csv(ratings_file)
    df2 = pd.read_csv("C:\\Users\\d-het\\MCaDR\\data\\ratings_test.csv")

    rmse = np.sqrt(mean_squared_error(df['rating'], df2['rating']))
    mae = mean_absolute_error(df['rating'], df2['rating'])
    r2 = r2_score(df2['rating'], df['rating'])
    return rmse, mae, r2


    