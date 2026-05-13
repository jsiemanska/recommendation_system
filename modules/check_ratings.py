import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def check_ratings(ratings_file, mode):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) 
    test_path = os.path.join(project_root, "data", "ratings_test.csv")
    valid_path = os.path.join(project_root, "data", "ratings_valid.csv")

    df = pd.read_csv(ratings_file)
    
    if mode == "predict":
        df2 = pd.read_csv(test_path)
    else:
        df2 = pd.read_csv(valid_path)

    rmse = np.sqrt(mean_squared_error(df['rating'], df2['rating']))
    mae = mean_absolute_error(df['rating'], df2['rating'])
    r2 = r2_score(df2['rating'], df['rating'])
    
    return rmse, mae, r2