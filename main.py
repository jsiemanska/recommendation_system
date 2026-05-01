import numpy as np
from modules.check_ratings import check_ratings
from modules.train import find_optimal_r, train_nmf_model
import argparse
import os
import pickle
from modules.predict import predict_nmf

#żeby odpalić test python main.py --mode predict --input_file data/ratings_test_no_ratings.csv --model_path models_trained/model_NMF.pkl --output_file results/preds.csv --alg NMF   

#żeby odpalić predict python main.py --mode train --train_file data/ratings_train.csv --model_path models_trained/model_NMF.pkl --alg NMF 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple NMF-based Recommender")

    parser.add_argument("--mode", type=str, required=True,
                        help="Mode of operation: 'train' or 'predict'.")

    parser.add_argument("--train_file", type=str, default="data/ratings.csv",
                        help="CSV file with training data (userId,movieId,rating).")

    parser.add_argument("--input_file", type=str, default="data/sample_test.csv",
                        help="CSV file with (userId,movieId) for prediction.")

    parser.add_argument("--model_path", type=str,
                        default="models_trained/model_NMF.pkl",
                        help="Path to save/load the trained model.")

    parser.add_argument("--output_file", type=str,
                        default="results/preds.csv",
                        help="Where to save predictions.")

    parser.add_argument("--alg", type=str,   required=True,
                        help="Algorithm to use (NMF, SVD1, SVD2, SGD or BEST. Only 'NMF' is implemented in sample project).")

    return parser.parse_args()


def main():
    args = parse_arguments()
    mode = args.mode.lower()
    alg = args.alg.upper()

    if alg != "NMF":
        print("Only --alg NMF is implemented in this sample project.")
        return

    if mode == "train":
        print("Training mode activated (NMF).")

        r = find_optimal_r(args.train_file, r_candidates=[26,27,28,29])
        Z_approx, user_map, movie_map = train_nmf_model(args.train_file, r)

        model_data = {
            "Z_approx": Z_approx,
            "user_map": user_map,
            "movie_map": movie_map
        }

        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

        with open(args.model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {args.model_path}")

    elif mode == "predict":
        print("Prediction mode activated (NMF).")

        if not os.path.exists(args.model_path):
            print("Model file does not exist. Please run training first.")
            return

        with open(args.model_path, "rb") as f:
            model_data = pickle.load(f)

        predictions = predict_nmf(args.input_file, model_data)

        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

        with open(args.output_file, "w") as f:
            f.write("userId,movieId,rating\n")
            for row in predictions:
                f.write(f"{row['userId']},{row['movieId']},{row['rating']}\n")

        print(f"Predictions saved to {args.output_file}")

        rmse, mae, r2 = check_ratings(args.output_file, mode = "predict")

        print(f"Rating statistics for {args.output_file}:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R^2: {r2:.4f}")
    elif mode == "validate":
        print("Validation mode activated (NMF).")

        if not os.path.exists(args.model_path):
            print("Model file does not exist. Please run training first.")
            return

        with open(args.model_path, "rb") as f:
            model_data = pickle.load(f)

        predictions = predict_nmf(args.input_file, model_data)

        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

        with open(args.output_file, "w") as f:
            f.write("userId,movieId,rating\n")
            for row in predictions:
                f.write(f"{row['userId']},{row['movieId']},{row['rating']}\n")

        print(f"Predictions saved to {args.output_file}")

        rmse, mae, r2 = check_ratings(args.output_file, mode = "validate")

        print(f"Rating statistics for {args.output_file}:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R^2: {r2:.4f}")

    else:
        print("Invalid --mode. Use 'train' or 'predict'.")


if __name__ == "__main__":
    main()

