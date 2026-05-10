import numpy as np
from modules.check_ratings import check_ratings
from modules.train import find_optimal_r, train_nmf_model, find_optimal_r_svd1, train_svd1_model, find_optimal_r_svd2, train_svd2_model, find_optimal_r_sgd, train_sgd_model
import argparse
import os
import pickle
from modules.predict import predict

#żeby odpalić test python main.py --mode predict --input_file data/ratings_test_no_ratings.csv --model_path models_trained/model_NMF.pkl --output_file results/preds.csv --alg NMF   

#żeby odpalić predict python main.py --mode train --train_file data/ratings_train.csv --model_path models_trained/model_NMF.pkl --alg NMF 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple NMF/SVD1-based Recommender")
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--train_file", type=str, default="data/ratings_train.csv")
    parser.add_argument("--input_file", type=str, default="data/ratings_test_no_ratings.csv")
    parser.add_argument("--model_path", type=str, default="models_trained/model_NMF.pkl")
    parser.add_argument("--output_file", type=str, default="results/preds.csv")
    parser.add_argument("--alg", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_arguments()
    mode = args.mode.lower()
    alg  = args.alg.upper()

    if alg not in ("NMF", "SVD1", "SVD2", "SGD"):
        print(f"Algorithm '{alg}' not implemented. Use NMF, SVD1, SVD2, or SGD.")
        return

    if mode == "train":
        print(f"Training mode activated ({alg}).")

        if alg == "NMF":
            r = find_optimal_r(args.train_file, r_candidates=[25])
            Z_approx, user_map, movie_map, user_means, movie_means, global_mean = train_nmf_model(args.train_file, r)

        elif alg == "SVD1":
            r = find_optimal_r_svd1(args.train_file, r_candidates=[50,55,60])
            Z_approx, user_map, movie_map, user_means, movie_means, global_mean = train_svd1_model(args.train_file, r)
            
        elif alg == "SVD2":
            r = find_optimal_r_svd2(args.train_file, r_candidates=np.arange(10, 61, 2))
            Z_approx, user_map, movie_map, user_means, movie_means, global_mean = train_svd2_model(args.train_file, r)
            
        
        elif alg == "SGD":
            r = find_optimal_r_sgd(args.train_file, r_candidates=np.arange(10, 61, 2))
            Z_approx, user_map, movie_map, user_means, movie_means, global_mean = train_sgd_model(args.train_file, n_factors=r)
            

        print("Trening zakończony, zapisuję model...")
        model_data = {"Z_approx": Z_approx, "user_map": user_map, "movie_map": movie_map, "user_means": user_means, "movie_means": movie_means, "global_mean": global_mean}
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        with open(args.model_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {args.model_path}")

    elif mode in ("predict", "validate"):
        print(f"{'Prediction' if mode == 'predict' else 'Validation'} mode activated ({alg}).")

        if not os.path.exists(args.model_path):
            print("Model file does not exist. Please run training first.")
            return

        with open(args.model_path, "rb") as f:
            model_data = pickle.load(f)

        predictions = predict(args.input_file, model_data)

        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            f.write("userId,movieId,rating\n")
            for row in predictions:
                f.write(f"{row['userId']},{row['movieId']},{row['rating']}\n")
        print(f"Predictions saved to {args.output_file}")

        rmse, mae, r2 = check_ratings(args.output_file, mode=mode)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R^2:  {r2:.4f}")

    else:
        print("Invalid --mode. Use 'train', 'predict', or 'validate'.")


if __name__ == "__main__":
    main()