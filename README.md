# Recomendation_system

### Dobrosława Hetmańczyk, Joanna Siemańska

## 1. Training

The following command:
```
project1_338681/python main.py --mode train --train_file data/ratings_train.csv --model_path models_trained/model_BEST.pkl --alg BEST  
```


- Reads the `ratings.csv` file
- Trains a `BEST` model
- Stores the model in a pickle file

---

## 2. Prediction

The following command:

```
project1_338681/python main.py --mode predict --input_file data/ratings_test_no_ratings.csv --model_path models_trained/model_BEST.pkl --output_file results/preds_BEST.csv --alg BEST
```

- Reads the `rating_test_no_ratings.csv` file (containing `userId` and `movieId`, but no ratings)
- Loads the trained model from `model_BEST.pkl`
- Uses the selected algorithm (`BEST`) to generate predictions
- Stores predictions (with columns `userId,movieId,rating`) in `preds_BEST.csv`

---


## Notes

- In the full project there are four methods implemented (`NMF`,`SVD1`, `SVD2`, `SGD`, `BEST`).
