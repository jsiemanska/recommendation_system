import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from modules.helper_functions import build_rating_matrix
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

def find_optimal_r(train_file, r_candidates=[20, 25, 30]):
    best_overall = {"rmse": float("inf")}
    
    Z, real_mask, user_map, movie_map = build_rating_matrix(train_file)

    # --- split REAL ratings into train / val ---
    np.random.seed(42)
    real_idx   = np.argwhere(real_mask)
    n_val      = int(0.15 * len(real_idx))
    val_choice = np.random.choice(len(real_idx), size=n_val, replace=False)
    val_idx    = real_idx[val_choice]

    # Build Z_train
    Z_train = Z.copy()
    Z_fill_only = _fill_only(train_file, user_map, movie_map)

    for i, j in val_idx:
        Z_train[i, j] = Z_fill_only[i, j]

    actual = Z[val_idx[:, 0], val_idx[:, 1]]

    print(f"\nStrategy: weighted")
    print(f"{'r':<5} | {'Val RMSE':<12}")
    print("-" * 22)

    for r in r_candidates:
        model   = NMF(n_components=r, init="random", random_state=0, max_iter=10_000)
        W       = model.fit_transform(Z_train)
        H       = model.components_
        Z_approx = np.dot(W, H)
        predicted = np.clip(Z_approx[val_idx[:, 0], val_idx[:, 1]], 1, 5)
        rmse    = np.sqrt(mean_squared_error(actual, predicted))
        print(f"{r:<5} | {rmse:.4f}")

        if rmse < best_overall["rmse"]:
            best_overall = {"rmse": rmse, "r": r, "strategy": "weighted"}

    print("\n" + "=" * 40)
    print(f"BEST → strategy={best_overall['strategy']}, r={best_overall['r']}, RMSE={best_overall['rmse']:.4f}")
    return best_overall["r"]


def _fill_only(train_file, user_map, movie_map):
    df = pd.read_csv(train_file)
    unique_users  = sorted(user_map,  key=user_map.get)
    unique_movies = sorted(movie_map, key=movie_map.get)
    
    u_means = df.groupby("userId")["rating"].mean().reindex(unique_users).values
    m_means = df.groupby("movieId")["rating"].mean().reindex(unique_movies).values
    return (0.5 * u_means[:, np.newaxis] + 0.5 * m_means[np.newaxis, :]).astype(np.float32)


def train_nmf_model(train_file, n_components):
    Z, _, user_map, movie_map = build_rating_matrix(train_file)
    model    = NMF(n_components=n_components, init="random", random_state=0, max_iter=10_000)
    W        = model.fit_transform(Z)
    H        = model.components_
    Z_approx = np.clip(np.dot(W, H), 1, 5)
    return Z_approx, user_map, movie_map

def train_masked_nmf(Z, mask, n_components=20, n_iter=50):
    """
    Alternating NMF that only updates based on known ratings.
    After each full NMF step, restore known ratings before the next iteration.
    """
    global_mean = Z[mask].mean()
    Z_work = Z.copy()
    
    for iteration in range(n_iter):
        model = NMF(n_components=n_components, init='random', 
                    random_state=0, max_iter=5000000)
        W = model.fit_transform(Z_work)
        H = model.components_
        Z_approx = W @ H
        
        # Only replace UNKNOWN entries with current approximation
        Z_work = Z.copy()
        Z_work[~mask] = Z_approx[~mask]
        
        # Clip to valid range
        Z_work = np.clip(Z_work, 0.5, 5.0)
    
    return Z_work, W, H

def find_optimal_r_svd(train_file, r_candidates=[2, 5, 10, 20, 30, 40, 50]):
    Z, user_map, movie_map = build_rating_matrix(train_file)
    
    np.random.seed(42)
    mask = np.random.rand(*Z.shape) < 0.10
    
    Z_train = Z.copy()
    col_means = np.mean(Z, axis=0)
    for col in range(Z.shape[1]):
        Z_train[mask[:, col], col] = col_means[col]

    best_r = None
    min_rmse = float('inf')
    
    print(f"{'r':<5} | {'Validation RMSE':<15}")
    print("-" * 25)

    for r in r_candidates:
        # Inicjalizacja i dopasowanie SVD
        svd = TruncatedSVD(n_components=r, random_state=42)
        svd.fit(Z_train)
        
        # Matematyka z projektu: Z ≈ U_r * \Lambda_r * V_r^T
        Sigma2 = np.diag(svd.singular_values_)
        VT = svd.components_
        
        # Zabezpieczenie przed dzieleniem przez zero (na wypadek zerowych wartości osobliwych)
        safe_singular_values = np.where(svd.singular_values_ == 0, 1e-10, svd.singular_values_)
        
        # W = U_r
        W = svd.transform(Z_train) / safe_singular_values
        # H = \Lambda_r * V_r^T
        H = np.dot(Sigma2, VT)
        
        # Rekonstrukcja
        Z_approx = np.dot(W, H)
        
        # Obliczanie RMSE tylko na ukrytych 10%
        actual = Z[mask]
        predicted = Z_approx[mask]
        
        # Przycinanie wyników do skali ocen (np. 1-5), co zazwyczaj poprawia RMSE
        predicted = np.clip(predicted, 1, 5)
        
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        print(f"{r:<5} | {rmse:.4f}")

        if rmse < min_rmse:
            min_rmse = rmse
            best_r = r

    print("-" * 25)
    print(f"Optimal r (SVD): {best_r} (RMSE: {min_rmse:.4f})")
    return best_r


def train_svd_model(train_file, n_components):
    Z, user_map, movie_map = build_rating_matrix(train_file)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(Z)
    
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    
    safe_singular_values = np.where(svd.singular_values_ == 0, 1e-10, svd.singular_values_)
    W = svd.transform(Z) / safe_singular_values
    H = np.dot(Sigma2, VT)
    
    Z_approx = np.dot(W, H)

    return Z_approx, user_map, movie_map

