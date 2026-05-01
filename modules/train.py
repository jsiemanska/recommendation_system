import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from modules.helper_functions import build_rating_matrix
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

def find_optimal_r(train_file, r_candidates=[5, 10, 20, 30, 50]):
    Z, mask, user_map, movie_map = build_rating_matrix(train_file)
    
    np.random.seed(42)
    # Only sample from REAL ratings for validation
    real_indices = np.argwhere(mask)
    val_idx = real_indices[np.random.choice(len(real_indices), 
                           size=int(0.15 * len(real_indices)), replace=False)]
    
    val_mask = np.zeros_like(mask)
    for i, j in val_idx:
        val_mask[i, j] = True
    
    train_mask = mask & ~val_mask
    Z_train = Z.copy()
    # Hide validation ratings with global mean
    Z_train[val_mask] = Z[mask].mean()
    
    for r in r_candidates:
        Z_approx, _, _ = train_masked_nmf(Z_train, train_mask, n_components=r)
        predicted = np.clip(Z_approx[val_mask], 0.5, 5.0)
        actual = Z[val_mask]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        print(f"r={r:<4} RMSE={rmse:.4f}")

def train_nmf_model(train_file, n_components):
    Z, user_map, movie_map = build_rating_matrix(train_file)

    model = NMF(n_components, init='random', random_state=0, max_iter=1000000)
    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.dot(W, H)

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

