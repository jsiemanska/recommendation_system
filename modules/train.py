import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from modules.helper_functions import build_rating_matrix
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

def find_optimal_r(train_file, r_candidates=[20, 25, 30]):
    best_overall = {"rmse": float("inf")}
    
    Z, real_mask, user_map, movie_map = build_rating_matrix(train_file)

    np.random.seed(42)
    real_idx  = np.argwhere(real_mask)
    n_val = int(0.2 * len(real_idx))
    val_choice = np.random.choice(len(real_idx), size=n_val, replace=False)
    val_idx = real_idx[val_choice]

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
        model = NMF(n_components=r, init="random", random_state=0, max_iter=10_000)
        W = model.fit_transform(Z_train)
        H = model.components_
        Z_approx = np.dot(W, H)
        predicted = np.clip(Z_approx[val_idx[:, 0], val_idx[:, 1]], 1, 5)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        print(f"{r:<5} | {rmse:.4f}")

        if rmse < best_overall["rmse"]:
            best_overall = {"rmse": rmse, "r": r, "strategy": "weighted"}

    print("\n" + "=" * 40)
    print(f"BEST → strategy={best_overall['strategy']}, r={best_overall['r']}, RMSE={best_overall['rmse']:.4f}")
    return best_overall["r"]


def _fill_only(train_file, user_map, movie_map):
    df = pd.read_csv(train_file)
    unique_users = sorted(user_map,  key=user_map.get)
    unique_movies = sorted(movie_map, key=movie_map.get)
    
    u_means = df.groupby("userId")["rating"].mean().reindex(unique_users).values
    m_means = df.groupby("movieId")["rating"].mean().reindex(unique_movies).values
    return (0.5 * u_means[:, np.newaxis] + 0.5 * m_means[np.newaxis, :]).astype(np.float32)


def train_nmf_model(train_file, n_components):
    df = pd.read_csv(train_file)
    Z, _, user_map, movie_map = build_rating_matrix(train_file)
    model = NMF(n_components=n_components, init="random", random_state=0, max_iter=10_000)
    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.clip(np.dot(W, H), 1, 5)
    user_means  = df.groupby("userId")["rating"].mean().to_dict()
    movie_means = df.groupby("movieId")["rating"].mean().to_dict()
    global_mean = float(df["rating"].mean())  

    return Z_approx, user_map, movie_map, user_means, movie_means, global_mean

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
        
        Z_work = Z.copy()
        Z_work[~mask] = Z_approx[~mask]
        
        # Clip to valid range
        Z_work = np.clip(Z_work, 0.5, 5.0)
    
    return Z_work, W, H

def train_svd1_model(train_file, n_components):
    df = pd.read_csv(train_file)
    Z, _, user_map, movie_map = build_rating_matrix(train_file)
    
    svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=3)
    svd.fit(Z)
    
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(Z) / svd.singular_values_
    H = np.dot(Sigma2, VT)
    
    Z_approx = np.clip(np.dot(W, H), 1, 5)
    global_mean = float(df["rating"].mean())
    user_means  = df.groupby("userId")["rating"].mean().to_dict()
    movie_means = df.groupby("movieId")["rating"].mean().to_dict()

    return Z_approx, user_map, movie_map, user_means, movie_means, global_mean

def find_optimal_r_svd1(train_file, r_candidates=[90,95]):
    best_overall = {"rmse": float("inf"), "strategy": "weighted"}

    # Uses the simplified build_rating_matrix
    Z, real_mask, user_map, movie_map = build_rating_matrix(train_file)

    # Hold out 15% of REAL ratings for validation
    np.random.seed(42)
    real_idx = np.argwhere(real_mask)
    n_val = int(0.15 * len(real_idx))
    val_choice = np.random.choice(len(real_idx), size=n_val, replace=False)
    val_idx = real_idx[val_choice]

    # Hide held-out real ratings using the simplified _fill_only
    Z_fill_only = _fill_only(train_file, user_map, movie_map)
    Z_train = Z.copy()
    for i, j in val_idx:
        Z_train[i, j] = Z_fill_only[i, j]

    actual = Z[val_idx[:, 0], val_idx[:, 1]]

    print(f"\nStrategy: weighted (SVD)")
    print(f"{'r':<5} | {'Val RMSE':<12}")
    print("-" * 22)

    for r in r_candidates:
        max_r = min(Z_train.shape) - 1
        if r >= max_r:
            print(f"{r:<5} | skipped (r >= {max_r})")
            continue

        svd = TruncatedSVD(n_components=r, random_state=42)
        svd.fit(Z_train)

        Sigma2 = np.diag(svd.singular_values_)
        VT = svd.components_
        W  = svd.transform(Z_train) / svd.singular_values_
        H  = np.dot(Sigma2, VT)
        Z_approx = np.dot(W, H)

        predicted = np.clip(Z_approx[val_idx[:, 0], val_idx[:, 1]], 1, 5)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        print(f"{r:<5} | {rmse:.4f}")

        if rmse < best_overall["rmse"]:
            best_overall["rmse"] = rmse
            best_overall["r"] = r

    print("\n" + "=" * 40)
    print(f"BEST SVD → r={best_overall['r']}, RMSE={best_overall['rmse']:.4f}")
    return best_overall["r"]

def train_svd2_model(train_file, n_components):
    df = pd.read_csv(train_file)
    Z, _, user_map, movie_map = build_rating_matrix(train_file)
    
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    U_r = U[:, :n_components]
    s_r = s[:n_components]
    Vt_r = Vt[:n_components, :]
    Z_approx = U_r @ np.diag(s_r) @ Vt_r
    Z_approx = np.clip(Z_approx, 1, 5)

    user_means = df.groupby("userId")["rating"].mean().to_dict()
    movie_means = df.groupby("movieId")["rating"].mean().to_dict()
    global_mean = float(df["rating"].mean())
    return Z_approx, user_map, movie_map, user_means, movie_means, global_mean


def find_optimal_r_svd2(train_file, r_candidates=[40,45,50,55,60]):
    best_overall = {"rmse": float("inf"), "strategy": "weighted"}    
    Z, real_mask, user_map, movie_map = build_rating_matrix(train_file)

        
    np.random.seed(42)
    real_idx = np.argwhere(real_mask)
    n_val = int(0.15 * len(real_idx))
    val_choice = np.random.choice(len(real_idx), size=n_val, replace=False)
    val_idx = real_idx[val_choice]

    Z_fill_only = _fill_only(train_file, user_map, movie_map)
    Z_train = Z.copy()
    for i, j in val_idx:
        Z_train[i, j] = Z_fill_only[i, j]
    actual = Z[val_idx[:, 0], val_idx[:, 1]]

    print("\nSVD2 – finding r")
    print(f"{'r':<5} | {'Val RMSE':<12}")
    print("-" * 22)
    best_rmse = float("inf")
    best_r = None
    
    
    for r in r_candidates:
        if r > min(Z_train.shape) - 1:
            print(f"{r:<5} | skipped (r > {min(Z_train.shape)-1})")
            continue
        
        U, s, Vt = np.linalg.svd(Z_train, full_matrices=False)
        U_r = U[:, :r]
        s_r = s[:r]
        Vt_r = Vt[:r, :]
        Z_approx = U_r @ np.diag(s_r) @ Vt_r
        
        preds = np.clip(Z_approx[val_idx[:, 0], val_idx[:, 1]], 1, 5)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        print(f"{r:<5} | {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_r = r
            
    print(f"Best SVD2: r = {best_r} (RMSE={best_rmse:.4f})")
    return best_r


def train_sgd_model(train_file, n_factors=50, learning_rate=0.005, reg=0.02, epochs=30):
    df = pd.read_csv(train_file)
    unique_users = sorted(df["userId"].unique())
    unique_movies = sorted(df["movieId"].unique())
    user_map = {uid: i for i, uid in enumerate(unique_users)}
    movie_map = {mid: j for j, mid in enumerate(unique_movies)}
    n_users, n_movies = len(user_map), len(movie_map)

    global_mean = df["rating"].mean()
    bu = np.zeros(n_users)
    bi = np.zeros(n_movies)
    np.random.seed(0)
    P = np.random.normal(0, 0.1, (n_users, n_factors))
    Q = np.random.normal(0, 0.1, (n_movies, n_factors))

    ratings = [(user_map[r.userId], movie_map[r.movieId], r.rating) for r in df.itertuples()]

    for epoch in range(epochs):
        np.random.shuffle(ratings)
        total_err = 0
        for u, m, r in ratings:
            pred = global_mean + bu[u] + bi[m] + np.dot(P[u], Q[m])
            err = r - pred
            total_err += err**2
            bu[u] += learning_rate * (err - reg * bu[u])
            bi[m] += learning_rate * (err - reg * bi[m])
            P[u] += learning_rate * (err * Q[m] - reg * P[u])
            Q[m] += learning_rate * (err * P[u] - reg * Q[m])
        rmse = np.sqrt(total_err / len(ratings))
        print(f"Epoch {epoch+1}/{epochs}, RMSE: {rmse:.4f}")


    Z_approx = np.zeros((n_users, n_movies))
    for u in range(n_users):
        for m in range(n_movies):
            pred = global_mean + bu[u] + bi[m] + np.dot(P[u], Q[m])
            Z_approx[u, m] = np.clip(pred, 1, 5)

    user_means = df.groupby("userId")["rating"].mean().to_dict()
    movie_means = df.groupby("movieId")["rating"].mean().to_dict()
    return Z_approx, user_map, movie_map, user_means, movie_means, global_mean

def find_optimal_r_sgd(train_file, r_candidates=np.arange(10, 101, 5),
                        learning_rate=0.005, reg=0.02, epochs=20):
    
    df = pd.read_csv(train_file)
    unique_users = sorted(df["userId"].unique())
    unique_movies = sorted(df["movieId"].unique())
    user_map = {uid: i for i, uid in enumerate(unique_users)}
    movie_map = {mid: j for j, mid in enumerate(unique_movies)}
    n_users, n_movies = len(user_map), len(movie_map)

    all_ratings = [(user_map[r.userId], movie_map[r.movieId], r.rating) for r in df.itertuples()]
    np.random.seed(42)
    np.random.shuffle(all_ratings)
    split = int(0.8 * len(all_ratings))
    train_ratings = all_ratings[:split]
    val_ratings = all_ratings[split:]

    print("\nSGD – finding r")
    print(f"{'r':<5} | {'Val RMSE':<12}")
    print("-" * 22)
    best_r, best_rmse = None, float("inf")
    for r in r_candidates:
        P = np.random.normal(0, 0.1, (n_users, r))
        Q = np.random.normal(0, 0.1, (n_movies, r))
        bu = np.zeros(n_users)
        bi = np.zeros(n_movies)
        global_mean = df["rating"].mean()

        for _ in range(epochs):
            np.random.shuffle(train_ratings)
            for u, m, rating in train_ratings:
                pred = global_mean + bu[u] + bi[m] + np.dot(P[u], Q[m])
                err = rating - pred
                bu[u] += learning_rate * (err - reg * bu[u])
                bi[m] += learning_rate * (err - reg * bi[m])
                P[u] += learning_rate * (err * Q[m] - reg * P[u])
                Q[m] += learning_rate * (err * P[u] - reg * Q[m])

        errors = []
        for u, m, true_r in val_ratings:
            pred = global_mean + bu[u] + bi[m] + np.dot(P[u], Q[m])
            pred = np.clip(pred, 1, 5)
            errors.append((true_r - pred)**2)
        rmse = np.sqrt(np.mean(errors))
        print(f"{r:<5} | {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_r = r
    print(f"Best r for SGD: {best_r} (RMSE={best_rmse:.4f})")
    return best_r


def train_hybrid_svd_sgd_improved(train_file, n_factors=50, svd_weight=0.4, learning_rate=0.01,
                                 reg=0.02, epochs=40, lr_decay=0.98,
                                 damping=5.0):
    """Improved hybrid model: blend SVD2 decomposition with SGD personalized factors."""
    df = pd.read_csv(train_file)
    Z, _, user_map, movie_map = build_rating_matrix(train_file)
    unique_users = sorted(df["userId"].unique())
    unique_movies = sorted(df["movieId"].unique())
    n_users, n_movies = len(user_map), len(movie_map)

    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    U_r = U[:, :n_factors]
    s_r = s[:n_factors]
    Vt_r = Vt[:n_factors, :]
    Z_svd = U_r @ np.diag(s_r) @ Vt_r

    global_mean = df["rating"].mean()
    bu = np.zeros(n_users)
    bi = np.zeros(n_movies)
    np.random.seed(42)
    P = np.random.normal(0, 0.01, (n_users, n_factors))
    Q = np.random.normal(0, 0.01, (n_movies, n_factors))

    ratings = [(user_map[r.userId], movie_map[r.movieId], r.rating) for r in df.itertuples()]
    current_lr = learning_rate

    for epoch in range(epochs):
        np.random.shuffle(ratings)
        total_err = 0.0
        for u, m, r in ratings:
            svd_pred = Z_svd[u, m]
            sgd_pred = global_mean + bu[u] + bi[m] + np.dot(P[u], Q[m])
            pred = svd_weight * svd_pred + (1 - svd_weight) * sgd_pred
            err = r - pred
            total_err += err ** 2

            bu[u] += current_lr * (err - (reg * bu[u]) / (damping + 1))
            bi[m] += current_lr * (err - (reg * bi[m]) / (damping + 1))
            P[u] += current_lr * (err * Q[m] - reg * P[u])
            Q[m] += current_lr * (err * P[u] - reg * Q[m])

        rmse = np.sqrt(total_err / len(ratings))
        current_lr *= lr_decay
        print(f"Epoch {epoch+1}/{epochs}, RMSE: {rmse:.4f}, LR: {current_lr:.6f}")

    Z_approx = np.zeros((n_users, n_movies))
    for u in range(n_users):
        for m in range(n_movies):
            svd_pred = Z_svd[u, m]
            sgd_pred = global_mean + bu[u] + bi[m] + np.dot(P[u], Q[m])
            pred = svd_weight * svd_pred + (1 - svd_weight) * sgd_pred
            Z_approx[u, m] = np.clip(pred, 1, 5)

    user_means = df.groupby("userId")["rating"].mean().to_dict()
    movie_means = df.groupby("movieId")["rating"].mean().to_dict()
    global_mean = float(df["rating"].mean())
    return Z_approx, user_map, movie_map, user_means, movie_means, global_mean


def find_optimal_hybrid(train_file,
                        n_factors_candidates=[20, 30, 40, 50, 60],
                        svd_weight_candidates=[0.2, 0.3, 0.4, 0.5, 0.6],
                        learning_rates=[0.005, 0.01, 0.02],
                        regs=[0.01, 0.02, 0.03],
                        epochs=20,
                        lr_decay=0.98,
                        damping=5.0):
    """Grid search for improved hybrid model parameters."""
    df = pd.read_csv(train_file)
    unique_users = sorted(df["userId"].unique())
    unique_movies = sorted(df["movieId"].unique())
    user_map = {uid: i for i, uid in enumerate(unique_users)}
    movie_map = {mid: j for j, mid in enumerate(unique_movies)}
    n_users, n_movies = len(user_map), len(movie_map)

    all_ratings = [(user_map[r.userId], movie_map[r.movieId], r.rating) for r in df.itertuples()]
    np.random.seed(42)
    np.random.shuffle(all_ratings)
    split = int(0.8 * len(all_ratings))
    train_ratings = all_ratings[:split]
    val_ratings = all_ratings[split:]

    Z, _, _, _ = build_rating_matrix(train_file)
    best_params = {"rmse": float("inf")}

    print("\n" + "="*80)
    print("GRID SEARCH: HYBRID SVD + SGD Hyperparameter Tuning")
    print("="*80)
    print(f"{'n_factors':<10} {'svd_w':<8} {'lr':<8} {'reg':<8} {'Val RMSE':<10}")
    print("-"*54)

    for n_factors in n_factors_candidates:
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        U_r = U[:, :n_factors]
        s_r = s[:n_factors]
        Vt_r = Vt[:n_factors, :]
        Z_svd_base = U_r @ np.diag(s_r) @ Vt_r

        for svd_weight in svd_weight_candidates:
            for lr in learning_rates:
                for reg in regs:
                    bu = np.zeros(n_users)
                    bi = np.zeros(n_movies)
                    np.random.seed(42)
                    P = np.random.normal(0, 0.01, (n_users, n_factors))
                    Q = np.random.normal(0, 0.01, (n_movies, n_factors))
                    current_lr = lr

                    for _ in range(epochs):
                        np.random.shuffle(train_ratings)
                        for u, m, r in train_ratings:
                            base = Z_svd_base[u, m]
                            residual = r - base
                            pred_residual = bu[u] + bi[m] + np.dot(P[u], Q[m])
                            err = residual - pred_residual
                            bu[u] += current_lr * (err - (reg * bu[u]) / (damping + 1))
                            bi[m] += current_lr * (err - (reg * bi[m]) / (damping + 1))
                            P[u] += current_lr * (err * Q[m] - reg * P[u])
                            Q[m] += current_lr * (err * P[u] - reg * Q[m])
                        current_lr *= lr_decay

                    errors = []
                    for u, m, r in val_ratings:
                        base = Z_svd_base[u, m]
                        pred = base + bu[u] + bi[m] + np.dot(P[u], Q[m])
                        pred = np.clip(pred, 1, 5)
                        errors.append((r - pred) ** 2)
                    rmse = np.sqrt(np.mean(errors))

                    print(f"{n_factors:<10} {svd_weight:<8} {lr:<8} {reg:<8} {rmse:.4f}")
                    if rmse < best_params["rmse"]:
                        best_params = {
                            "rmse": rmse,
                            "n_factors": n_factors,
                            "svd_weight": svd_weight,
                            "lr": lr,
                            "reg": reg
                        }

    print("="*80)
    print(f"BEST PARAMS: n_factors={best_params['n_factors']}, svd_weight={best_params['svd_weight']}, lr={best_params['lr']}, reg={best_params['reg']}, RMSE={best_params['rmse']:.4f}")
    print("="*80)
    return best_params


