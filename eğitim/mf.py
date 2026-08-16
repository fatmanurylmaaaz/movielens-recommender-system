"""
Matrix Factorization (gradyan inişiyle) - MovieLens (ml-latest-small)

svd.py'den farkı: gizli (latent) vektörler tek seferlik lineer cebir
çözümüyle değil, ncf.py'deki gibi gradyan inişiyle (Adam) epoch epoch
öğrenilir.
ncf.py'den farkı: MLP yok — sadece kullanıcı ve film vektörlerinin iç
çarpımı + bias terimleri (klasik Funk-MF / Netflix Prize tarzı matrix
factorization). NCF'in "derin" hali degil, MF'in "saf/klasik" hali.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from svd import load_data, build_mappers

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

CACHE_FILE = Path(__file__).resolve().parent / "cache" / "mf_model.pt"
TOP_N = 10
MIN_RATINGS = 20  # az oylanan filmlerin vektoru az veriyle fit edilir -> asiri skor riski
TEST_SIZE = 0.2
RANDOM_STATE = 42
EMBEDDING_DIM = 32
EPOCHS = 20
BATCH_SIZE = 512
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-3  # L2 regularizasyon: vektorlerin asiri buyumesini/overfit'i frenler

torch.manual_seed(RANDOM_STATE)


class RatingsDataset(Dataset):
    def __init__(self, ratings: pd.DataFrame, user_mapper: dict, movie_mapper: dict):
        self.user_idx = ratings["userId"].map(user_mapper).values.astype("int64")
        self.movie_idx = ratings["movieId"].map(movie_mapper).values.astype("int64")
        self.rating = ratings["rating"].values.astype("float32")

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, i):
        return self.user_idx[i], self.movie_idx[i], self.rating[i]


class MatrixFactorization(nn.Module):
    """pred(u,i) = global_mean + bias_u + bias_i + p_u . q_i"""

    def __init__(self, n_users: int, n_movies: int, global_mean: float, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, embedding_dim)
        self.movie_factors = nn.Embedding(n_movies, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_mean = global_mean

        nn.init.normal_(self.user_factors.weight, std=0.1)
        nn.init.normal_(self.movie_factors.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_idx, movie_idx):
        dot = (self.user_factors(user_idx) * self.movie_factors(movie_idx)).sum(dim=1)
        bias = self.user_bias(user_idx).squeeze(1) + self.movie_bias(movie_idx).squeeze(1)
        return self.global_mean + bias + dot


def train_mf(model: MatrixFactorization, train_loader: DataLoader, epochs: int = EPOCHS,
             lr: float = LEARNING_RATE, weight_decay: float = WEIGHT_DECAY):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for user_idx, movie_idx, rating in train_loader:
            optimizer.zero_grad()
            pred = model(user_idx, movie_idx)
            loss = criterion(pred, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(rating)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"  epoch {epoch:2d}/{epochs}  loss (MSE): {avg_loss:.4f}")

    return model


def train_or_load(n_users: int, n_movies: int, global_mean: float, train_loader: DataLoader,
                   use_cache: bool = True) -> MatrixFactorization:
    model = MatrixFactorization(n_users, n_movies, global_mean)
    if use_cache and CACHE_FILE.exists():
        model.load_state_dict(torch.load(CACHE_FILE, weights_only=True))
        model.eval()
        return model

    print(f"===== MF Eğitimi ({EPOCHS} epoch) =====")
    train_mf(model, train_loader)

    if use_cache:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CACHE_FILE)

    model.eval()
    return model


@torch.no_grad()
def evaluate(model: MatrixFactorization, test_ratings: pd.DataFrame, user_mapper: dict, movie_mapper: dict):
    test = test_ratings[
        test_ratings["userId"].isin(user_mapper) & test_ratings["movieId"].isin(movie_mapper)
    ].copy()
    user_idx = torch.tensor(test["userId"].map(user_mapper).values, dtype=torch.long)
    movie_idx = torch.tensor(test["movieId"].map(movie_mapper).values, dtype=torch.long)

    preds = model(user_idx, movie_idx).clamp(0.5, 5.0).numpy()
    y_true = test["rating"].values

    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
    mae = float(np.mean(np.abs(y_true - preds)))
    return rmse, mae, len(test)


@torch.no_grad()
def recommend_for_user(model: MatrixFactorization, user_id: int, movies: pd.DataFrame, ratings: pd.DataFrame,
                        user_mapper: dict, movie_mapper: dict, k: int = TOP_N, min_ratings: int = MIN_RATINGS):
    if user_id not in user_mapper:
        return []

    already_watched = set(ratings[ratings["userId"] == user_id]["movieId"])
    well_known = set(ratings.groupby("movieId").size().loc[lambda c: c >= min_ratings].index)
    candidate_movie_ids = [mid for mid in movie_mapper if mid not in already_watched and mid in well_known]

    user_idx = torch.tensor([user_mapper[user_id]] * len(candidate_movie_ids), dtype=torch.long)
    movie_idx = torch.tensor([movie_mapper[mid] for mid in candidate_movie_ids], dtype=torch.long)

    preds = model(user_idx, movie_idx).clamp(0.5, 5.0).numpy()
    order = np.argsort(preds)[::-1][:k]

    results = []
    for i in order:
        mid = candidate_movie_ids[i]
        title = movies.loc[movies["movieId"] == mid, "title"].values[0]
        results.append((title, float(preds[i])))
    return results


def main():
    ratings, movies = load_data()
    user_mapper, movie_mapper, movie_inv_mapper = build_mappers(ratings)

    train_ratings, test_ratings = train_test_split(ratings, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    global_mean = float(train_ratings["rating"].mean())

    train_dataset = RatingsDataset(train_ratings, user_mapper, movie_mapper)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = train_or_load(len(user_mapper), len(movie_mapper), global_mean, train_loader)

    rmse, mae, n = evaluate(model, test_ratings, user_mapper, movie_mapper)
    print(f"\n===== MF Değerlendirme =====")
    print(f"RMSE: {rmse:.4f}   MAE: {mae:.4f}   (n={n} tahmin)\n")

    for user_id in [1, 414]:
        print(f"===== Kullanıcı {user_id} için MF Önerisi =====")
        results = recommend_for_user(model, user_id, movies, ratings, user_mapper, movie_mapper)
        df = pd.DataFrame(results, columns=["title", "predicted_rating"])
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
