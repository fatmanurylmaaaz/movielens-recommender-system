"""
Neural Collaborative Filtering (NCF) - MovieLens (ml-latest-small)

svd.py'den fark: kullanıcı/film gizli vektörleri tek seferlik bir lineer
cebir çözümüyle (svds) değil, gradyan inişiyle (backpropagation) iteratif
olarak öğrenilir. Embedding katmanları + küçük bir MLP, epoch'lar boyunca
kayıp (loss) azaltılarak eğitilir -> "gerçek" iteratif model eğitimi.
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

CACHE_FILE = Path(__file__).resolve().parent / "cache" / "ncf_model.pt"
TOP_N = 10
MIN_RATINGS = 20  # az puanlanan filmlerin embedding'i az egitilir -> asiri/gurultulu tahminler verebilir
TEST_SIZE = 0.2
RANDOM_STATE = 42
EMBEDDING_DIM = 32
HIDDEN_DIMS = (64, 32, 16)
EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

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


class NCF(nn.Module):
    """Kullanıcı + film embedding'lerini birleştirip küçük bir MLP'den geçirir."""

    def __init__(self, n_users: int, n_movies: int, embedding_dim: int = EMBEDDING_DIM, hidden_dims=HIDDEN_DIMS):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx, movie_idx):
        u = self.user_embedding(user_idx)
        m = self.movie_embedding(movie_idx)
        x = torch.cat([u, m], dim=1)
        return self.mlp(x).squeeze(1)


def train_ncf(model: NCF, train_loader: DataLoader, epochs: int = EPOCHS, lr: float = LEARNING_RATE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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


def train_or_load(n_users: int, n_movies: int, train_loader: DataLoader, use_cache: bool = True) -> NCF:
    model = NCF(n_users, n_movies)
    if use_cache and CACHE_FILE.exists():
        model.load_state_dict(torch.load(CACHE_FILE, weights_only=True))
        model.eval()
        return model

    print(f"===== NCF Eğitimi ({EPOCHS} epoch) =====")
    train_ncf(model, train_loader)

    if use_cache:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CACHE_FILE)

    model.eval()
    return model


@torch.no_grad()
def evaluate(model: NCF, test_ratings: pd.DataFrame, user_mapper: dict, movie_mapper: dict):
    # mapper'da olmayan (train'de hic gorulmemis) kullanici/film test'te varsa elenir
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
def recommend_for_user(model: NCF, user_id: int, movies: pd.DataFrame, ratings: pd.DataFrame,
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

    train_dataset = RatingsDataset(train_ratings, user_mapper, movie_mapper)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = train_or_load(len(user_mapper), len(movie_mapper), train_loader)

    rmse, mae, n = evaluate(model, test_ratings, user_mapper, movie_mapper)
    print(f"\n===== NCF Değerlendirme =====")
    print(f"RMSE: {rmse:.4f}   MAE: {mae:.4f}   (n={n} tahmin)\n")

    for user_id in [1, 414]:
        print(f"===== Kullanıcı {user_id} için NCF Önerisi =====")
        results = recommend_for_user(model, user_id, movies, ratings, user_mapper, movie_mapper)
        df = pd.DataFrame(results, columns=["title", "predicted_rating"])
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
