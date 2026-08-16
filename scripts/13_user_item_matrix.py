# User-item matrisi olusturma (SVD icin hazirlik)

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"
ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")

# userId / movieId gercek degerlerini 0'dan baslayan siralı indekslere ceviriyoruz
user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_mapper = {uid: i for i, uid in enumerate(user_ids)}
movie_mapper = {mid: i for i, mid in enumerate(movie_ids)}
user_inv_mapper = {i: uid for uid, i in user_mapper.items()}
movie_inv_mapper = {i: mid for mid, i in movie_mapper.items()}

user_index = ratings["userId"].map(user_mapper)
movie_index = ratings["movieId"].map(movie_mapper)

n_users = len(user_mapper)
n_movies = len(movie_mapper)

# satir = kullanici, sutun = film -> csr_matrix ile seyrek (sparse) matris
user_item = csr_matrix((ratings["rating"], (user_index, movie_index)), shape=(n_users, n_movies))

print("=== user-item matris boyutu ===")
print(f"{n_users} kullanici x {n_movies} film")

dolu_hucre = user_item.nnz
toplam_hucre = n_users * n_movies
print(f"dolu hucre: {dolu_hucre}, toplam hucre: {toplam_hucre}")
print(f"sparsity: %{(1 - dolu_hucre / toplam_hucre) * 100:.2f}  (EDA'daki %98.30 ile tutarli olmali)")

# her kullanicinin ortalama puanini hesapla (mean-centering icin)
user_means = np.array(user_item.sum(axis=1)).flatten() / np.diff(user_item.indptr).clip(min=1)

print("\n=== ilk 5 kullanicinin ortalama puani ===")
print(user_means[:5])

# matrisleri ve mapper'lari diskte saklayalim ki sonraki adimlarda yeniden kullanabilelim
import scipy.sparse as sp
sp.save_npz(SCRIPT_DIR / "user_item_matrix.npz", user_item)
np.save(SCRIPT_DIR / "user_means.npy", user_means)

import pickle
with open(SCRIPT_DIR / "cf_mappers.pkl", "wb") as f:
    pickle.dump(
        {
            "user_mapper": user_mapper,
            "movie_mapper": movie_mapper,
            "user_inv_mapper": user_inv_mapper,
            "movie_inv_mapper": movie_inv_mapper,
        },
        f,
    )

print("\nKaydedildi: user_item_matrix.npz, user_means.npy, cf_mappers.pkl")
