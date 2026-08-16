# SVD ile collaborative filtering modeli egitimi

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

SCRIPT_DIR = Path(__file__).resolve().parent

with open(SCRIPT_DIR / "cf_mappers.pkl", "rb") as f:
    mappers = pickle.load(f)
user_mapper = mappers["user_mapper"]
movie_mapper = mappers["movie_mapper"]

train = pd.read_csv(SCRIPT_DIR / "ratings_train.csv")

n_users = len(user_mapper)
n_movies = len(movie_mapper)

user_index = train["userId"].map(user_mapper)
movie_index = train["movieId"].map(movie_mapper)

# sadece TRAIN puanlariyla matris kuruyoruz (test verisini modele hic gostermiyoruz)
train_matrix = csr_matrix((train["rating"], (user_index, movie_index)), shape=(n_users, n_movies))

# her kullanicinin train'deki ortalama puanini hesapla (mean-centering icin)
sums = np.array(train_matrix.sum(axis=1)).flatten()
counts = np.diff(train_matrix.indptr)
user_means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)

# sadece DOLU hucrelerden kullanicinin ortalamasini cikariyoruz (sparse yapiyi koruyarak)
centered_data = train_matrix.data - user_means[train_matrix.tocoo().row]
centered_matrix = csr_matrix((centered_data, train_matrix.indices, train_matrix.indptr), shape=train_matrix.shape)

# SVD: k gizli faktor ile matrisi ayristir
K = 20
U, sigma, Vt = svds(centered_matrix.astype(float), k=K)

print("=== SVD sonucu boyutlar ===")
print("U (kullanici x k):", U.shape)
print("sigma (tekil degerler):", sigma.shape)
print("Vt (k x film):", Vt.shape)

print("\n=== tekil degerler (buyukten kucuge onem sirasi) ===")
# svds sonuclari kucukten buyuge dondurur, okunabilirlik icin ters ceviriyoruz
print(np.sort(sigma)[::-1])

np.savez(
    SCRIPT_DIR / "svd_model.npz",
    U=U, sigma=sigma, Vt=Vt, user_means=user_means,
)
print("\nModel kaydedildi: svd_model.npz")
