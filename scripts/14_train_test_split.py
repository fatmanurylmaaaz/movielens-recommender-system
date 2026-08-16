# Train/test ayrimi (collaborative filtering icin)

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"
ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")

# her (userId, movieId, rating) satirini bagimsiz bir ornek gibi ele alip %80/%20 ayiriyoruz
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

print("=== boyutlar ===")
print("train:", train.shape, " test:", test.shape)

# cold start kontrolu: test setinde train'de hic gorulmemis kullanici/film var mi?
train_users = set(train["userId"])
train_movies = set(train["movieId"])

unseen_users = set(test["userId"]) - train_users
unseen_movies = set(test["movieId"]) - train_movies

test_rows_with_unseen_user = test["userId"].isin(unseen_users).sum()
test_rows_with_unseen_movie = test["movieId"].isin(unseen_movies).sum()

print("\n=== cold start kontrolu ===")
print(f"train'de hic gorulmemis kullanici sayisi: {len(unseen_users)}")
print(f"train'de hic gorulmemis film sayisi: {len(unseen_movies)}")
print(f"bu kullanicilardan gelen test satiri: {test_rows_with_unseen_user} / {len(test)}")
print(f"bu filmlerden gelen test satiri: {test_rows_with_unseen_movie} / {len(test)}")

train.to_csv(SCRIPT_DIR / "ratings_train.csv", index=False)
test.to_csv(SCRIPT_DIR / "ratings_test.csv", index=False)
print("\nKaydedildi: ratings_train.csv, ratings_test.csv")
