# TF-IDF vektorlestirme

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

SCRIPT_DIR = Path(__file__).resolve().parent
movies = pd.read_csv(SCRIPT_DIR / "movies_processed.csv")
movies["content_text"] = movies["content_text"].fillna("")

# TfidfVectorizer: metni once kelimelere ayirir (tokenize), sonra her kelimenin
# TF-IDF agirligini hesaplar. min_df=2 -> sadece 1 filmde gecen (essiz/nadir yazim
# hatasi gibi) kelimeleri elemek icin en az 2 filmde gecmesini sart kosuyoruz.
vectorizer = TfidfVectorizer(min_df=2)
tfidf_matrix = vectorizer.fit_transform(movies["content_text"])

print("=== TF-IDF matris boyutu ===")
print(f"{tfidf_matrix.shape[0]} film x {tfidf_matrix.shape[1]} benzersiz kelime (token)")

feature_names = vectorizer.get_feature_names_out()
print("\n=== ornek token'lar (ilk 20) ===")
print(list(feature_names[:20]))

# bir filmin vektorunde en yuksek agirlikli kelimelere bakalim (Toy Story ornegi)
toy_story_idx = movies[movies["clean_title"] == "Toy Story"].index[0]
row = tfidf_matrix[toy_story_idx].toarray().flatten()
top_indices = row.argsort()[::-1][:10]

print("\n=== 'Toy Story' icin en yuksek agirlikli 10 kelime ===")
for i in top_indices:
    if row[i] > 0:
        print(f"{feature_names[i]}: {row[i]:.3f}")

import scipy.sparse as sp
sp.save_npz(SCRIPT_DIR / "tfidf_matrix.npz", tfidf_matrix)
movies.to_csv(SCRIPT_DIR / "movies_processed.csv", index=False)
print("\nTF-IDF matrisi kaydedildi: scripts/tfidf_matrix.npz")
