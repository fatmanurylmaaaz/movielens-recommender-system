# Cosine similarity ile benzer film bulma (kavram dogrulama)

import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = Path(__file__).resolve().parent
movies = pd.read_csv(SCRIPT_DIR / "movies_processed.csv")
tfidf_matrix = sp.load_npz(SCRIPT_DIR / "tfidf_matrix.npz")

# Ornek: "Toy Story" filmine en benzer filmleri bulalim
query_title = "Toy Story"
query_idx = movies[movies["clean_title"] == query_title].index[0]

# Tum 9742x9742 matrisi hesaplamak yerine, sadece bu 1 filmin diger tum filmlerle
# olan benzerligini hesapliyoruz -> (1, 9742) boyutunda bir satir, cok daha hafif.
sim_scores = cosine_similarity(tfidf_matrix[query_idx], tfidf_matrix).flatten()

print(f"=== '{query_title}' filmine en benzer 10 film ===")
top_indices = sim_scores.argsort()[::-1]
count = 0
for idx in top_indices:
    if idx == query_idx:
        continue  # filmin kendisini atla (benzerligi her zaman 1.0 olur)
    print(f"{movies.loc[idx, 'clean_title']}: {sim_scores[idx]:.3f}")
    count += 1
    if count == 10:
        break

# TfidfVectorizer varsayilan olarak vektorleri L2-normalize eder (uzunlugu 1 yapar).
# Bu durumda cosine similarity = basit ic carpim (dot product) ile ayni sonucu verir.
# Bunu kanitlayalim:
import numpy as np
dot_product = tfidf_matrix[query_idx].dot(tfidf_matrix.T).toarray().flatten()
print("\n=== dogrulama: cosine_similarity ile dot product ayni mi? ===")
print("max fark:", np.abs(sim_scores - dot_product).max())
