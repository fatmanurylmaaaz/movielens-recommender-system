# Genre + tag'leri birlestirip icerik metni (content soup) olusturma

import pandas as pd
from pathlib import Path

# Script nerden calistirilirsa calistirilsin dogru yolu bulmak icin __file__ kullanilir
SCRIPT_DIR = Path(__file__).resolve().parent
movies = pd.read_csv(SCRIPT_DIR / "movies_processed.csv")
movies["genre_text"] = movies["genre_text"].fillna("")
movies["tag_text"] = movies["tag_text"].fillna("")

# genre_text'i 2 kez tekrarlayarak biraz daha fazla agirlik veriyoruz
# (genre her filmde guvenilir bilgi, tag'ler sadece bazi filmlerde var)
movies["content_text"] = (
    movies["genre_text"] + " " + movies["genre_text"] + " " + movies["tag_text"]
).str.strip()

print("=== content_text ornekleri ===")
print(movies[["clean_title", "genre_text", "tag_text", "content_text"]].head(5).to_string(index=False))

print("\n=== bos content_text (genre'siz VE tag'siz) film sayisi ===")
empty_mask = movies["content_text"] == ""
print(empty_mask.sum())
print(movies[empty_mask][["clean_title"]].head(10).to_string(index=False))

movies.to_csv(SCRIPT_DIR / "movies_processed.csv", index=False)
print("\nGuncellenmis veri kaydedildi: scripts/movies_processed.csv")
