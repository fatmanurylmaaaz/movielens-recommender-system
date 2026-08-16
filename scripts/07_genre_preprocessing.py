# Genre verisini on isleme (temizlik + format donusumu)

import pandas as pd
from pathlib import Path

# Script nerden calistirilirsa calistirilsin dogru yolu bulmak icin __file__ kullanilir
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"
movies = pd.read_csv(f"{DATA_DIR}/movies.csv")

print("=== islenmeden once ===")
print(movies[["title", "genres"]].head())

# "(no genres listed)" -> bos liste, digerleri "|" ile ayrilip listeye cevriliyor
def parse_genres(genre_str):
    if genre_str == "(no genres listed)":
        return []
    return genre_str.split("|")

movies["genre_list"] = movies["genres"].apply(parse_genres)

# TF-IDF'in "Sci-Fi" / "Film-Noir" gibi birlesik turleri ikiye bolmemesi icin
# tirenleri kaldirip tek kelime haline getiriyoruz, kelimeleri bosluk ile ayirip
# tek bir metin (string) haline getiriyoruz -> vektorlestirme icin gerekli format
# birleşik tür isimlerini tek kelime haline getirdik.
movies["genre_text"] = movies["genre_list"].apply(
    lambda genres: " ".join(g.replace("-", "").replace(" ", "").lower() for g in genres)
)

# title icindeki yili ayri bir sutuna cikar, temiz basligi ayri tut
# yıl bilgisi ayrı bir year sütununa çıkarıldı.
# orn. "Toy Story (1995)" -> clean_title="Toy Story", year=1995
extracted = movies["title"].str.extract(r"^(.*)\s\((\d{4})\)\s*$")
# yil bulunamayan basliklarda (orn. "Babylon 5") clean_title NaN kalmasin,
# orijinal title'a dussun -> arama fonksiyonlari NaN ile karsilasmasin
movies["clean_title"] = extracted[0].fillna(movies["title"])
movies["year"] = pd.to_numeric(extracted[1])

print("\n=== islendikten sonra ===")
print(movies[["clean_title", "year", "genre_list", "genre_text"]].head())

print("\n=== yil cikarilamayan (format disi) film sayisi ===")
print(movies["year"].isna().sum())

print("\n=== benzersiz genre_text token ornekleri ===")
all_tokens = set(" ".join(movies["genre_text"]).split())
print(sorted(all_tokens))

print("\n=== genre'siz film sayisi (genre_text bos) ===")
print((movies["genre_text"] == "").sum())

movies.to_csv(SCRIPT_DIR / "movies_processed.csv", index=False)
print("\nIslenmis veri kaydedildi: scripts/movies_processed.csv")
