# Tag verisini normalize edip film bazinda birlestirme

import pandas as pd
from pathlib import Path

# Script nerden calistirilirsa calistirilsin dogru yolu bulmak icin __file__ kullanilir
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"
tags = pd.read_csv(f"{DATA_DIR}/tags.csv")
movies = pd.read_csv(SCRIPT_DIR / "movies_processed.csv")

print("=== tag on isleme oncesi ornekler ===")
print(tags["tag"].head(10).tolist())

# normalize: kucuk harf + bas/son bosluk temizligi (ayni etiketin farkli yazimlarini birlestirmek icin)
tags["tag_clean"] = tags["tag"].str.lower().str.strip()

# icerikle ilgisiz, kullanicinin kisisel organizasyonuna dair "gurultu" etiketlerini filtrele
NOISE_TAGS = {"in netflix queue"}
before = len(tags)
tags = tags[~tags["tag_clean"].isin(NOISE_TAGS)]
print(f"\nfiltrelenen gurultu etiket satiri: {before - len(tags)}")

# ayni filme birden fazla kullanici / birden fazla etiket eklenmis olabilir
# hepsini tek bir metinde birlestiriyoruz -> film basina 1 satir
movie_tags = tags.groupby("movieId")["tag_clean"].apply(lambda s: " ".join(s)).reset_index()
movie_tags.columns = ["movieId", "tag_text"]

print("\n=== film basina birlesmis tag ornegi ===")
print(movie_tags.head())

# script'i birden fazla kez calistirinca ayni dosyaya tekrar tekrar merge yapmamak icin
# (yoksa tag_text sutunu zaten varsa pandas tag_text_x/tag_text_y diye ikiye ayirir)
movies = movies.drop(columns=["tag_text"], errors="ignore")

# movies tablosuna sol birlestirme (join) ile ekle; tag'i olmayan filmlerde bos string kalsin
movies = movies.merge(movie_tags, on="movieId", how="left")
movies["tag_text"] = movies["tag_text"].fillna("")

print("\n=== tag_text eklenmis movies orneği (tag'i olan ilk 5 film) ===")
print(movies[movies["tag_text"] != ""][["clean_title", "genre_text", "tag_text"]].head())

print("\ntag_text dolu film sayisi:", (movies["tag_text"] != "").sum(), "/", len(movies))

movies.to_csv(SCRIPT_DIR / "movies_processed.csv", index=False)
print("\nGuncellenmis veri kaydedildi: scripts/movies_processed.csv")
