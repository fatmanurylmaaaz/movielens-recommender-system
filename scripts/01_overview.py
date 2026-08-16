
#Genel veri yapısını incele (shape, dtypes, eksik değerler)

import pandas as pd  # Veri analizi ve CSV dosyalarını okumak için pandas kütüphanesi
from pathlib import Path

# Script nerden calistirilirsa calistirilsin dogru yolu bulmak icin __file__ kullanilir
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"

# CSV dosyalarını DataFrame olarak belleğe yükle
ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")  # Kullanıcıların film puanları
movies = pd.read_csv(f"{DATA_DIR}/movies.csv")    # Film bilgileri (isim, tür vb.)
tags = pd.read_csv(f"{DATA_DIR}/tags.csv")        # Kullanıcıların filmlere eklediği etiketler
links = pd.read_csv(f"{DATA_DIR}/links.csv")      # MovieLens ile IMDb ve TMDb bağlantıları

# Tüm DataFrame'leri sırayla incelemek için liste oluşturuyoruz
for name, df in [("ratings", ratings),
                 ("movies", movies),
                 ("tags", tags),
                 ("links", links)]:

    # İncelenen tablonun adını yazdır
    print(f"\n=== {name} ===")

    # Veri setinin boyutunu gösterir
    # (satır sayısı, sütun sayısı)
    print("shape:", df.shape)

    # Her sütunun veri tipini gösterir
    # int64, float64, object vb.
    print(df.dtypes)

    # Her sütunda kaç tane eksik (NaN) değer olduğunu hesaplar
    print("eksik deger sayisi:\n", df.isna().sum())

# ratings veri setinin ilk 5 satırını gösterir
# Verinin doğru okunup okunmadığını hızlıca kontrol etmek için kullanılır
print("\n=== ratings head ===")
print(ratings.head())

# Tekrarlanan (duplicate) kayıtları kontrol ediyoruz
print("\n=== duplicate satir kontrolu ===")

# ratings veri setinde tamamen aynı olan satırların sayısını verir
print("ratings duplicate:", ratings.duplicated().sum())

# movies veri setinde movieId sütununda tekrar eden film ID'si var mı kontrol eder
# Sonucun 0 olması beklenir çünkü her filmin benzersiz bir ID'si olmalıdır.
print("movies duplicate movieId:", movies["movieId"].duplicated().sum())
