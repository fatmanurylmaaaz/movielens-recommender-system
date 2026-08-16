# Tags (etiket) verisini analiz et

import pandas as pd   # Veri okuma ve analiz işlemleri için
from pathlib import Path

# Script nerden calistirilirsa calistirilsin dogru yolu bulmak icin __file__ kullanilir
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"

# Gerekli veri setlerini oku
tags = pd.read_csv(f"{DATA_DIR}/tags.csv")       # Kullanıcıların filmlere eklediği etiketler
movies = pd.read_csv(f"{DATA_DIR}/movies.csv")   # Film bilgileri

# ---------------------------------
# Tags veri setine genel bakış
# ---------------------------------

# Tags veri setinin ilk 5 satırını göster
# Verinin doğru okunup okunmadığını kontrol etmek için kullanılır.
print("=== tags orneği ===")
print(tags.head())

# ---------------------------------
# En sık kullanılan etiketler
# ---------------------------------

# Ham (orijinal) haliyle en çok kullanılan ilk 15 etiketi göster
# Büyük/küçük harf farklılıkları dikkate alınır.
print("\n=== en cok kullanilan 15 tag (ham, kucuk/buyuk harf duyarli) ===")
print(tags["tag"].value_counts().head(15))

# ---------------------------------
# Tag'leri normalize et
# ---------------------------------

# Aynı anlamdaki etiketlerin farklı yazımlarını birleştirmek için:
# - str.lower() -> tüm harfleri küçültür.
# - str.strip() -> baştaki ve sondaki boşlukları kaldırır.
tags_norm = tags["tag"].str.lower().str.strip()

# Normalizasyon öncesi ve sonrası benzersiz tag sayılarını göster
# Böylece yazım farklılıklarından kaynaklanan tekrarlar görülebilir.
print("\n=== normalize edilince benzersiz tag sayisi ===")
print("ham benzersiz tag:", tags["tag"].nunique())
print("normalize benzersiz tag:", tags_norm.nunique())

# Normalize edilmiş etiketler arasında en sık kullanılan ilk 15 etiketi göster
print("\n=== en cok kullanilan 15 tag (normalize edilmis) ===")
print(tags_norm.value_counts().head(15))

# ---------------------------------
# Tag kapsamını hesapla
# ---------------------------------

# Veri setindeki toplam benzersiz film sayısı
n_movies_total = movies["movieId"].nunique()

# En az bir etikete sahip film sayısı
n_movies_tagged = tags["movieId"].nunique()

# Tag eklenmiş film oranını hesapla
print(f"\n=== tag kapsami ===")
print(
    f"toplam film: {n_movies_total}, "
    f"en az 1 tag'i olan film: {n_movies_tagged} "
    f"(%{n_movies_tagged/n_movies_total*100:.1f})"
)

# ---------------------------------
# Kullanıcı başına tag sayısı
# ---------------------------------

# Her kullanıcının kaç adet tag eklediğini hesapla
# describe() ile ortalama, minimum, maksimum vb. istatistikleri göster.
print("\n=== kullanici basina tag sayisi ===")
print(tags.groupby("userId").size().describe())

# ---------------------------------
# Film başına tag sayısı
# ---------------------------------

# Sadece en az bir tag eklenmiş filmleri dikkate alarak
# film başına kaç tag bulunduğunu hesapla.
print("\n=== film basina tag sayisi (sadece tag'i olanlar) ===")
print(tags.groupby("movieId").size().describe())