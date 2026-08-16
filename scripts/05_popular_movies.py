# En popüler ve en yüksek puanlı filmleri analiz et

import pandas as pd                  # Veri okuma ve analiz işlemleri için
import matplotlib                    # Grafik oluşturmak için
matplotlib.use("Agg")                # Grafik penceresi açmadan dosyaya kaydetmeyi sağlar
import matplotlib.pyplot as plt      # Grafik çizim fonksiyonları
from pathlib import Path

# Script nerden calistirilirsa calistirilsin dogru yolu bulmak icin __file__ kullanilir
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"

# Gerekli veri setlerini oku
ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")   # Kullanıcı puanları
movies = pd.read_csv(f"{DATA_DIR}/movies.csv")     # Film bilgileri

# -------------------------------------------------
# Film bazında istatistikleri hesapla
# -------------------------------------------------

# movieId'ye göre grupla ve her film için:
# - Kaç kez puanlandığını (rating_count)
# - Ortalama puanını (rating_mean)
# hesapla.
stats = ratings.groupby("movieId").agg(
    rating_count=("rating", "count"),
    rating_mean=("rating", "mean"),
)

# Film isimlerini eklemek için movies veri seti ile birleştir
stats = stats.merge(
    movies[["movieId", "title"]],
    on="movieId"
)

# -------------------------------------------------
# En popüler filmler
# -------------------------------------------------

# En çok puan alan 10 filmi göster
# Burada popülerlik, filmin aldığı puan sayısı ile ölçülmektedir.
print("=== en cok puanlanan (en populer) 10 film ===")

print(
    stats.sort_values("rating_count", ascending=False)
    [["title", "rating_count", "rating_mean"]]
    .head(10)
    .to_string(index=False)
)

# -------------------------------------------------
# En yüksek ortalama puanlı filmler (Filtre Yok)
# -------------------------------------------------

# Sadece ortalama puana göre sıralama yapılır.
# Ancak yalnızca 1-2 kişinin oy verdiği filmler de üst sıralarda çıkabilir.
# Bu nedenle bu yöntem güvenilir değildir (Naif yaklaşım).
print("\n=== NAIF en yuksek ortalama puanli 10 film (puan sayisi filtresi YOK) ===")

print(
    stats.sort_values("rating_mean", ascending=False)
    [["title", "rating_count", "rating_mean"]]
    .head(10)
    .to_string(index=False)
)

# -------------------------------------------------
# Minimum puan sayısı filtresi
# -------------------------------------------------

# Bir filmin değerlendirmeye alınabilmesi için
# en az 50 puan almış olması şartı belirleniyor.
# Böylece yalnızca 1-2 kişinin yüksek puan verdiği filmler
# üst sıralara çıkmaz ve ortalama puanlar daha güvenilir olur.
MIN_RATINGS = 50

print(
    f"\n=== en yuksek ortalama puanli 10 film (en az {MIN_RATINGS} puan almis olma sarti ile) ==="
)

# En az 50 puan alan filmleri filtrele
filtered = stats[stats["rating_count"] >= MIN_RATINGS]

# Filtrelenmiş filmleri ortalama puana göre sırala
print(
    filtered.sort_values("rating_mean", ascending=False)
    [["title", "rating_count", "rating_mean"]]
    .head(10)
    .to_string(index=False)
)

# -------------------------------------------------
# Grafik Oluşturma
# -------------------------------------------------

# 7x5 boyutunda grafik oluştur
fig, ax = plt.subplots(figsize=(7, 5))

# Her filmi bir nokta olarak göster
# X ekseni = aldığı puan sayısı
# Y ekseni = ortalama puanı
ax.scatter(
    stats["rating_count"],
    stats["rating_mean"],
    alpha=0.3,          # Noktaların saydamlığı
    s=15,               # Nokta boyutu
    color="#4C72B0"
)

# Minimum puan sayısını gösteren dikey çizgi ekle
ax.axvline(
    MIN_RATINGS,
    color="#C44E52",
    linestyle="--",
    label=f"{MIN_RATINGS} puan esigi"
)

# X eksenini logaritmik ölçeğe çevir
# Böylece çok fazla puan alan filmler grafiği bozmaz.
ax.set_xscale("log")

# Grafik başlığı
ax.set_title("Puan sayisi (log) vs ortalama puan")

# Eksen isimleri
ax.set_xlabel("Puan sayisi (log olcek)")
ax.set_ylabel("Ortalama puan")

# Açıklama kutusunu göster
ax.legend()

# Grafik düzenini iyileştir
plt.tight_layout()

# Grafiği PNG olarak kaydet
plt.savefig(SCRIPT_DIR / "out_popular_movies.png", dpi=120)

# Kaydetme işleminin başarılı olduğunu bildir
print("\nGrafik kaydedildi: scripts/out_popular_movies.png")