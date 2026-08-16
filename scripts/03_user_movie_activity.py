# Kullanıcı ve film başına puan sayısı dağılımını incele (Sparsity Analizi)

import pandas as pd                  # Veri okuma ve analiz işlemleri için
import matplotlib                    # Grafik oluşturmak için
matplotlib.use("Agg")                # Grafik penceresi açmadan dosyaya kaydetmeyi sağlar
import matplotlib.pyplot as plt      # Grafik çizim fonksiyonları
from pathlib import Path

# Script nerden calistirilirsa calistirilsin dogru yolu bulmak icin __file__ kullanilir
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"

# ratings.csv dosyasını DataFrame olarak oku
ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")

# Her kullanıcının kaç adet film puanladığını hesapla
# groupby("userId") kullanıcıları gruplar,
# size() ise her grubun kaç satır içerdiğini döndürür.
ratings_per_user = ratings.groupby("userId").size()

# Her filmin kaç kez puanlandığını hesapla
# Böylece filmlerin popülerlik dağılımı elde edilir.
ratings_per_movie = ratings.groupby("movieId").size()

# Kullanıcı başına verilen puan sayılarının istatistiklerini göster
# (ortalama, minimum, maksimum, standart sapma vb.)
print("=== kullanici basina puan sayisi ===")
print(ratings_per_user.describe())

# Film başına verilen puan sayılarının istatistiklerini göster
print("\n=== film basina puan sayisi ===")
print(ratings_per_movie.describe())

# En az film puanlayan 5 kullanıcıyı göster
print(
    "\nEn az puanlanan 5 kullanici:\n",
    ratings_per_user.sort_values().head()
)

# En fazla film puanlayan 5 kullanıcıyı göster
print(
    "\nEn cok puanlayan 5 kullanici:\n",
    ratings_per_user.sort_values(ascending=False).head()
)

# Sadece bir kez puanlanmış film sayısını hesapla
# Bu değer, veri setindeki az bilinen filmleri gösterir.
print(
    "\nSadece 1 kez puanlanmis film sayisi:",
    (ratings_per_movie == 1).sum(),
    "/",
    len(ratings_per_movie)
)

# -----------------------------
# Sparsity (Seyreklik) Hesabı
# -----------------------------

# Benzersiz kullanıcı sayısı
n_users = ratings["userId"].nunique()

# Benzersiz film sayısı
n_movies = ratings["movieId"].nunique()

# Toplam verilen puan sayısı
n_ratings = len(ratings)

# Eğer her kullanıcı her filmi puanlasaydı oluşacak
# toplam kullanıcı-film eşleşmesi
possible = n_users * n_movies

# Sparsity (Boşluk Oranı) Formülü
# 1 - (Mevcut Rating Sayısı / Olası Rating Sayısı)
sparsity = 1 - (n_ratings / possible)

# Sparsity sonuçlarını ekrana yazdır
print(f"\n=== sparsity ===")
print(f"benzersiz kullanici: {n_users}, benzersiz film: {n_movies}")
print(f"olasi hucre sayisi: {possible}, dolu hucre sayisi: {n_ratings}")
print(f"matrisin bosluk orani (sparsity): %{sparsity*100:.2f}")

# ---------------------------------
# Grafiklerin Oluşturulması
# ---------------------------------

# Yan yana iki histogram oluştur
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# -----------------------------
# Sol Histogram
# -----------------------------
# Her kullanıcının kaç film puanladığını gösterir.
axes[0].hist(
    ratings_per_user,
    bins=50,
    color="#4C72B0"
)

axes[0].set_title("Kullanici basina puan sayisi")
axes[0].set_xlabel("Puan sayisi")
axes[0].set_ylabel("Kullanici adedi")

# -----------------------------
# Sağ Histogram
# -----------------------------
# Her filmin kaç kullanıcı tarafından puanlandığını gösterir.
axes[1].hist(
    ratings_per_movie,
    bins=50,
    color="#DD8452"
)

axes[1].set_title("Film basina puan sayisi")
axes[1].set_xlabel("Puan sayisi")
axes[1].set_ylabel("Film adedi")

# Grafiklerin düzenli görünmesini sağlar
plt.tight_layout()

# Grafiği PNG olarak kaydet
plt.savefig(SCRIPT_DIR / "out_user_movie_activity.png", dpi=120)

# Kaydetme işleminin başarılı olduğunu bildir
print("\nGrafik kaydedildi: scripts/out_user_movie_activity.png")