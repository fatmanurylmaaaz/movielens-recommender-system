# Film türü (Genre) dağılımını analiz et

import pandas as pd                  # Veri okuma ve analiz işlemleri için
import matplotlib                    # Grafik oluşturmak için
matplotlib.use("Agg")                # Grafik penceresi açmadan dosyaya kaydetmeyi sağlar
import matplotlib.pyplot as plt      # Grafik çizim fonksiyonları
from pathlib import Path

# Script nerden calistirilirsa calistirilsin dogru yolu bulmak icin __file__ kullanilir
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"

# movies.csv dosyasını DataFrame olarak oku
movies = pd.read_csv(f"{DATA_DIR}/movies.csv")

# İlk 5 filmin adı ve tür bilgisini göster
# Verinin doğru okunup okunmadığını kontrol etmek için kullanılır.
print("=== genre orneği (ilk 5 film) ===")
print(movies[["title", "genres"]].head())

# ---------------------------------------------------
# Genre sütununu analiz edilebilir hale getir
# ---------------------------------------------------

# genres sütunu şu şekilde tutulmaktadır:
# Adventure|Animation|Comedy
# Drama|Romance
#
# Önce split("|") ile türleri liste haline getiriyoruz.
# Daha sonra explode() ile listedeki her türü ayrı satıra dönüştürüyoruz.
# Böylece her tür tek tek sayılabilir hale gelir.
genre_series = movies["genres"].str.split("|").explode()

# Her film türünün kaç filmde bulunduğunu hesapla
print("\n=== genre basina film sayisi ===")
genre_counts = genre_series.value_counts()
print(genre_counts)

# Tür bilgisi bulunmayan film sayısını hesapla
# "(no genres listed)" ifadesi türü belirtilmeyen filmleri temsil eder.
print("\n=== genre'siz film sayisi ===")
print((movies["genres"] == "(no genres listed)").sum())

# Her film için kaç farklı tür bulunduğunu hesapla
# split("|") ile türleri ayırır,
# apply(len) ile her filmde kaç tür olduğunu sayar.
# describe() ise bu sayıların istatistiklerini verir.
print("\n=== film basina ortalama genre sayisi ===")
print(movies["genres"].str.split("|").apply(len).describe())

# ---------------------------------
# Grafik Oluşturma
# ---------------------------------

# 8x5 boyutunda grafik oluştur
fig, ax = plt.subplots(figsize=(8, 5))

# Tür dağılımını yatay çubuk grafik (Horizontal Bar Chart) olarak çiz
genre_counts.plot(
    kind="barh",
    ax=ax,
    color="#55A868"
)

# En fazla filme sahip türlerin üst tarafta görünmesini sağlar
ax.invert_yaxis()

# Grafik başlığı
ax.set_title("Genre basina film sayisi")

# X ekseni adı
ax.set_xlabel("Film adedi")

# Grafik elemanlarının düzgün yerleşmesini sağlar
plt.tight_layout()

# Grafiği PNG olarak kaydet
plt.savefig(SCRIPT_DIR / "out_genre_distribution.png", dpi=120)

# Kaydetme işleminin başarılı olduğunu bildir
print("\nGrafik kaydedildi: scripts/out_genre_distribution.png")