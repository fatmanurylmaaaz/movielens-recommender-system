# Rating (puan) dağılımını analiz et

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

# Rating sütununun temel istatistiklerini göster
# (ortalama, minimum, maksimum, standart sapma vb.)
print("=== rating istatistikleri ===")
print(ratings["rating"].describe())

# Her rating değerinin kaç kez verildiğini hesapla
# sort_index() ile puanlar küçükten büyüğe sıralanır
print("\n=== rating deger sayimlari ===")
print(ratings["rating"].value_counts().sort_index())

# Grafik oluşturmak için 7x4 boyutunda bir figür oluştur
fig, ax = plt.subplots(figsize=(7, 4))

# Rating dağılımını sütun (bar) grafiği olarak çiz
ratings["rating"].value_counts().sort_index().plot(
    kind="bar",          # Grafik türü
    ax=ax,               # Çizilecek eksen
    color="#4C72B0"      # Sütun rengi
)

# Grafiğin başlığını belirle
ax.set_title("Rating Dagilimi")

# X ekseninin adını belirle
ax.set_xlabel("Rating (yildiz)")

# Y ekseninin adını belirle
ax.set_ylabel("Adet")

# Grafik elemanlarının taşmasını önleyerek düzenli görünmesini sağlar
plt.tight_layout()

# Grafiği PNG formatında scripts klasörüne kaydet
plt.savefig(SCRIPT_DIR / "out_rating_distribution.png", dpi=120)

# Kaydetme işleminin başarılı olduğunu kullanıcıya bildir
print("\nGrafik kaydedildi: scripts/out_rating_distribution.png")