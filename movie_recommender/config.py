"""Proje genelinde kullanilan sabitler ve yollar."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "ml-latest-small"
CACHE_DIR = PACKAGE_DIR / "cache"

# EDA/tuning script'lerinde (scripts/) bulunan degerler
MIN_RATINGS = 5          # 05 ve 22 numarali scriptlerde belirlenen "az bilinen film" esigi
SVD_K = 10                # 24_tune_svd_k.py ile RMSE'ye gore secilen en iyi deger
HYBRID_ALPHA = 0.9        # 26_tune_alpha.py ile Precision@10'a gore secilen en iyi deger
TFIDF_MIN_DF = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42
RELEVANCE_THRESHOLD = 4.0
NOISE_TAGS = {"in netflix queue"}
