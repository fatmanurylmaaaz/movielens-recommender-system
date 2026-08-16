# movie_recommender

`scripts/` klasöründeki 26 adımlık öğrenme sürecinin (EDA, ön işleme, content-based, collaborative filtering, hybrid) paketlenmiş, yeniden kullanılabilir hali. `scripts/` klasörüne dokunulmadı, bu paket ayrı ve bağımsız.

## Klasör yapısı

```
movie_recommender/
    __init__.py          # from .pipeline import MovieRecommender
    config.py             # DATA_DIR, MIN_RATINGS, SVD_K=10, HYBRID_ALPHA=0.9 gibi tum sabitler
    data.py                # CSV okuma fonksiyonlari
    preprocessing.py       # genre/tag temizligi + content_text (scripts/07-09)
    content_based.py       # TF-IDF, find_movie, recommend_similar (scripts/10-12)
    collaborative.py       # SVD egitim/tahmin (scripts/13-18)
    hybrid.py               # user profile + agirlikli birlesim (scripts/20-22)
    evaluate.py             # rmse, mae, precision_recall_at_k (scripts/17, 25-26)
    pipeline.py             # MovieRecommender sinifi (hepsini birlestiren orkestratork)
    recommend.py            # CLI giris noktasi
    cache/model.pkl         # egitilmis modelin onbellegi (ilk fit() sonrasi olusur)
```

## Kullanım

Komut satırından (proje kökünden çalıştırılmalı):

```bash
python -m movie_recommender.recommend --model content --movie-title "Toy Story"
python -m movie_recommender.recommend --model cf --user-id 1
python -m movie_recommender.recommend --model hybrid --user-id 1 --k 5
```

Python içinden:

```python
from movie_recommender import MovieRecommender

model = MovieRecommender().fit()   # ilk calistirmada egitir, sonrakilerde cache'den okur
model.recommend_content("Toy Story", k=10)
model.recommend_cf(user_id=1, k=10)
model.recommend_hybrid(user_id=1, k=10, alpha=0.9)
model.evaluate_cf()                 # (rmse, mae) dondurur
```

Cache'i atlayıp modeli sıfırdan eğitmek için `fit(use_cache=False)` ya da CLI'da `--no-cache`.

## Doğrulama

Paket, `scripts/` klasöründeki orijinal script'lerle birebir aynı sonuçları üretiyor:
- Content-based: "Toy Story" için en benzer film `A Bug's Life` (benzerlik: 0.841) — `11_cosine_similarity.py` ile aynı.
- CF: RMSE 0.9184 — `24_tune_svd_k.py`'de bulunan en iyi `k=10` sonucuyla birebir aynı.

Yani paketleme davranışı değiştirmedi, sadece kodu (importlib hack'leri, tekrarlanan fonksiyonlar, dağınık script'ler yerine) düzenli modüllere ve tek bir `MovieRecommender` sınıfına taşıdı.
