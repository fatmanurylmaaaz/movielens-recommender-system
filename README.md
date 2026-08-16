# Film Öneri Sistemi — MovieLens (ml-latest-small)

MovieLens `ml-latest-small` veri seti (610 kullanıcı, 9.742 film, 100.836 puanlama, 3.683 etiket) üzerine kurulmuş, üç farklı yaklaşımı (content-based, collaborative filtering, hybrid) içeren bir film öneri sistemi projesi. Proje, model geliştirme sürecini adım adım öğrenmek amacıyla EDA'dan başlayıp temiz, yeniden kullanılabilir bir Python paketiyle sonuçlanacak şekilde ilerledi.

## Klasör yapısı

```
movielens-recommender-system/
    dataset/ml-latest-small/     # ham MovieLens verisi (ratings, movies, tags, links, README.txt)
    scripts/                     # ogrenme sureci: 26 numarali script, EDA'dan hybrid modele
    movie_recommender/           # paketlenmis, yeniden kullanilabilir surum (bkz. kendi README'si)
    egitim/                      # ikinci ogrenme turu: SVD, MF, NCF varyasyonlari + birlesik degerlendirme
    streamlit_app.py             # movie_recommender'i web arayuzunde gorsellestiren Streamlit uygulamasi
    item_based_collaborative_filtering.ipynb   # kNN + cosine similarity ile item-based CF denemesi
```

## `scripts/` — Öğrenme Süreci

Script'ler numaralandırılmış sırayla çalıştırılacak şekilde tasarlandı, her biri bir önceki adımın çıktısını kullanıyor:

| Aşama | Script'ler | Ne yapıyor |
|---|---|---|
| **EDA** | `01`–`06` | Veri yapısı, rating dağılımı, sparsity (%98.3), genre dağılımı, popülerlik tuzağı, tag analizi |
| **Ön işleme** | `07`–`09` | Genre/tag temizliği, `content_text` ("content soup") oluşturma |
| **Content-based model** | `10`–`12` | TF-IDF vektörleştirme, cosine similarity, `recommend()` fonksiyonu |
| **Collaborative filtering (SVD)** | `13`–`18` | User-item matrisi, train/test ayrımı, mean-centered SVD, RMSE/MAE değerlendirme |
| **Model kıyaslama** | `19` | Content-based ve CF'nin ürettiği önerileri karşılaştırma |
| **Hybrid model** | `20`–`23` | Kullanıcı profil vektörü, skor normalizasyonu, ağırlıklı birleşim, 3 modelin kıyası |
| **Hiperparametre ayarı** | `24`, `26` | SVD'nin `k` değeri ve hybrid'in `alpha` değeri için tarama |
| **Nicel değerlendirme** | `25` | Precision@K / Recall@K ile üç modelin sıralama kalitesi |

### Öne çıkan bulgular

- **Sparsity %98.3** — kullanıcı-film matrisinin neredeyse tamamı boş, bu yüzden basit komşuluk yöntemleri yerine matrix factorization (SVD) tercih edildi.
- **En iyi SVD `k=10`** (RMSE 0.9184) — daha yüksek `k` değerleri (30, 50, 100) overfitting nedeniyle daha kötü sonuç verdi.
- **CF, content-based'den Precision@10'da ~11 kat daha güçlü** (0.1260 vs 0.0114) — ama content-based'in küçük bir katkısı (`alpha=0.9`) saf CF'yi bile hafifçe geçti.
- **Cold-start kanıtlandı:** train setinde hiç olmayan bir film için SVD tahmini, matematiksel olarak tam baseline'a (kullanıcı ortalamasına) eşitleniyor.

## `movie_recommender/` — Paketlenmiş Sürüm

Öğrenme sürecinin sonunda çıkan tüm mantık, tekrar kullanılabilir bir Python paketine taşındı (`scripts/` klasörüne dokunulmadan, ayrı ve bağımsız). Detaylar için [movie_recommender/README.md](movie_recommender/README.md).

```bash
python -m movie_recommender.recommend --model hybrid --user-id 1 --k 5
```

```python
from movie_recommender import MovieRecommender
model = MovieRecommender().fit()
model.recommend_hybrid(user_id=1, k=10)
```

## `eğitim/` — İkinci Öğrenme Turu (Model Varyasyonları)

`movie_recommender` paketlendikten sonra, collaborative filtering'in farklı varyasyonlarını (klasik SVD dışında) öğrenmek için ayrı, bağımsız script'lerden oluşan ikinci bir tur:

| Script | Ne yapıyor |
|---|---|
| `popularity_based.py` | Non-personalized: en çok oy alan / IMDB ağırlıklı puana göre en iyi filmler |
| `content_based.py` | Genre + tag TF-IDF ve cosine similarity |
| `collaborative_filtering.py` | Item-based CF (kNN + cosine similarity) |
| `svd.py` | Klasik SVD (`scipy.svds`, tek seferlik lineer cebir çözümü) |
| `mf.py` | Matrix Factorization — embedding + bias, **gradyan inişiyle** (PyTorch, Adam) eğitilen Funk-MF |
| `ncf.py` | Neural Collaborative Filtering — embedding + MLP, gradyan inişiyle eğitilen derin model |
| `hybrid.py` | Content-based + item-based CF'nin `alpha` ağırlıklı birleşimi |
| `evaluate.py` | Tüm modelleri (content, item-based CF, SVD, MF, NCF, hybrid) aynı train/test split ve kullanıcı örneklemiyle **tek tabloda** kıyaslıyor |

Eğitilen modeller `eğitim/cache/` altında önbelleklenir (`svd_model.pkl`, `mf_model.pt`, `ncf_model.pt`).

### Öne çıkan bulgular

- **RMSE sıralaması:** NCF (0.8940) < MF (0.9012) < SVD (0.9159) < Item-Based CF (0.9288) — gradyan inişiyle eğitilen modeller (MF, NCF) klasik SVD'yi puan tahmininde geçti.
- **Ama Precision@10'da tam tersi bir tablo:** SVD (0.1340) ve MF (0.1144), NCF'yi (0.0732) ve content-based/item-based CF/hybrid'i (hepsi <0.03) açık farkla geçti. **En düşük RMSE = en iyi öneri kalitesi anlamına gelmiyor** — SVD/MF'nin ürettiği yoğun (dense), tutarlı global yapı sıralama kalitesinde daha güçlü çıktı.
- **Az oy alan filmler tuzağı:** Item-based CF, NCF ve (daha hafif) SVD'de, çok az puanlanan filmler gürültülü/aşırı skorlarla önerileri domine edebiliyor — `MIN_RATINGS` eşiğiyle filtrelendi.
- **MF'de overfitting:** İlk denemede training loss 0.12'ye kadar düşse de test RMSE 1.0'ı geçti (ezberleme); `weight_decay` ile düzeltildi.

## `streamlit_app.py` — Web Arayüzü

`movie_recommender` paketini interaktif bir web arayüzünde görselleştiren Streamlit uygulaması:

```bash
streamlit run streamlit_app.py
```

4 sekme: **İçerik Bazlı** (film seçip benzerlerini görme), **Collaborative Filtering** (kullanıcıya özel SVD tahminleri), **Hybrid** (alpha kaydırıcısıyla canlı ayarlanabilir birleşim), **Model Performansı** (RMSE/MAE). Model `@st.cache_resource` ile önbelleklenir, sadece ilk açılışta eğitilir/yüklenir.

## `item_based_collaborative_filtering.ipynb`

Item-based kNN + cosine similarity ile yazılmış ayrı bir collaborative filtering denemesi (scipy sparse matris + scikit-learn `NearestNeighbors`). `scripts/13-18`'deki SVD yaklaşımından farklı, daha basit bir CF yöntemi örneği olarak duruyor.

## Gereksinimler

`pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib` (Python 3.8 ile test edildi). `eğitim/mf.py` ve `eğitim/ncf.py` için ayrıca `torch`; `streamlit_app.py` için `streamlit` ve `plotly`.
