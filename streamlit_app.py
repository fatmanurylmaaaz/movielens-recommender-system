"""
movie_recommender paketini web arayüzünde görselleştiren Streamlit uygulaması.
Çalıştırmak için proje kökünden: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# movie_recommender paketi nereden calistirilirsa calistirilsin bulunabilsin diye path'e eklenir
sys.path.insert(0, str(Path(__file__).resolve().parent))
from movie_recommender import MovieRecommender

st.set_page_config(page_title="Film Öneri Sistemi", page_icon="🎬", layout="wide")

BAR_COLOR = "#2a78d6"  # tek seri -> sequential mavi, legend gerekmez


# @st.cache_resource: sayfa her widget etkilesiminde bastan calisir,
# bu dekorator olmadan model her tiklamada yeniden egitilir/yuklenirdi
@st.cache_resource(show_spinner="Model eğitiliyor / önbellekten yükleniyor...")
def load_model():
    return MovieRecommender().fit()


def horizontal_bar(df: pd.DataFrame, x_col: str, y_col: str, x_title: str):
    # content/cf/hybrid sekmelerinin ucu de ayni grafigi kullaniyor -> ortak fonksiyon
    fig = px.bar(
        df.sort_values(x_col),
        x=x_col,
        y=y_col,
        orientation="h",
        color_discrete_sequence=[BAR_COLOR],
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title=x_title,
        yaxis_title="",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    # plotly'nin varsayilan hover kutusunu ("trace 0" gibi) daha okunakli bir formatla degistirir
    fig.update_traces(hovertemplate=f"%{{y}}<br>{x_title}: %{{x:.3f}}<extra></extra>")
    return fig


model = load_model()  # sayfa her yenilendiginde cagrilir ama cache sayesinde islem tekrar etmez

st.title("🎬 Film Öneri Sistemi")
st.caption("MovieLens ml-latest-small verisiyle eğitilmiş content-based, collaborative filtering ve hybrid modeller")

# st.tabs bir liste dondurur, sirayla degiskenlere ayristirilir; her biri "with tab_x:" ile kullanilir
tab_content, tab_cf, tab_hybrid, tab_metrics = st.tabs(
    ["İçerik Bazlı", "Collaborative Filtering", "Hybrid", "Model Performansı"]
)

with tab_content:
    st.subheader("Bir filme benzer film öner")
    # recommend_content de clean_title (yilsiz baslik) dondurdugu icin secim listesi de bununla tutarli olmali
    titles = sorted(model.movies["clean_title"].dropna().unique())
    default_idx = titles.index("Toy Story") if "Toy Story" in titles else 0
    movie_title = st.selectbox("Film seç", titles, index=default_idx)
    # her sekmede ayni isimli slider oldugu icin benzersiz key sart, yoksa Streamlit hata verir
    k = st.slider("Kaç film önerilsin", 5, 20, 10, key="content_k")

    results = model.recommend_content(movie_title, k=k)
    if not results:
        # content_text bos olan filmlerde (tur/etiket bilgisi yoksa) tfidf vektoru uretilemez
        st.warning("Bu film için içerik verisi bulunamadı.")
    else:
        df = pd.DataFrame(results, columns=["title", "similarity"])
        st.plotly_chart(horizontal_bar(df, "similarity", "title", "Benzerlik"), use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab_cf:
    st.subheader("Bir kullanıcı için puan tahmini")
    user_ids = sorted(model.all_ratings["userId"].unique())
    user_id = st.selectbox("Kullanıcı ID", user_ids, key="cf_user")
    k = st.slider("Kaç film önerilsin", 5, 20, 10, key="cf_k")

    results = model.recommend_cf(user_id, k=k)
    df = pd.DataFrame(results, columns=["title", "predicted_rating"])
    st.plotly_chart(horizontal_bar(df, "predicted_rating", "title", "Tahmini Puan"), use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

with tab_hybrid:
    st.subheader("İçerik + Collaborative Filtering birleşimi")
    user_ids = sorted(model.all_ratings["userId"].unique())
    user_id = st.selectbox("Kullanıcı ID", user_ids, key="hybrid_user")
    k = st.slider("Kaç film önerilsin", 5, 20, 10, key="hybrid_k")
    alpha = st.slider(
        "Alpha (CF ağırlığı)", 0.0, 1.0, 0.9, 0.05,
        help="1.0 = tamamen collaborative filtering, 0.0 = tamamen içerik bazlı",
    )

    results = model.recommend_hybrid(user_id, k=k, alpha=alpha)
    if not results:
        # kullanicinin user_profiles'ta karsiligi yoksa (hic puanlamamis/CF'de yoksa) hybrid skor uretilemez
        st.warning("Bu kullanıcı için öneri üretilemedi (yeterli puanlama/içerik verisi yok).")
    else:
        df = pd.DataFrame(results, columns=["title", "score"])
        st.plotly_chart(horizontal_bar(df, "score", "title", "Hybrid Skor"), use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab_metrics:
    st.subheader("Collaborative Filtering Model Performansı")
    rmse, mae = model.evaluate_cf()  # test setindeki gercek puan vs SVD tahmini karsilastirmasi
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.4f}")
    col2.metric("MAE", f"{mae:.4f}")
    st.caption("Test setindeki gerçek puanlar ile SVD tahminleri arasındaki hata (düşük = iyi).")
