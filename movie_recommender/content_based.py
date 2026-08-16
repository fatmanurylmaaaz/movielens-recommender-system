"""TF-IDF + cosine similarity ile icerik tabanli oneri (scripts/10-12'nin paketlenmis hali)."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from . import config


def build_tfidf(movies):
    vectorizer = TfidfVectorizer(min_df=config.TFIDF_MIN_DF)
    tfidf_matrix = vectorizer.fit_transform(movies["content_text"])
    return vectorizer, tfidf_matrix


def find_movie(title_query, movies):
    """'The X' <-> 'X, The' donusumunu de destekleyen esnek baslik aramasi."""
    query = title_query.strip()

    def starts_with(q):
        return movies[movies["clean_title"].str.lower().str.startswith(q.lower(), na=False)]

    result = starts_with(query)
    if result.empty and query.lower().startswith("the "):
        result = starts_with(query[4:] + ", The")
    if not result.empty:
        return result

    contains = movies[
        movies["clean_title"].str.contains(query, case=False, regex=False, na=False)
    ].copy()
    return contains.sort_values(by="clean_title", key=lambda s: s.str.len())


def recommend_similar(movie_title, movies, tfidf_matrix, k=10):
    """Verilen filme icerik (genre/tag) bakimindan en benzer k filmi dondurur."""
    matches = find_movie(movie_title, movies)
    if matches.empty:
        return []

    query_idx = matches.index[0]
    if movies.loc[query_idx, "content_text"] == "":
        return []

    sim_scores = cosine_similarity(tfidf_matrix[query_idx], tfidf_matrix).flatten()
    top_indices = [i for i in sim_scores.argsort()[::-1] if i != query_idx][:k]
    return [(movies.loc[i, "clean_title"], float(sim_scores[i])) for i in top_indices]
