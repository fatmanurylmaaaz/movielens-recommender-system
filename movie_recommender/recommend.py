"""Komut satirindan oneri almak icin giris noktasi.

Kullanim ornekleri (proje kokunden calistirilmali):
    python -m movie_recommender.recommend --model content --movie-title "Toy Story"
    python -m movie_recommender.recommend --model cf --user-id 1
    python -m movie_recommender.recommend --model hybrid --user-id 1 --k 5
"""

import argparse
from .pipeline import MovieRecommender
from . import config


def main():
    parser = argparse.ArgumentParser(description="Film oneri sistemi (content-based / CF / hybrid)")
    parser.add_argument("--model", choices=["content", "cf", "hybrid"], default="hybrid")
    parser.add_argument("--user-id", type=int, help="CF/hybrid modeli icin gerekli")
    parser.add_argument("--movie-title", type=str, help="content-based modeli icin gerekli")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=config.HYBRID_ALPHA, help="hybrid agirligi")
    parser.add_argument("--no-cache", action="store_true", help="modeli sifirdan egit, cache kullanma")
    args = parser.parse_args()

    model = MovieRecommender().fit(use_cache=not args.no_cache)

    if args.model == "content":
        if not args.movie_title:
            parser.error("--model content icin --movie-title gerekli")
        results = model.recommend_content(args.movie_title, k=args.k)
    elif args.model == "cf":
        if args.user_id is None:
            parser.error("--model cf icin --user-id gerekli")
        results = model.recommend_cf(args.user_id, k=args.k)
    else:
        if args.user_id is None:
            parser.error("--model hybrid icin --user-id gerekli")
        results = model.recommend_hybrid(args.user_id, k=args.k, alpha=args.alpha)

    if not results:
        print("Oneri uretilemedi (yetersiz veri ya da eslesme bulunamadi).")
        return

    for title, score in results:
        print(f"{title}  (skor: {score:.3f})")


if __name__ == "__main__":
    main()
