"""Ham CSV dosyalarini okuma fonksiyonlari (dataset/ml-latest-small)."""

import pandas as pd
from . import config


def load_movies():
    return pd.read_csv(config.DATA_DIR / "movies.csv")


def load_ratings():
    return pd.read_csv(config.DATA_DIR / "ratings.csv")


def load_tags():
    return pd.read_csv(config.DATA_DIR / "tags.csv")


def load_links():
    return pd.read_csv(config.DATA_DIR / "links.csv")
