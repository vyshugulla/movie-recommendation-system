"""
Movie Recommendation System — core logic.
Content-based (TF-IDF + cosine similarity) and collaborative (user-kNN) filtering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- Column normalization + validation ---

REQUIRED_MOVIES_COLS = {"movieId", "title", "genres"}
REQUIRED_RATINGS_COLS = {"userId", "movieId", "rating"}


def _normalize_name(name: str) -> str:
    """Lowercase, strip spaces/underscores for matching."""
    return "".join(name.strip().lower().replace("_", "").split())


def standardize_movie_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to map common header variants to canonical movie columns.
    This helps with files coming from Excel or other tools.
    """
    mapping: dict[str, str] = {}
    for col in df.columns:
        norm = _normalize_name(col)
        if norm in {"movieid", "movie"}:
            mapping[col] = "movieId"
        elif norm in {"title", "movietitle", "name"}:
            mapping[col] = "title"
        elif norm in {"genres", "genre"} or "genre" in norm:
            mapping[col] = "genres"
    if mapping:
        df = df.rename(columns=mapping)
    return df


def standardize_ratings_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to map common header variants to canonical ratings columns.
    """
    mapping: dict[str, str] = {}
    for col in df.columns:
        norm = _normalize_name(col)
        if norm in {"userid", "user"}:
            mapping[col] = "userId"
        elif norm in {"movieid", "movie"}:
            mapping[col] = "movieId"
        elif norm in {"rating", "score", "stars"} or "rating" in norm:
            mapping[col] = "rating"
    if mapping:
        df = df.rename(columns=mapping)
    return df


def validate_movies(df: pd.DataFrame) -> tuple[bool, str]:
    """Check movies DataFrame has required columns. Return (ok, message)."""
    missing = REQUIRED_MOVIES_COLS - set(df.columns)
    if missing:
        return False, f"movies.csv missing columns: {sorted(missing)}. Required: {REQUIRED_MOVIES_COLS}"
    return True, "OK"


def validate_ratings(df: pd.DataFrame) -> tuple[bool, str]:
    """Check ratings DataFrame has required columns. Return (ok, message)."""
    missing = REQUIRED_RATINGS_COLS - set(df.columns)
    if missing:
        return False, f"ratings.csv missing columns: {sorted(missing)}. Required: {REQUIRED_RATINGS_COLS}"
    return True, "OK"


# --- Content-based ---


def build_content_string(title: str, genres: str) -> str:
    """Create content = (title + ' ' + genres) lowercased, pipe replaced with space."""
    g = (genres or "").replace("|", " ").strip()
    t = (title or "").strip()
    return f"{t} {g}".lower()


def prepare_content(df: pd.DataFrame) -> pd.Series:
    """Build content series for TF-IDF from movies DataFrame."""
    return df.apply(lambda r: build_content_string(r["title"], r["genres"]), axis=1)


def fit_content_model(movies: pd.DataFrame):
    """
    Fit TF-IDF on movie content (title + genres).
    Returns (vectorizer, tfidf_matrix, movie_id_to_idx).
    """
    content = prepare_content(movies)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(content)
    movie_id_to_idx = {mid: i for i, mid in enumerate(movies["movieId"].values)}
    return vectorizer, tfidf, movie_id_to_idx


def content_similarity_matrix(tfidf_matrix):
    """Compute pairwise cosine similarity (dense)."""
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend_content_based(
    movies: pd.DataFrame,
    tfidf_matrix,
    sim_matrix: np.ndarray,
    movie_id_to_idx: dict,
    selected_movie_id: int,
    top_n: int,
) -> pd.DataFrame:
    """
    Return top-N similar movies (excluding self).
    Columns: Title, Genres, Similarity score.
    """
    if selected_movie_id not in movie_id_to_idx:
        return pd.DataFrame(columns=["Title", "Genres", "Similarity score"])
    idx = movie_id_to_idx[selected_movie_id]
    sims = sim_matrix[idx].copy()
    sims[idx] = -1.0  # exclude self
    top_indices = np.argsort(sims)[::-1][:top_n]
    rows = []
    for i in top_indices:
        if sims[i] <= 0:
            break
        row = movies.iloc[i]
        rows.append({
            "Title": row["title"],
            "Genres": row["genres"],
            "Similarity score": round(float(sims[i]), 4),
        })
    return pd.DataFrame(rows)


# --- Collaborative (user-kNN) ---


def build_user_item_matrix(ratings: pd.DataFrame):
    """
    Pivot ratings to user-item matrix. Rows=users, cols=movies.
    Returns (matrix, user_ids, movie_ids).
    """
    pivot = ratings.pivot_table(
        index="userId", columns="movieId", values="rating", aggfunc="mean"
    )
    user_ids = pivot.index.values
    movie_ids = pivot.columns.values
    return pivot.fillna(0).values, user_ids, movie_ids


def mean_center_ignoring_zeros(matrix: np.ndarray) -> np.ndarray:
    """
    Mean-center each row; treat 0 as missing (do not count in mean).
    """
    out = matrix.astype(float).copy()
    for i in range(out.shape[0]):
        row = out[i]
        mask = row != 0
        if mask.any():
            out[i] = row - row[mask].mean()
        # else leave row as-is (all zeros)
    return out


def user_similarity_matrix(centered: np.ndarray) -> np.ndarray:
    """User-user cosine similarity (rows are users)."""
    # Avoid div-by-zero: normalize rows; zero-norm rows stay zero
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = centered / norms
    return cosine_similarity(normalized, normalized)


def predict_ratings_user_knn(
    user_item: np.ndarray,
    user_sim: np.ndarray,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    target_user_id: int,
    k: int,
) -> np.ndarray:
    """
    For target user, predict rating for each movie using weighted average of
    neighbors' centered ratings, then add back target user's mean.
    Returns 1D array of predicted ratings (same order as movie_ids).
    """
    try:
        u_idx = np.where(user_ids == target_user_id)[0][0]
    except IndexError:
        return np.full(len(movie_ids), np.nan)
    user_means = np.nanmean(np.where(user_item != 0, user_item, np.nan), axis=1)
    user_means = np.where(np.isnan(user_means), 0.0, user_means)
    centered = mean_center_ignoring_zeros(user_item)
    sim_row = user_sim[u_idx].copy()
    sim_row[u_idx] = -1  # exclude self
    neighbor_idx = np.argsort(sim_row)[::-1][:k]
    pred = np.zeros(len(movie_ids))
    for m_idx in range(len(movie_ids)):
        # weighted sum of neighbors' centered ratings for this movie
        num = 0.0
        den = 0.0
        for n in neighbor_idx:
            r = centered[n, m_idx]
            if not np.isnan(r) and user_item[n, m_idx] != 0:
                w = max(0, sim_row[n])
                num += w * r
                den += w
        if den > 0:
            pred[m_idx] = user_means[u_idx] + (num / den)
        else:
            pred[m_idx] = user_means[u_idx]
    return pred


def recommend_collaborative(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    user_item: np.ndarray,
    user_sim: np.ndarray,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    target_user_id: int,
    k: int,
    top_n: int,
) -> pd.DataFrame:
    """
    Top-N movies not yet rated by user, by predicted rating.
    Columns: Title, Genres, Predicted rating.
    """
    try:
        u_idx = np.where(user_ids == target_user_id)[0][0]
    except IndexError:
        return pd.DataFrame(columns=["Title", "Genres", "Predicted rating"])
    already_rated = set(
        ratings.loc[ratings["userId"] == target_user_id, "movieId"].values
    )
    pred = predict_ratings_user_knn(
        user_item, user_sim, user_ids, movie_ids, target_user_id, k
    )
    mid_to_pred = {movie_ids[j]: pred[j] for j in range(len(movie_ids))}
    candidates = [(mid, mid_to_pred[mid]) for mid in movie_ids if mid not in already_rated]
    candidates.sort(key=lambda x: -x[1])
    top = candidates[:top_n]
    if not top:
        return pd.DataFrame(columns=["Title", "Genres", "Predicted rating"])
    movie_id_to_row = movies.set_index("movieId").loc
    rows = []
    for mid, p in top:
        try:
            r = movie_id_to_row[mid]
            rows.append({
                "Title": r["title"],
                "Genres": r["genres"],
                "Predicted rating": round(float(p), 2),
            })
        except KeyError:
            continue
    return pd.DataFrame(rows)


# --- Evaluation ---

RELEVANT_THRESHOLD = 4.0
RANDOM_SEED = 42


def train_test_split_ratings(ratings: pd.DataFrame, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split by rows. test_frac in (0,1)."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(ratings)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return ratings.iloc[train_idx].copy(), ratings.iloc[test_idx].copy()


def evaluate_collaborative(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    k_neighbors: int,
    rank_k: int,
) -> dict:
    """
    Compute RMSE on test set, and macro Precision@K / Recall@K.
    Relevant = rating >= 4.0.
    """
    user_item, user_ids, movie_ids = build_user_item_matrix(train_ratings)
    centered = mean_center_ignoring_zeros(user_item)
    user_sim = user_similarity_matrix(centered)
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    # RMSE on test
    preds = []
    actuals = []
    for _, row in test_ratings.iterrows():
        uid, mid, rating = row["userId"], row["movieId"], row["rating"]
        if uid not in user_id_to_idx or mid not in movie_id_to_idx:
            continue
        pred = predict_ratings_user_knn(
            user_item, user_sim, user_ids, movie_ids, uid, k_neighbors
        )
        m_idx = movie_id_to_idx[mid]
        preds.append(pred[m_idx])
        actuals.append(rating)
    if not preds:
        rmse = float("nan")
    else:
        rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2)))

    # Precision@K and Recall@K per user (macro)
    test_by_user = test_ratings.groupby("userId")
    precisions = []
    recalls = []
    for uid, grp in test_by_user:
        if uid not in user_id_to_idx:
            continue
        relevant = set(grp.loc[grp["rating"] >= RELEVANT_THRESHOLD, "movieId"].values)
        if not relevant:
            continue
        pred = predict_ratings_user_knn(
            user_item, user_sim, user_ids, movie_ids, uid, k_neighbors
        )
        already_rated_train = set(
            train_ratings.loc[train_ratings["userId"] == uid, "movieId"].values
        )
        candidates = [
            (movie_ids[j], pred[j])
            for j in range(len(movie_ids))
            if movie_ids[j] not in already_rated_train
        ]
        candidates.sort(key=lambda x: -x[1])
        top_k_ids = [mid for mid, _ in candidates[:rank_k]]
        if len(top_k_ids) < rank_k:
            # fewer than K candidates: still compute with actual list length
            pass
        hits = len(set(top_k_ids) & relevant)
        precisions.append(hits / min(rank_k, len(top_k_ids)) if top_k_ids else 0.0)
        recalls.append(hits / len(relevant) if relevant else 0.0)
    if not precisions:
        prec_k = rec_k = float("nan")
    else:
        prec_k = float(np.mean(precisions))
        rec_k = float(np.mean(recalls))
    return {"rmse": rmse, "precision_at_k": prec_k, "recall_at_k": rec_k}
