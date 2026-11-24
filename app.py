# Run: python -m streamlit run app.py
"""
Movie Recommendation System (V0) — Streamlit app.
Content-based + Collaborative filtering demo.
"""

import io
import streamlit as st
import pandas as pd

from recommender import (
    validate_movies,
    validate_ratings,
    standardize_movie_columns,
    standardize_ratings_columns,
    fit_content_model,
    content_similarity_matrix,
    recommend_content_based,
    build_user_item_matrix,
    mean_center_ignoring_zeros,
    user_similarity_matrix,
    predict_ratings_user_knn,
    recommend_collaborative,
    train_test_split_ratings,
    evaluate_collaborative,
    REQUIRED_MOVIES_COLS,
    REQUIRED_RATINGS_COLS,
)

# --- Embedded sample dataset (MovieLens-style) ---

SAMPLE_MOVIES_CSV = """movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
4,Waiting to Exhale (1995),Comedy|Drama|Romance
5,Father of the Bride Part II (1995),Comedy
6,Heat (1995),Action|Crime|Thriller
7,Sabrina (1995),Comedy|Romance
8,Tom and Huck (1995),Adventure|Children
9,Sudden Death (1995),Action
10,GoldenEye (1995),Action|Adventure|Thriller
11,American President The (1995),Comedy|Drama|Romance
12,Dracula: Dead and Loving It (1995),Comedy|Horror
13,Balto (1995),Adventure|Animation|Children
14,Nixon (1995),Drama
15,Cutthroat Island (1995),Action|Adventure|Romance
16,Casino (1995),Drama|Thriller
17,Sense and Sensibility (1995),Drama|Romance
18,Four Rooms (1995),Comedy
19,Ace Ventura: When Nature Calls (1995),Comedy
20,Money Train (1995),Action|Comedy|Drama|Thriller
"""

SAMPLE_RATINGS_CSV = """userId,movieId,rating,timestamp
1,1,4.0,964982703
1,2,4.0,964981247
1,3,3.5,964982931
1,6,4.0,964982224
1,10,4.0,964981357
1,11,4.0,964982931
1,14,3.0,964982931
1,16,4.0,964982224
2,1,5.0,974982703
2,2,3.0,974981247
2,5,4.0,974982931
2,6,5.0,974982224
2,7,4.0,974981357
2,10,5.0,974982931
2,11,4.0,974982224
2,15,3.0,974982931
2,17,5.0,974982224
3,1,3.5,984982703
3,3,4.0,984981247
3,4,4.0,984982931
3,6,3.0,984982224
3,8,4.0,984981357
3,10,4.0,984982931
3,12,3.0,984982224
3,18,4.0,984982931
4,2,4.0,994982703
4,5,3.0,994981247
4,7,5.0,994982931
4,9,4.0,994982224
4,11,4.0,994981357
4,14,3.5,994982931
4,16,5.0,994982224
4,19,4.0,994982931
5,1,5.0,1004982703
5,4,4.0,1004981247
5,6,5.0,1004982931
5,8,3.0,1004982224
5,10,5.0,1004981357
5,13,4.0,1004982931
5,17,4.0,1004982224
5,20,3.5,1004982931
"""


@st.cache_data
def load_sample_data():
    """Load embedded sample movies and ratings."""
    movies = pd.read_csv(io.StringIO(SAMPLE_MOVIES_CSV))
    ratings = pd.read_csv(io.StringIO(SAMPLE_RATINGS_CSV))
    return movies, ratings


def load_uploaded_movies(uploaded_file) -> pd.DataFrame | None:
    """Load movies from uploaded CSV. Returns None on error."""
    if uploaded_file is None:
        return None
    try:
        # Use Python engine with sep=None so pandas can infer delimiters
        # (handles commas, semicolons, etc. from tools like Excel).
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception as e:
        st.error(f"Could not read movies CSV: {e}")
        return None


def load_uploaded_ratings(uploaded_file) -> pd.DataFrame | None:
    """Load ratings from uploaded CSV. Returns None on error."""
    if uploaded_file is None:
        return None
    try:
        # Use the same flexible delimiter inference as for movies.
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception as e:
        st.error(f"Could not read ratings CSV: {e}")
        return None


@st.cache_resource
def get_content_model(movies: pd.DataFrame):
    """Cached TF-IDF and similarity for content-based (keyed by dataset)."""
    if movies is None or len(movies) == 0:
        return None
    vectorizer, tfidf, movie_id_to_idx = fit_content_model(movies)
    sim = content_similarity_matrix(tfidf)
    return vectorizer, tfidf, movie_id_to_idx, sim


@st.cache_resource
def get_collaborative_model(ratings: pd.DataFrame):
    """Cached user-item matrix and user similarity (keyed by ratings)."""
    if ratings is None or len(ratings) == 0:
        return None
    user_item, user_ids, movie_ids = build_user_item_matrix(ratings)
    centered = mean_center_ignoring_zeros(user_item)
    user_sim = user_similarity_matrix(centered)
    return user_item, user_ids, movie_ids, user_sim


def main():
    st.set_page_config(page_title="Movie Recommendation System", layout="wide")
    st.title("Movie Recommendation System (V0)")
    st.caption("Content-based + Collaborative filtering demo")

    # --- Sidebar: Dataset ---
    st.sidebar.header("Dataset")
    use_sample = st.sidebar.radio(
        "Data source",
        ["Use sample data", "Upload CSVs"],
        index=0,
    )
    movies = None
    ratings = None
    if use_sample == "Use sample data":
        movies, ratings = load_sample_data()
    else:
        up_movies = st.sidebar.file_uploader("movies.csv", type=["csv"])
        up_ratings = st.sidebar.file_uploader("ratings.csv", type=["csv"])
        if up_movies is not None:
            movies = load_uploaded_movies(up_movies)
            if movies is not None:
                movies = standardize_movie_columns(movies)
        if up_ratings is not None:
            ratings = load_uploaded_ratings(up_ratings)
            if ratings is not None:
                ratings = standardize_ratings_columns(ratings)
        if movies is not None and ratings is not None:
            ok_m, msg_m = validate_movies(movies)
            ok_r, msg_r = validate_ratings(ratings)
            if not ok_m:
                st.sidebar.error(msg_m + f" Found columns: {list(movies.columns)}")
                movies = None
            if not ok_r:
                st.sidebar.error(msg_r + f" Found columns: {list(ratings.columns)}")
                ratings = None

    # Dataset stats
    if movies is not None and ratings is not None:
        n_users = ratings["userId"].nunique()
        n_movies = movies["movieId"].nunique()
        n_ratings = len(ratings)
        st.sidebar.metric("Users", n_users)
        st.sidebar.metric("Movies", n_movies)
        st.sidebar.metric("Ratings", n_ratings)
    else:
        if use_sample == "Upload CSVs" and (movies is None or ratings is None):
            st.sidebar.info("Upload both movies.csv and ratings.csv to continue.")
        else:
            st.sidebar.warning("No valid dataset loaded.")

    # --- Mode ---
    st.sidebar.header("Mode")
    mode = st.sidebar.radio(
        "Recommendation mode",
        ["Content-Based", "Collaborative (User-kNN)"],
        index=0,
    )

    # --- Main: mode-specific controls and run ---
    result_table = None
    eval_metrics = None

    if movies is not None and ratings is not None:
        if mode == "Content-Based":
            st.subheader("Content-Based Recommendations")
            titles = movies["title"].tolist()
            title_to_id = dict(zip(movies["title"], movies["movieId"]))
            selected_title = st.selectbox(
                "Select a movie",
                options=titles,
                index=0,
            )
            top_n = st.slider("Top N", min_value=5, max_value=20, value=10, key="cb_topn")
            if st.button("Recommend Similar Movies"):
                selected_movie_id = title_to_id.get(selected_title)
                if selected_movie_id is None:
                    st.error("Selected movie not found in dataset.")
                else:
                    model = get_content_model(movies)
                    if model is None:
                        st.error("Could not build content model (no movies).")
                    else:
                        _, _, movie_id_to_idx, sim = model
                        result_table = recommend_content_based(
                            movies, model[1], sim, movie_id_to_idx, selected_movie_id, top_n
                        )
                        if result_table.empty:
                            st.warning("No similar movies found (e.g. unknown title or no overlap).")
                        else:
                            st.dataframe(result_table, use_container_width=True, hide_index=True)

        else:
            st.subheader("Collaborative (User-kNN) Recommendations")
            user_ids_sorted = sorted(ratings["userId"].unique())
            selected_user = st.selectbox(
                "Select user (userId)",
                options=user_ids_sorted,
                index=0,
            )
            top_n = st.slider("Top N", min_value=5, max_value=20, value=10, key="col_topn")
            max_neighbors = max(1, len(user_ids_sorted) - 1) 
            default_neighbors = min(15, max_neighbors)  
            k = st.slider(     
                "k neighbors",     
                min_value=1,     
                max_value=max_neighbors,     
                value=default_neighbors,     
                key="col_k",     
                help="The maximum adjusts to the number of users in the loaded dataset.", 
            )
            if st.button("Recommend For User"):
                collab = get_collaborative_model(ratings)
                if collab is None:
                    st.error("Could not build collaborative model (no ratings).")
                else:
                    user_item, user_ids, movie_ids, user_sim = collab
                    result_table = recommend_collaborative(
                        movies,
                        ratings,
                        user_item,
                        user_sim,
                        user_ids,
                        movie_ids,
                        selected_user,
                        k,
                        top_n,
                    )
                    if result_table.empty:
                        st.warning(
                            "No recommendations (user may have no ratings or too few neighbors)."
                        )
                    else:
                        st.dataframe(result_table, use_container_width=True, hide_index=True)

        # --- Evaluation: collaborative filtering only ---
        if mode == "Collaborative (User-kNN)":
            st.sidebar.header("Collaborative Filtering Evaluation")
            st.sidebar.caption(
                "These metrics evaluate User-kNN only, not Content-Based Filtering."
            )

            train_fraction = st.sidebar.slider(
                "Train fraction (rest = test)",
                min_value=0.5,
                max_value=0.95,
                value=0.8,
                step=0.05,
            )
            test_frac = 1.0 - train_fraction

            rank_k = st.sidebar.slider(
                "K for ranking metrics",
                min_value=1,
                max_value=20,
                value=10,
            )

            max_eval_neighbors = max(1, ratings["userId"].nunique() - 1)
            default_eval_neighbors = min(15, max_eval_neighbors)

            k_neighbors_eval = st.sidebar.slider(
                "K neighbors for evaluation",
                min_value=1,
                max_value=max_eval_neighbors,
                value=default_eval_neighbors,
                key="eval_k",
            )

            if st.sidebar.button("Run Collaborative Evaluation"):
                train_r, test_r = train_test_split_ratings(ratings, test_frac)
                eval_metrics = evaluate_collaborative(
                    train_r, test_r, movies, k_neighbors_eval, rank_k
                )
                st.session_state["last_eval"] = eval_metrics
    # Show last evaluation metrics if available
    if "last_eval" in st.session_state:
        m = st.session_state["last_eval"]
        st.subheader("Collaborative Filtering Evaluation Results") 
        st.caption(     
            "RMSE, Precision@K and Recall@K are calculated for the User-kNN collaborative filtering method." 
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("RMSE", f"{m['rmse']:.4f}" if not pd.isna(m["rmse"]) else "N/A")
        with c2:
            st.metric("Precision@K", f"{m['precision_at_k']:.4f}" if not pd.isna(m["precision_at_k"]) else "N/A")
        with c3:
            st.metric("Recall@K", f"{m['recall_at_k']:.4f}" if not pd.isna(m["recall_at_k"]) else "N/A")

    # --- How it works ---
    with st.expander("How it works"):
        st.markdown("""
**Content-Based (TF-IDF + cosine similarity)**  
We build a text “content” for each movie: *title* and *genres* (with `|` replaced by spaces), lowercased.
TF-IDF turns this into a vector per movie. We compute cosine similarity between all movie vectors.
For a selected movie, we return the top-N most similar movies (by cosine similarity), excluding the movie itself.

**Collaborative (User-kNN)**  
We build a user–item matrix of ratings and mean-center each user’s ratings (ignoring zeros as “no rating”).
We compute user–user cosine similarity on these centered vectors. To recommend for a user, we take their
k nearest neighbors and predict ratings as a weighted average of the neighbors’ centered ratings (weighted by
similarity), then add back the user’s mean. We recommend the top-N movies the user hasn’t rated yet, by predicted rating.

**Evaluation**  
- **RMSE**: Root mean squared error on the test set (predictions vs. actual ratings).  
- **Relevant**: Test items with rating ≥ 4.0.  
- **Precision@K**: For each user, we recommend top-K items; Precision@K = (hits in relevant) / K.  
- **Recall@K**: For each user, Recall@K = (hits in relevant) / (number of relevant items).  
We report macro-average Precision@K and Recall@K over users that have at least one relevant item and at least K candidates.
""")

    if movies is None or ratings is None:
        st.info("Load a dataset (sample or upload) to use recommendations and evaluation.")


if __name__ == "__main__":
    main()
