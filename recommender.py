
# =====================================================
# PERSONALIZED MOVIE RECOMMENDATION ENGINE
# Backend Module
# =====================================================

import os
import joblib
import pandas as pd
import numpy as np


# =====================================================
# 1. LOAD TRAINED MODEL
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "recommendation_engine.pkl"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model not found!")

print("Loading trained recommendation model...")

model = joblib.load(MODEL_PATH)

movies = model["movies"]
movies_full = model["movies_full"]

user_movie_matrix = model["user_movie_matrix"]
user_similarity_df = model["user_similarity_df"]

cosine_sim = model["cosine_sim"]

print("Recommendation model loaded successfully!")


# =====================================================
# 2. CONTENT-BASED FILTERING
# =====================================================

def get_content_scores(user_id):

    user_ratings = user_movie_matrix.loc[user_id]

    # Select movies liked by the user
    liked_movies = user_ratings[user_ratings >= 4]

    if liked_movies.empty:
        liked_movies = user_ratings[
            user_ratings > 0
        ].nlargest(5)

    # Map movie IDs to matrix positions
    movie_index = dict(zip(
        movies_full["movie_id"],
        movies_full.index
    ))

    scores = {}

    for movie_id in user_movie_matrix.columns:

        # Skip movies already watched
        if user_ratings[movie_id] > 0:
            continue

        target_idx = movie_index.get(movie_id)

        if target_idx is None:
            continue

        similarities = []

        for liked_id, rating in liked_movies.items():

            liked_idx = movie_index.get(liked_id)

            if liked_idx is not None:

                similarity = cosine_sim[
                    target_idx, liked_idx
                ]

                similarities.append(
                    (float(similarity), float(rating - 3))
                )

        if similarities:

            total_weight = sum(
                weight for _, weight in similarities
            )

            if total_weight > 0:

                scores[movie_id] = sum(
                    similarity * weight
                    for similarity, weight in similarities
                ) / total_weight

    return pd.Series(scores, dtype="float64")


# =====================================================
# 3. COLLABORATIVE FILTERING
# =====================================================

def get_collaborative_scores(user_id, n_neighbors=10):

    similar_users = (
        user_similarity_df.loc[user_id]
        .drop(user_id)
        .nlargest(n_neighbors)
    )

    # Get ratings from similar users
    neighbor_ratings = user_movie_matrix.loc[
        similar_users.index
    ]

    weights = similar_users.clip(lower=0)

    # Calculate weighted rating predictions
    weighted_sum = neighbor_ratings.T.dot(weights)

    rating_weights = (
        neighbor_ratings.gt(0).T.dot(weights)
    )

    predicted_scores = weighted_sum.div(
        rating_weights.replace(0, np.nan)
    )

    # Exclude movies already rated by the user
    user_ratings = user_movie_matrix.loc[user_id]

    predicted_scores = predicted_scores[
        user_ratings == 0
    ]

    predicted_scores = pd.to_numeric(
        predicted_scores,
        errors="coerce"
    )

    return predicted_scores.dropna().astype("float64")


# =====================================================
# 4. SCORE NORMALIZATION
# =====================================================

def normalize_scores(scores):

    scores = pd.to_numeric(
        scores,
        errors="coerce"
    ).astype("float64")

    scores = scores.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if scores.dropna().empty:
        return scores.fillna(0.0)

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return pd.Series(
            0.5,
            index=scores.index,
            dtype="float64"
        )

    return (
        (scores - minimum) /
        (maximum - minimum)
    ).fillna(0.0)


# =====================================================
# 5. HYBRID RECOMMENDATION ENGINE
# =====================================================

def recommend_hybrid(
    user_id,
    top_n=10,
    content_weight=0.4,
    collaborative_weight=0.6
):

    # Validate user
    if user_id not in user_movie_matrix.index:
        raise ValueError(
            f"User ID {user_id} not found!"
        )

    # Validate recommendation count
    if top_n < 1:
        raise ValueError(
            "top_n must be greater than zero!"
        )

    # Validate weights
    if content_weight < 0 or collaborative_weight < 0:
        raise ValueError("Weights cannot be negative!")

    total_weight = (
        content_weight + collaborative_weight
    )

    if total_weight == 0:
        raise ValueError(
            "At least one weight must be positive!"
        )

    # Normalize weights
    content_weight /= total_weight
    collaborative_weight /= total_weight

    # Generate recommendation scores
    content_scores = get_content_scores(user_id)

    collaborative_scores = get_collaborative_scores(
        user_id
    )

    # Combine both recommendation methods
    combined = pd.concat(
        [
            content_scores.rename("content_score"),
            collaborative_scores.rename(
                "collaborative_score"
            )
        ],
        axis=1
    )

    # Explicitly enforce numeric data types
    combined["content_score"] = pd.to_numeric(
        combined["content_score"],
        errors="coerce"
    ).astype("float64")

    combined["collaborative_score"] = pd.to_numeric(
        combined["collaborative_score"],
        errors="coerce"
    ).astype("float64")

    # Normalize individual scores
    combined["content_score"] = normalize_scores(
        combined["content_score"]
    )

    combined["collaborative_score"] = normalize_scores(
        combined["collaborative_score"]
    )

    # Calculate final hybrid score
    combined["hybrid_score"] = (
        content_weight *
        combined["content_score"].fillna(0.0)
        +
        collaborative_weight *
        combined["collaborative_score"].fillna(0.0)
    )

    # Final numeric conversion
    combined["hybrid_score"] = pd.to_numeric(
        combined["hybrid_score"],
        errors="coerce"
    ).astype("float64")

    # Remove invalid scores
    combined = combined.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(subset=["hybrid_score"])

    # Select top recommendations
    top_movies = combined.nlargest(
        top_n,
        "hybrid_score"
    )

    # Convert movie IDs into a column
    results = top_movies.rename_axis(
        "movie_id"
    ).reset_index()

    # Attach movie information
    results = results.merge(
        movies_full[
            ["movie_id", "title", "genres"]
        ],
        on="movie_id",
        how="left"
    )

    # Return final recommendations
    return results[
        [
            "movie_id",
            "title",
            "genres",
            "content_score",
            "collaborative_score",
            "hybrid_score"
        ]
    ].reset_index(drop=True)


# =====================================================
# 6. BACKEND STATUS
# =====================================================

print("Hybrid recommendation engine initialized!")
print("Total movies:", len(movies_full))
print("Total users:", len(user_movie_matrix))
