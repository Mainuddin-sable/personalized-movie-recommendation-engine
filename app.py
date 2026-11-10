
# ============================================================
# PERSONALIZED MOVIE RECOMMENDATION ENGINE
# Premium Streamlit Web Application
# ============================================================

import os
import streamlit as st
import pandas as pd
import plotly.express as px

from recommender import (
    recommend_hybrid,
    movies_full,
    movies,
    user_movie_matrix
)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSS_PATH = os.path.join(BASE_DIR, "styles.css")

# Load custom CSS
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


# ============================================================
# 2. SIDEBAR
# ============================================================

st.sidebar.markdown("# 🎬 CineMatch")
st.sidebar.caption("Your Personal Movie Discovery Platform")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Home",
        "🎯 Recommendations",
        "🔍 Movie Search",
        "🔥 Popular Movies",
        "📊 Analytics"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 👤 User Profile")

user_ids = sorted(
    [int(user_id) for user_id in user_movie_matrix.index]
)

selected_user = st.sidebar.selectbox(
    "Select User ID",
    user_ids,
    index=user_ids.index(196) if 196 in user_ids else 0
)

st.sidebar.caption(
    f"User ID: {selected_user}"
)

st.sidebar.divider()

st.sidebar.markdown("### ⚙️ Recommendation Settings")

content_weight = st.sidebar.slider(
    "Content-Based Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.1
)

collaborative_weight = 1.0 - content_weight

st.sidebar.caption(
    f"Collaborative Weight: {collaborative_weight:.1f}"
)

st.sidebar.divider()

st.sidebar.markdown(
    "Developed with Python, Scikit-learn & Streamlit"
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def show_hero(title, description):

    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_metric(label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_movie_cards(data, columns_per_row=3):

    if data.empty:
        st.info("No movies available.")
        return

    for start in range(0, len(data), columns_per_row):

        cols = st.columns(columns_per_row)

        for col, (_, movie) in zip(
            cols,
            data.iloc[start:start + columns_per_row].iterrows()
        ):

            with col:

                title = str(movie.get("title", "Unknown Movie"))
                genres = str(movie.get("genres", "Unknown"))
                score = movie.get("hybrid_score", None)

                if pd.notna(score):
                    score_text = f"{float(score):.3f}"
                else:
                    score_text = "Popular"

                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">
                            🎬 {title}
                        </div>
                        <div class="movie-genre">
                            {genres}
                        </div>
                        <div class="movie-score">
                            Recommendation Score: {score_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def get_popular_movies(limit=10):

    # Number of ratings per movie
    rating_counts = user_movie_matrix.gt(0).sum(axis=0)

    # Average rating per movie
    average_ratings = user_movie_matrix.replace(
        0, float("nan")
    ).mean(axis=0)

    popularity = pd.DataFrame({
        "movie_id": rating_counts.index,
        "total_ratings": rating_counts.values,
        "average_rating": average_ratings.values
    })

    popularity = popularity.merge(
        movies[["movie_id", "title"]],
        on="movie_id",
        how="left"
    )

    return popularity.sort_values(
        "total_ratings",
        ascending=False
    ).head(limit)


# ============================================================
# 4. GENERATE PERSONALIZED RECOMMENDATIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_recommendations(user_id, content_weight, collaborative_weight):

    return recommend_hybrid(
        user_id=user_id,
        top_n=10,
        content_weight=content_weight,
        collaborative_weight=collaborative_weight
    )


try:

    with st.spinner("Finding movies for you..."):

        recommendations = load_recommendations(
            int(selected_user),
            float(content_weight),
            float(collaborative_weight)
        )

except Exception as error:

    st.error(f"Recommendation error: {error}")
    st.stop()


# ============================================================
# 5. HOME PAGE
# ============================================================

if page == "🏠 Home":

    show_hero(
        "Discover Your Next Favourite Movie 🍿",
        "Personalized movie recommendations powered by Machine Learning."
    )

    # Dashboard metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        show_metric("TOTAL MOVIES", f"{len(movies_full):,}")

    with col2:
        show_metric("TOTAL USERS", f"{len(user_movie_matrix):,}")

    with col3:
        show_metric("YOUR RECOMMENDATIONS", len(recommendations))

    st.markdown("---")

    st.subheader("✨ Recommended For You")

    show_movie_cards(recommendations.head(6))

    st.markdown("---")

    st.subheader("🔥 Popular Movies")

    popular = get_popular_movies(6)

    show_movie_cards(popular)

    st.markdown(
        """
        <div class="footer">
            CineMatch | Personalized Movie Recommendation Engine
            <br>
            Built using Python, Machine Learning and Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 6. RECOMMENDATIONS PAGE
# ============================================================

elif page == "🎯 Recommendations":

    show_hero(
        "Your Personal Recommendations 🎯",
        f"Movie suggestions generated for User {selected_user}."
    )

    st.subheader("Movies Selected For You")

    st.caption(
        "Recommendations combine movie genres and similar-user preferences."
    )

    show_movie_cards(recommendations)

    st.markdown("---")

    st.subheader("📈 Recommendation Score Comparison")

    chart_data = recommendations[
        [
            "title",
            "content_score",
            "collaborative_score",
            "hybrid_score"
        ]
    ].melt(
        id_vars="title",
        var_name="Recommendation Method",
        value_name="Score"
    )

    fig = px.bar(
        chart_data,
        x="title",
        y="Score",
        color="Recommendation Method",
        barmode="group",
        template="plotly_dark",
        title="Recommendation Scores"
    )

    fig.update_layout(
        xaxis_title="Movies",
        yaxis_title="Normalized Score",
        xaxis_tickangle=-45,
        paper_bgcolor="#0b0d14",
        plot_bgcolor="#141824",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 7. MOVIE SEARCH PAGE
# ============================================================

elif page == "🔍 Movie Search":

    show_hero(
        "Explore Movies 🔍",
        "Search the MovieLens collection by movie title."
    )

    search_query = st.text_input(
        "Search Movie",
        placeholder="Example: Toy Story, Star Wars, Fargo..."
    )

    if search_query.strip():

        results = movies_full[
            movies_full["title"].str.contains(
                search_query,
                case=False,
                na=False
            )
        ]

        if not results.empty:

            st.success(f"{len(results)} matching movies found.")

            show_movie_cards(results.head(18))

        else:

            st.warning("No matching movies found.")

    else:

        st.info("Enter a movie title to start searching.")

        st.subheader("Suggested Movies")

        show_movie_cards(movies_full.head(6))


# ============================================================
# 8. POPULAR MOVIES PAGE
# ============================================================

elif page == "🔥 Popular Movies":

    show_hero(
        "Most Popular Movies 🔥",
        "Discover movies with the highest number of user ratings."
    )

    popular = get_popular_movies(20)

    st.dataframe(
        popular[
            [
                "title",
                "average_rating",
                "total_ratings"
            ]
        ].reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Most Rated Movies")

    fig = px.bar(
        popular.head(10),
        x="title",
        y="total_ratings",
        template="plotly_dark",
        title="Top 10 Movies by Number of Ratings"
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        paper_bgcolor="#0b0d14",
        plot_bgcolor="#141824",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 9. ANALYTICS PAGE
# ============================================================

elif page == "📊 Analytics":

    show_hero(
        "Recommendation Analytics 📊",
        "Explore the scores generated by your recommendation engine."
    )

    st.subheader("Recommendation Score Distribution")

    score_data = recommendations[
        [
            "content_score",
            "collaborative_score",
            "hybrid_score"
        ]
    ].melt(
        var_name="Method",
        value_name="Score"
    )

    fig = px.box(
        score_data,
        x="Method",
        y="Score",
        color="Method",
        template="plotly_dark",
        title="Score Distribution by Recommendation Method"
    )

    fig.update_layout(
        paper_bgcolor="#0b0d14",
        plot_bgcolor="#141824",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Recommendation Data")

    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 10. FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "CineMatch v1.0 | Machine Learning Project"
)
