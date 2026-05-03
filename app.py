import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Movie Recommendation System Using Machine Learning",
    page_icon="🎬",
    layout="wide"
)

# -------------------- UI CSS --------------------
st.markdown(
    """
    <style>
    .movie-card {
        padding:18px;
        border-radius:14px;
        background: linear-gradient(145deg, #1c1f26, #232733);
        text-align:center;
        height:200px;
        display:flex;
        flex-direction:column;
        justify-content:space-between;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
    .movie-title {
        font-size:16px;
        font-weight:600;
        line-height:1.3;
        color:#f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
    movies = pd.DataFrame(movies_dict)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["Tags"]).toarray()
    similarity = cosine_similarity(vectors)

    return movies, similarity

movies, similarity = load_data()

# -------------------- RECOMMEND FUNCTION --------------------
def recommend(movie_title, min_rating=0):
    if movie_title not in movies["title"].values:
        return [], []

    idx = movies[movies["title"] == movie_title].index[0]
    scores = sorted(
        list(enumerate(similarity[idx])),
        key=lambda x: x[1],
        reverse=True
    )[1:]

    titles, ratings = [], []
    for i, _ in scores:
        r = movies.iloc[i].vote_average
        if r < min_rating:
            continue
        titles.append(movies.iloc[i].title)
        ratings.append(r)
        if len(titles) == 5:
            break

    return titles, ratings

# -------------------- SIDEBAR --------------------
st.sidebar.title("🎥 About Project")
st.sidebar.info(
    """
    **Movie Recommendation System**

    🔹 Content-Based Filtering  
    🔹 Cosine Similarity  
    🔹 Streamlit Frontend  
    🔹 Python + ML  

    Built for **Placement Showcase**
    """
)

min_rating = st.sidebar.slider(
    "Minimum recommended movie rating",
    min_value=0.0,
    max_value=10.0,
    value=4.0,
    step=0.5,
    help="Only include recommended movies with IMDb-style ratings above this threshold."
)

# -------------------- MAIN UI --------------------
st.markdown(
    """
    <h1 style='text-align:center;color:#f5c518;'>🎬 Movie Recommendation System</h1>
    <p style='text-align:center;color:#9aa0a6;'>
    Type a movie name and press Enter or click Recommend
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------- INPUT (SEARCH + ENTER SUPPORT) --------------------
selected_movie = st.selectbox(
    "🎞️ Search or select a movie",
    movies["title"].values,
    index=None,
    placeholder="Type movie name..."
)

recommend_btn = st.button("✨ Recommend Movies")

# -------------------- TRIGGER LOGIC --------------------
if selected_movie and (recommend_btn or st.session_state.get("last_movie") != selected_movie):
    st.session_state["last_movie"] = selected_movie
    selected_data = movies[movies["title"] == selected_movie].iloc[0]

    st.markdown(
        f"""
        <div style='background:#15181f;padding:18px;border-radius:14px;margin-bottom:20px;'>
            <h2 style='margin:0;color:#f5c518;'>Selected Movie</h2>
            <p style='margin:4px 0;color:#f5f5f5;'>Title: <strong>{selected_data.title}</strong></p>
            <p style='margin:4px 0;color:#f5f5f5;'>Status: <strong>{selected_data.status}</strong></p>
            <p style='margin:4px 0;color:#f5f5f5;'>Rating: <strong>{selected_data.vote_average:.1f}</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    titles, ratings = recommend(selected_movie, min_rating)

    if titles:
        st.success("✅ Movies you may like:")
        cols = st.columns(min(5, len(titles)))

        for i in range(len(titles)):
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">🎬 {titles[i]}</div>
                        <div style="color:#f5c518;">Rating: {ratings[i]:.1f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning("⚠️ No recommendations found above the selected rating threshold.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#9aa0a6;'>🚀 Built with Streamlit | ML Placement Project</p>",
    unsafe_allow_html=True
)
