"""
Streamlit demo for the movie recommendation system.
Run with: streamlit run app/app.py
"""
import sys
import os
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from cf_models import train_svd
from recommend import recommend_for_user

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender (Collaborative Filtering)")
st.caption("Built with SVD matrix factorization on MovieLens 100k")


@st.cache_data
def load_data():
    ratings = pd.read_csv("data/ratings_clean.csv")
    movies = pd.read_csv("data/movies_clean.csv")
    return ratings, movies


@st.cache_resource
def load_model(ratings):
    model, _ = train_svd(ratings)
    return model


ratings, movies = load_data()
model = load_model(ratings)

user_ids = sorted(ratings["user_id"].unique())
selected_user = st.selectbox("Select a user ID", user_ids)

if st.button("Get Recommendations"):
    recs = recommend_for_user(selected_user, model, ratings, movies, n=10)
    st.subheader(f"Top picks for User {selected_user}")
    for _, row in recs.iterrows():
        st.write(f"- {row['title']}")

st.divider()
st.subheader("New user? Try popularity-based recommendations")
if st.button("Show popular movies"):
    recs = recommend_for_user(-1, model, ratings, movies, n=10)  # unknown id -> cold start
    for _, row in recs.iterrows():
        st.write(f"- {row['title']}")
