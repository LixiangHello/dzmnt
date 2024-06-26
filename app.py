import streamlit as st
import pandas as pd
from fastai.collab import load_learner


@st.cache_data
def read_pkl():
    return load_learner(r"dongman.pkl")


@st.cache_data
def read_csv():
    return pd.read_csv(r"anime.csv", encoding="gbk")


learn = read_pkl()
# Read the anime data from the CSV file
animes_df = read_csv()


# Function to get a random set of anime for the user to rate
def get_random_anime(n=3):
    return animes_df.sample(n).reset_index(drop=True)


# Function to recommend anime based on user ratings
def recommend_anime(ratings, n=5):
    user_id = max(ratings["user_id"]) + 1
    new_data = pd.DataFrame(
        {
            "user_id": [user_id] * len(animes_df),
            "anime_id": animes_df["anime_id"],
            "anime": animes_df["anime"],
        }
    )
    dls = learn.dls.test_dl(new_data)
    preds, _ = learn.get_preds(dl=dls)
    new_data["rating"] = preds
    sorted_data = new_data.sort_values(by="rating", ascending=False).head(n)
    return sorted_data.merge(animes_df, on="anime_id")


# Initialize the Streamlit session
st.title("我们二次元是这样的")
st.header("动漫推荐")

# Initialize the user ratings DataFrame
if "ratings" not in st.session_state:
    st.session_state["ratings"] = pd.DataFrame(
        columns=["user_id", "anime_id", "rating"]
    )

# Get a random set of anime for the user to rate
if "random_animes" not in st.session_state:
    st.session_state["random_animes"] = get_random_anime()

# Let the user rate the anime
if "ratings" not in st.session_state:
    st.session_state["ratings"] = pd.DataFrame(
        columns=["user_id", "anime_id", "rating"]
    )
st.subheader("请对动漫进行评分(1-5分)")
for i, anime in st.session_state["random_animes"].iterrows():
    rating = st.slider(
        f'推荐动漫[{i+1}]: {anime["anime"]}', 1, 5, key=f"rec_rating_kk{i}"
    )

    if st.button(f"提交评分 动漫{i+1}", key=f"button_{i}"):
        new_rating = pd.DataFrame(
            {"user_id": [1], "anime_id": [anime["anime_id"]], "rating": [rating]}
        )
        st.session_state["ratings"] = pd.concat(
            [st.session_state["ratings"], new_rating], ignore_index=True
        )
        st.write("评分已提交! ")

# Recommend anime based on the user ratings
if st.button("推荐动漫"):
    recommended_animes = recommend_anime(st.session_state["ratings"])
    st.session_state["recommended_animes"] = recommended_animes
    # 显示推荐结果
    if "recommended_animes" in st.session_state:
        st.subheader("推荐动漫")
        total_rating = 0
        idx = 1
        for i, anime in enumerate(st.session_state["recommended_animes"]):
            # rating = st.slider(
            #     f'推荐动漫[{i+1}]: {anime["anime"]}', 1, 5, key=f"rec_rating_{i}"
            # )
            rating = st.slider(
                f'1', 1, 5, key=f"rec_rating_{idx}"
            )
            total_rating += rating
            idx += 1
        if len(st.session_state["recommended_animes"]) > 0:
            satisfaction = total_rating / len(st.session_state["recommended_animes"])
            st.write(f"用户满意度：{satisfaction:.2f} / 5")
