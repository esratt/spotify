
# --- app.py ---
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Model ve veriyi yükle
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
df_combined = pd.read_csv("df_combined.csv")

st.title("🎧 Şarkı Popülerlik Tahmin ve Öneri Sistemi")

menu = st.sidebar.selectbox("Menü Seçin", ["🎯 Popülerlik Tahmini", "🎵 Şarkı Öneri Sistemi"])

if menu == "🎯 Popülerlik Tahmini":
    st.subheader("Şarkı Bilgilerini Girin")

    artist = st.selectbox("🎤 Sanatçı", df_combined["artists"].unique())
    album_options = df_combined[df_combined["artists"] == artist]["album"].unique()
    album = st.selectbox("💿 Albüm", album_options)
    genre = st.selectbox("🎼 Tür", df_combined["genre"].unique())
    explicit = st.radio("⛔ Explicit İçerik Var mı?", ["Evet", "Hayır"])
    duration = st.slider("⏱️ Süre (saniye)", 60, 300, 180)

    # Eşleşen satırı çek
    row = df_combined[
        (df_combined["artists"] == artist) &
        (df_combined["album"] == album) &
        (df_combined["genre"] == genre)
    ].iloc[0]

    # Özellikleri oluştur
    features = pd.DataFrame({
        "artist_count": [row["artist_count"]],
        "album_freq": [row["album_freq"]],
        "duration_min": [duration / 60],
        "genre_popularity": [row["genre_popularity"]],
        "explicit_encoded": [1 if explicit == "Evet" else 0],
        "main_artist_encoded": [row["main_artist_encoded"]]
    })

    # Ölçekle ve tahmin et
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    st.success(f"🎯 Tahmini Popülerlik Skoru: {round(prediction, 2)}")

elif menu == "🎵 Şarkı Öneri Sistemi":
    st.subheader("Bir Şarkı Seçin")
    song_name = st.selectbox("🎵 Şarkı İsmi", df_combined["name"].unique())

    selected_song = df_combined[df_combined["name"] == song_name].iloc[0]
    feature_cols = ["artist_count", "album_freq", "duration_min",
                    "genre_popularity", "explicit_encoded", "main_artist_encoded"]

    similarity = cosine_similarity(
        [selected_song[feature_cols].values],
        df_combined[feature_cols].values
    )[0]

    df_combined["similarity"] = similarity
    recommended = df_combined.sort_values("similarity", ascending=False).iloc[1:6]
    st.write("🎶 Önerilen Şarkılar:")
    st.table(recommended[["name", "artists", "genre", "popularity"]])
