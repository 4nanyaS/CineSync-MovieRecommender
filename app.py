from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

app = Flask(__name__)
CORS(app)  # allow frontend to talk to backend

# ---------- LOAD DATA ----------
df = pd.read_csv("tmdb_5000_movies.csv")
df = df[['title', 'overview', 'genres']]
df['overview'] = df['overview'].fillna('')

def clean_genres(genres_str):
    genre_list = ast.literal_eval(genres_str)
    return " ".join([g['name'] for g in genre_list])

df['genres'] = df['genres'].apply(clean_genres)
df['content'] = df['overview'] + " " + df['genres']

# ---------- TF-IDF ----------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---------- RECOMMENDER ----------
def recommend_movie(title):
    title = title.lower()

    if title not in df['title'].str.lower().values:
        return ["Movie not found "]

    idx = df[df['title'].str.lower() == title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# ---------- API ----------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    movie_name = data.get("movie_name", "")
    recommendations = recommend_movie(movie_name)

    return jsonify({
        "recommendations": recommendations
    })

if __name__ == "__main__":
    app.run(debug=True)
