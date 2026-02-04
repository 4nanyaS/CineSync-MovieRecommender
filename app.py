from flask import Flask, request, jsonify, send_from_directory  # <-- Added send_from_directory
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os  # <-- Added for Render deployment

app = Flask(__name__, static_folder='.')  # <-- tell Flask current folder has static files
CORS(app)  # allow frontend to talk to backend

# ---------- FRONTEND ROUTE ----------
@app.route('/')
def index():
    # Serve index.html from current folder
    return send_from_directory('.', 'index.html')

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
        return ["Movie not found"]

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

# ---------- RUN ----------
if __name__ == "__main__":
    # Use host='0.0.0.0' for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
