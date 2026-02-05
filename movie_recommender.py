import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

df = pd.read_csv("tmdb_5000_movies.csv")

df = df[['title', 'overview', 'genres']]

df['overview'] = df['overview'].fillna('')

def clean_genres(genres_str):
    genre_list = ast.literal_eval(genres_str)
    genre_names = [g['name'] for g in genre_list]
    return ' '.join(genre_names)

df['genres'] = df['genres'].apply(clean_genres)

df['content'] = df['overview'] + ' ' + df['genres']

print(df[['title', 'content']].head())

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])
print(tfidf_matrix.shape)

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim[0][:5])

def recommend_movie(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices]

