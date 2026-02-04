import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load dataset
df = pd.read_csv("tmdb_5000_movies.csv")

# Keep only relevant columns
df = df[['title', 'overview', 'genres']]

# Fill missing overviews with empty string
df['overview'] = df['overview'].fillna('')

# Convert genres from JSON-like string to plain text
def clean_genres(genres_str):
    genre_list = ast.literal_eval(genres_str)
    genre_names = [g['name'] for g in genre_list]
    return ' '.join(genre_names)

df['genres'] = df['genres'].apply(clean_genres)

# Combine genres + overview into a single column
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
    # Get index of the movie
    idx = df[df['title'] == title].index[0]

    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 5 similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:6]

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return movie titles
    return df['title'].iloc[movie_indices]

# Example usage
print("Movies similar to 'Avatar':")
print(recommend_movie('Avatar'))
