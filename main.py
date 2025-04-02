import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv('movies.csv')

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director', 'title']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']+' '+movies_data['title']

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

user_input = input("What's on your mind: ")

# Function to recommend movies
def recommend_movies(user_input, movies_data, feature_vectors, vectorizer):
    user_input = user_input.lower()
    
    # Check if the input is a movie title
    if user_input in movies_data["title"].str.lower().values:
        movie_index = movies_data[movies_data["title"].str.lower() == user_input].index[0]
        query_vector = feature_vectors[movie_index]  # Use the movie's vector
    else:
        # Treat as a keyword search
        query_vector = vectorizer.transform([user_input])
    
    # Compute similarity
    similarities = cosine_similarity(query_vector, feature_vectors).flatten()
    
    # Get top matches, excluding exact match if a movie was searched
    movie_indices = similarities.argsort()[::-1]
    print("\nRecommended Movies:")
    count = 1
    for i in movie_indices:
        if count > 10:
            break
        print(f"{count}.{movies_data.iloc[i]['title']}")
        count += 1 
        
recommend_movies(user_input, movies_data, feature_vectors, vectorizer)
