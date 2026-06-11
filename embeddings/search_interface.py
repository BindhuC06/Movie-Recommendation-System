
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

df_cleaned = pd.read_csv('tmdb_5000_cleaned.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')

artifacts_dir = 'embeddings'
index_path = os.path.join(artifacts_dir, 'movies_faiss_flat.index')

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    print(f"Successfully loaded FAISS index. Total movies indexed: {index.ntotal}")
else:
    raise FileNotFoundError(f"Could not find FAISS index file at {index_path}")


def recommend_movies(user_query, top_k=5):
    """
    Vectorizes a natural language query and retrieves the top_k most similar movies.
    """
    # 1. Convert the raw user string into a 384-dimensional dense vector
    query_vector = model.encode([user_query]).astype('float32')
    
    # 2. Query the FAISS index for the 'k' nearest spatial neighbors using L2 distance
    distances, indices = index.search(query_vector, top_k)
    
    print(f"\n🎯 Top {top_k} Recommendations for: '{user_query}'")
    print("-" * 60)
    
    # 3.display the results
    for rank in range(top_k):
        movie_idx = indices[0][rank]
        distance_score = distances[0][rank]
        
        # Guard against index out of bounds errors
        if movie_idx < len(df_cleaned):
            movie_data = df_cleaned.iloc[movie_idx]
            print(f"{rank + 1}. 🎬 {movie_data['title']}")
            print(f"   Genres: {movie_data['genres_cleaned']}")
            print(f"   Overview: {movie_data['overview'][:120]}...")
            print(f"   [Vector Distance: {distance_score:.4f}]\n")

# sample output : 
# recommend_movies("scary psychological horror movie set in space", top_k=3)
# recommend_movies("heartwarming animated family comedy", top_k=3)
# 🎯 Top 3 Recommendations for: 'scary psychological horror movie set in space'
# ------------------------------------------------------------
# 1. 🎬 Space/Time
#    Genres: ['Science Fiction', 'Action', 'Thriller']
#    Overview: After a fatal test shuts down their project, a disgraced team of scientists enters the criminal underworld to rebuild a ...
#    [Vector Distance: 0.8395]

# 2. 🎬 The Astronaut
#    Genres: ['Horror', 'Science Fiction', 'Thriller']
#    Overview: After returning from her first space mission, astronaut Sam Walker is placed under NASA’s care at a high security house ...
#    [Vector Distance: 0.8702]

# 3. 🎬 Scary Movie 2
#    Genres: ['Comedy']
#    Overview: While the original parodied slasher flicks like Scream, Keenen Ivory Wayans's sequel to Scary Movie takes comedic aim at...
#    [Vector Distance: 0.8815]


# 🎯 Top 3 Recommendations for: 'heartwarming animated family comedy'
# ------------------------------------------------------------
# 1. 🎬 A Goofy Movie
#    Genres: ['Adventure', 'Comedy', 'Romance', 'Animation', 'Family']
#    Overview: Goofy’s teenage son Max is desperate to impress his crush and fit in at school. After well-meaning but ignorant Goofy su...
#    [Vector Distance: 0.7854]

# 2. 🎬 Onward
#    Genres: ['Adventure', 'Animation', 'Comedy', 'Family', 'Fantasy']
#    Overview: In a suburban fantasy world, two teenage elf brothers embark on an extraordinary quest to discover if there is still a l...
#    [Vector Distance: 0.8091]

# 3. 🎬 Klaus
#    Genres: ['Animation', 'Family', 'Comedy']
#    Overview: A selfish postman and a reclusive toymaker form an unlikely friendship, delivering joy to a cold, dark town that despera...
#    [Vector Distance: 0.8178]
