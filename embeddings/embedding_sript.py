
import numpy as np
import pandas as pd
import faiss
import os
from sentence_transformers import SentenceTransformer

def create_soup(row):
    genres = row['genres_cleaned']
    genres_str = " ".join(genres) if isinstance(genres, list) else str(genres)
    return f"Title: {row['title']}. Genres: {genres_str}. Overview: {row['overview']}"

df_cleaned['content_soup'] = df_cleaned.apply(create_soup, axis=1)
print("Sample content soup string completed:\n", df_cleaned['content_soup'].iloc[0])

# Initialize the lightweight, high-performance MiniLM transformer --384 dimensions
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert the text series into a native Python list for tokenization
sentences = df_cleaned['content_soup'].tolist()

print("\nEncoding movie text profiles into dense vector space of 384 dimentions.....")
embeddings = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True)

# Save the raw mathematical matrix to disk
np.save('embeddings/movie_embeddings.npy', embeddings)
print(f"Embeddings matrix saved successfully. Shape: {embeddings.shape}")


# Get the feature dimensionality (which is 384 for all-MiniLM-L6-v2)
dimension = embeddings.shape[1]

# Instantiate an IndexFlatL2 index matrix, which computes exact Euclidean distance (L2)
index = faiss.IndexFlatL2(dimension)

# Coerce the embeddings array explicitly to float32 (FAISS requirement)
if not embeddings.dtype == np.float32:
    embeddings = embeddings.astype('float32')

# Add vectors to the index space
index.add(embeddings)
print(f"Total vectors indexed inside FAISS: {index.ntotal}")

# Serialize and save the index file directly to disk
faiss.write_index(index, 'embeddings/movies_faiss_flat.index')
print("FAISS index successfully compiled and written to disk.")
