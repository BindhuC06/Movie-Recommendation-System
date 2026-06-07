
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

df=pd.read_json('/content/drive/MyDrive/Movie_Recommendation/Movie-Recommendation-System/data/raw_movies.json')

# Drop rows with missing overviews
df.dropna(subset=['overview'], inplace=True)

# Deduplicate based on movie title
df.drop_duplicates(subset=['title'], keep='first', inplace=True)


# Parse th Genre Strings
def parse_python_list_string(genre_str):
    if isinstance(genre_str, list):
        return genre_str
    try:
        return ast.literal_eval(genre_str)
    except (ValueError, SyntaxError):
        return []

# Apply the safe AST parsing function first
df['genres_cleaned'] = df['genres'].apply(parse_python_list_string)


#Vectorize/One-Hot Encode Genres
mlb = MultiLabelBinarizer()
# Now mlb fits on successfully populated lists (e.g., ['Horror', 'Mystery'])
genre_encoded = mlb.fit_transform(df['genres_cleaned'])

#binary DataFrame with correct genre names as columns
df_genres = pd.DataFrame(genre_encoded, columns=mlb.classes_, index=df.index)

# Combine the original columns with the new one-hot encoded features
df_cleaned = pd.concat([df, df_genres], axis=1)

print("Final Cleaned Features Array:")
print(df_cleaned.columns.tolist())

# Save the structurally sound dataset to disk
df_cleaned.to_csv('tmdb_5000_cleaned.csv', index=False)
print("\nCleaned dataset successfully saved to disk.")
