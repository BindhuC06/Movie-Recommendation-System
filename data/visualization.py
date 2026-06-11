import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load dataset
df = pd.read_csv("data/tmdb_5000_cleaned.csv")

# Genre columns
genre_cols = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "TV Movie", "Thriller", "War", "Western"
]

# Genre Frequency
genre_counts = df[genre_cols].sum().sort_values(ascending=False)

# Top 8 Genres Correlation
top_genres = list(genre_counts.index[:8])
correlation_matrix = df[top_genres].corr()

# Create Figure
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
sns.set_theme(style="whitegrid")

# -------------------------
# Graph 1: Vote Average Distribution
# -------------------------
sns.histplot(
    df["vote_average"],
    bins=15,
    kde=True,
    color="royalblue",
    ax=axes[0]
)

axes[0].set_title("Distribution of Movie Vote Averages")
axes[0].set_xlabel("Vote Average")
axes[0].set_ylabel("Number of Movies")

# -------------------------
# Graph 2: Genre Frequency
# -------------------------
sns.barplot(
    x=genre_counts.values,
    y=genre_counts.index,
    palette="viridis",
    ax=axes[1]
)

axes[1].set_title("Frequency of Movie Genres")
axes[1].set_xlabel("Number of Movies")
axes[1].set_ylabel("Genre")

# -------------------------
# Graph 3: Genre Correlation
# -------------------------
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    square=True,
    ax=axes[2]
)

axes[2].set_title("Correlation of Top 8 Genres")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
