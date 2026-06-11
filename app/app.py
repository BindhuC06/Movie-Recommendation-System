import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

# Set page configuration for a spacious, modern grid layout
st.set_page_config(page_title="TMDB Semantic Recommender", layout="wide")

# ==========================================================
# 1. OPTIMIZATION LAYER: Resource Caching
# ==========================================================

@st.cache_resource
def load_backend_assets():
    """
    Loads the heavy ML model, FAISS index, and dataset exactly once.
    The assets remain cached in server RAM across all user sessions.
    """
    # Load dataset
    df = pd.read_csv('data/tmdb_5000_cleaned.csv')
    # Re-evaluate the string representation of lists back into actual python lists
    import ast
    if 'genres_cleaned' in df.columns:
        df['genres_cleaned'] = df['genres_cleaned'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    
    # Load the semantic transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load the structural FAISS index matrix
    index_path = os.path.join('embeddings', 'movies_faiss_flat.index')
    index = faiss.read_index(index_path)
    
    # Load raw embeddings for direct vector-to-vector lookups ("Find Similar")
    embeddings_path = os.path.join('embeddings', 'movie_embeddings.npy')
    embeddings = np.load(embeddings_path).astype('float32')
    
    return df, model, index, embeddings

# Instantiate the cached components
with st.spinner("Initializing Deep Learning Models & Vector Spaces..."):
    df_cleaned, model, index, raw_embeddings = load_backend_assets()


# ==========================================================
# 2. CORE QUERY ENGINE
# ==========================================================

def search_movies(query_text=None, query_vector_idx=None, top_k=30):
    """
    Dual-mode query execution engine supporting text queries or direct vector index lookups.
    """
    if query_vector_idx is not None:
        # Vector-to-Vector Mode: Extract the exact row vector from memory
        query_vec = np.array([raw_embeddings[query_vector_idx]]).astype('float32')
    elif query_text:
        # Text-to-Vector Mode: Encode natural language text string
        query_vec = model.encode([query_text]).astype('float32')
    else:
        return pd.DataFrame()
        
    # Query FAISS index for neighbors (fetching extra rows to account for downstream filters)
    distances, indices = index.search(query_vec, top_k)
    
    # Extract matching movie rows preserving the distance-sorted rank
    matched_indices = indices[0]
    valid_indices = [idx for idx in matched_indices if 0 <= idx < len(df_cleaned)]
    
    result_df = df_cleaned.iloc[valid_indices].copy()
    # Add distance score metadata to the records
    result_df['vector_distance'] = [distances[0][i] for i, idx in enumerate(matched_indices) if 0 <= idx < len(df_cleaned)]
    return result_df


# ==========================================================
# 3. USER INTERFACE LAYOUT
# ==========================================================

st.title("🎬 TMDB Semantic Discovery Engine")
st.markdown("Explore movies contextually using neural semantic search embeddings.")

# --- Sidebar Controls (Layout Task) ---
st.sidebar.header("Discovery Filters")

# Extract unique genres for the dropdown filtering module
all_genres = sorted(list(set([g for sublist in df_cleaned['genres_cleaned'].dropna() for g in sublist])))
selected_genre = st.sidebar.selectbox("Filter by Genre Category", ["All Domains"] + all_genres)

# Adult content toggle placeholder logic (TMDB datasets often lack explicit adult flags, 
# but we can filter by specific genres like 'Horror'/'Thriller' or simulate the flag check)
adult_toggle = st.sidebar.toggle("Permit Mature Content themes", value=True)

# --- Main Search Input Bar ---
user_query = st.text_input("What kind of cinematic experience are you searching for today?", 
                           placeholder="e.g., A gritty dystopian cyberpunk thriller with heavy philosophical themes")

# State tracking initialization for "Find Similar" buttons
if "target_vector_idx" not in st.session_state:
    st.session_state.target_vector_idx = None
if "last_search_mode" not in st.session_state:
    st.session_state.last_search_mode = "text"

# --- Query Routing Logic ---
results = pd.DataFrame()

# If the user clicked a "Find Similar" button, prioritize that vector state
if st.session_state.target_vector_idx is not None and st.session_state.last_search_mode == "vector":
    with st.spinner("Computing geometric vector distances..."):
        results = search_movies(query_vector_idx=st.session_state.target_vector_idx)
else:
    if user_query:
        with st.spinner("Tokenizing string and running vector retrieval..."):
            results = search_movies(query_text=user_query)
            st.session_state.last_search_mode = "text"

# --- Apply Post-Retrieval Filters ---
if not results.empty:
    # 1. Category Dropdown Filter
    if selected_genre != "All Domains":
        results = results[results['genres_cleaned'].apply(lambda x: selected_genre in x if isinstance(x, list) else False)]
    
    # 2. Simulated Content Pruning based on mature toggle
    if not adult_toggle:
        results = results[~results['genres_cleaned'].apply(lambda x: 'Horror' in x if isinstance(x, list) else False)]


# ==========================================================
# 4. RESULTS RENDER MATRIX (Cards Grid)
# ==========================================================

if not results.empty:
    st.subheader("Discovered Matches")
    
    # Render cards using an adaptive 3-column layout grid
    cols = st.columns(3)
    
    for rank, (idx, row) in enumerate(results.head(12).iterrows()):
        col_target = cols[rank % 3]
        
        with col_target:
            # Build valid TMDB image path. Fallback to placeholder if path is null/missing
            poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}" if pd.notna(row['poster_path']) else "https://via.placeholder.com/500x750?text=No+Poster"
            
            # HTML/CSS injection inside individual Streamlit blocks for styled cards
            st.image(poster_url, use_container_width=True)
            st.markdown(f"### {row['title']}")
            
            # Extract meta indicators
            vote_avg = row.get('vote_average', 'N/A')
            st.markdown(f"**⭐ Score:** {vote_avg} | **Score Radius:** {row.get('vector_distance', 0.0):.4f}")
            
            # Clean display of tags
            genre_tags = ", ".join(row['genres_cleaned']) if isinstance(row['genres_cleaned'], list) else "General"
            st.caption(f"**Genres:** {genre_tags}")
            
            # Text clipping boundary for overview snippet
            overview_snippet = row['overview'] if len(row['overview']) <= 150 else f"{row['overview'][:150]}..."
            st.write(overview_snippet)
            
            # Unique dynamic button key creation utilizing dataframe indices
            if st.button("🔗 Find Similar Movies", key=f"sim_{idx}"):
                st.session_state.target_vector_idx = idx
                st.session_state.last_search_mode = "vector"
                st.rerun()  # Forces streamlit to refresh the application layout with new state variables
                
            st.markdown("---")
elif user_query:
    st.info("Zero records matched your strict sidebar filter criteria. Try expanding your search space.")
else:
    st.info("Please enter a semantic prompt above or select filters to activate the neural retrieval loop.")
