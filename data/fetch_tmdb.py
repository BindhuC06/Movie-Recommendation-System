# Script to fetch data from TMDB API
import os
import json
import time
import requests

def fetch_tmdb_data(api_key, target_count=5000):
    """
    Fetches top-rated or popular movies from TMDB until target_count is reached.
    Extracts: title, overview, genres, poster_path, vote_average
    """
    # 1. Fetch the genre mapping list first (TMDB returns genre IDs by default)
    genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}&language=en-US"
    genre_response = requests.get(genre_url)
    
    if genre_response.status_code != 200:
        print(f"Error fetching genres: {genre_response.status_code}")
        return
        
    genres_data = genre_response.json().get('genres', [])
    genre_map = {g['id']: g['name'] for g in genres_data}

    movies_list = []
    page = 1
    results_per_page = 20
    pages_needed = (target_count // results_per_page) + 1

    print(f"Starting data collection. Target: ~{target_count} movies ({pages_needed} pages)...")

    # 2. Iterate through TMDB pages
    while len(movies_list) < target_count and page <= pages_needed:
        # Using the discover endpoint to get a broad mix of popular English movies
        url = (
            f"https://api.themoviedb.org/3/discover/movie?"
            f"api_key={api_key}&language=en-US&sort_by=popularity.desc"
            f"&page={page}&include_adult=false"
        )
        
        try:
            response = requests.get(url)
            if response.status_code == 429:
                print("Rate limit hit. Sleeping for 5 seconds...")
                time.sleep(5)
                continue
            elif response.status_code != 200:
                print(f"Failed to fetch page {page}. Status code: {response.status_code}")
                break
                
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                print("No more results available from API.")
                break

            # 3. Parse out only the requested features
            for item in results:
                # Map genre IDs to their string names
                item_genres = [genre_map.get(gid) for gid in item.get('genre_ids', []) if gid in genre_map]
                
                movie_details = {
                    'title': item.get('title'),
                    'overview': item.get('overview'),
                    'genres': item_genres,
                    'poster_path': item.get('poster_path'),
                    'vote_average': item.get('vote_average')
                }
                movies_list.append(movie_details)
                
                # Break early if we hit our target mid-page
                if len(movies_list) >= target_count:
                    break
                    
            if page % 25 == 0 or page == pages_needed:
                print(f"Progress: Fetched {len(movies_list)} movies (Page {page}/{pages_needed})")
                
            page += 1
            time.sleep(0.1) # Small polite delay between API hits
            
        except Exception as e:
            print(f"An error occurred on page {page}: {e}")
            break

    # 4. Save the collected data to the data/ directory
    os.makedirs('data', exist_ok=True)
    output_path = 'data/raw_movies.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(movies_list, f, ensure_ascii=False, indent=4)
        
    print(f"\nSuccess! Successfully saved {len(movies_list)} records to '{output_path}'")

if __name__ == "__main__":
    # Pull the API key from the execution environment securely
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")
    
    if not TMDB_API_KEY:
        print("ERROR: TMDB_API_KEY environment variable not set.")
        print("Please set it in your environment or Colab userdata before running.")
    else:
        fetch_tmdb_data(api_key=TMDB_API_KEY, target_count=5000)