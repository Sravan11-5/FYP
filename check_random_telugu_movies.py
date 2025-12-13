"""
Check random Telugu movies to see review availability pattern
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
BASE_URL = "https://api.themoviedb.org/3"

def get_random_telugu_movies():
    """Fetch random Telugu movies from 2020-2024"""
    headers = {
        'Authorization': f'Bearer {TMDB_API_KEY}',
        'accept': 'application/json'
    }
    
    params = {
        'with_original_language': 'te',
        'primary_release_date.gte': '2020-01-01',
        'primary_release_date.lte': '2024-12-31',
        'sort_by': 'popularity.desc',
        'page': 1
    }
    
    url = f"{BASE_URL}/discover/movie"
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json().get('results', [])[:20]  # Get top 20
    return []

def check_reviews(tmdb_id):
    """Quick check for review count"""
    headers = {
        'Authorization': f'Bearer {TMDB_API_KEY}',
        'accept': 'application/json'
    }
    
    url = f"{BASE_URL}/movie/{tmdb_id}/reviews"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return len(response.json().get('results', []))
    return 0

def main():
    print("\n" + "="*80)
    print("🔍 CHECKING 20 RANDOM TELUGU MOVIES (2020-2024)")
    print("="*80)
    
    movies = get_random_telugu_movies()
    
    if not movies:
        print("❌ Failed to fetch movies")
        return
    
    results = []
    
    for movie in movies:
        title = movie.get('title', 'Unknown')
        original_title = movie.get('original_title', '')
        tmdb_id = movie.get('id')
        year = movie.get('release_date', '')[:4]
        popularity = movie.get('popularity', 0)
        vote_count = movie.get('vote_count', 0)
        
        review_count = check_reviews(tmdb_id)
        
        results.append({
            'title': title,
            'original_title': original_title,
            'year': year,
            'popularity': popularity,
            'vote_count': vote_count,
            'reviews': review_count
        })
        
        status = "✅" if review_count > 0 else "❌"
        print(f"{status} {title[:40]:<40} | Year: {year} | Popularity: {popularity:.1f} | Votes: {vote_count} | Reviews: {review_count}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 STATISTICAL SUMMARY")
    print("="*80)
    
    total_movies = len(results)
    movies_with_reviews = sum(1 for r in results if r['reviews'] > 0)
    movies_without_reviews = total_movies - movies_with_reviews
    total_reviews = sum(r['reviews'] for r in results)
    avg_popularity = sum(r['popularity'] for r in results) / total_movies
    avg_votes = sum(r['vote_count'] for r in results) / total_movies
    
    print(f"\n📈 Out of {total_movies} popular Telugu movies (2020-2024):")
    print(f"   ✅ Movies WITH reviews: {movies_with_reviews} ({movies_with_reviews/total_movies*100:.1f}%)")
    print(f"   ❌ Movies WITHOUT reviews: {movies_without_reviews} ({movies_without_reviews/total_movies*100:.1f}%)")
    print(f"   📝 Total reviews: {total_reviews}")
    print(f"   ⭐ Average popularity: {avg_popularity:.1f}")
    print(f"   🗳️ Average vote count: {avg_votes:.0f}")
    
    if movies_with_reviews > 0:
        avg_reviews_per_movie = total_reviews / movies_with_reviews
        print(f"   📊 Average reviews per movie (that has reviews): {avg_reviews_per_movie:.1f}")
    
    print("\n💡 CONCLUSION:")
    if movies_without_reviews / total_movies > 0.8:
        print("   Most Telugu movies DON'T have reviews on TMDB!")
        print("   This is a DATA AVAILABILITY issue, not a technical problem.")
    elif movies_with_reviews / total_movies > 0.5:
        print("   Many Telugu movies DO have reviews!")
        print("   Your script might need adjustment.")
    else:
        print("   Review availability is mixed for Telugu movies.")

if __name__ == "__main__":
    main()
