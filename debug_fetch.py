"""
Debug test to see what's happening during fetch
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.collectors.tmdb_collector import TMDBDataCollector
from app.config import settings

async def debug_fetch():
    print("\n🔍 Testing TMDB Search for Inception...")
    
    tmdb = TMDBDataCollector()
    
    # Search
    results = await tmdb.search_movie("Inception", language="en")
    print(f"\n✅ Found {len(results)} results")
    
    if results:
        movie = results[0]
        tmdb_id = movie['id']
        print(f"\nMovie: {movie['title']} (ID: {tmdb_id})")
        
        # Get details
        details = await tmdb.get_movie_details(tmdb_id)
        if details:
            print(f"✅ Got movie details")
            print(f"   Genres: {[g['name'] for g in details.get('genres', [])]}")
        
        # Get reviews
        print(f"\n🔍 Fetching reviews...")
        translated = await tmdb.get_reviews_and_translate(tmdb_id, max_reviews=5)
        print(f"✅ Got {len(translated)} Telugu-translated reviews")
        
        if translated:
            print(f"\n📝 Sample review:")
            print(f"   Telugu: {translated[0]['text'][:100]}...")
            print(f"   English: {translated[0]['original_text'][:100]}...")

if __name__ == "__main__":
    asyncio.run(debug_fetch())
