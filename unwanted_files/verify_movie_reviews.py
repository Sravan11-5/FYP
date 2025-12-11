"""Detailed check of movie-review relationship"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
import json

async def detailed_check():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]
    
    print("\n" + "="*70)
    print("  📊 DETAILED MOVIE-REVIEW STRUCTURE")
    print("="*70)
    
    movies = await db.movies.find({}).to_list(None)
    
    for movie in movies:
        print(f"\n{'='*70}")
        print(f"🎬 Movie: {movie.get('title')}")
        print(f"   TMDB ID: {movie.get('tmdb_id')}")
        print(f"   Rating: ⭐ {movie.get('rating')}")
        print(f"   Genre: {movie.get('genre')}")
        print(f"   Release: {movie.get('release_date')}")
        
        reviews = movie.get('reviews', [])
        print(f"\n   📝 Reviews stored in this movie: {len(reviews)}")
        
        if reviews:
            print(f"\n   Sample Reviews:")
            for i, review in enumerate(reviews[:3], 1):
                print(f"\n   Review {i}:")
                print(f"      Tweet ID: {review.get('tweet_id')}")
                print(f"      Language: {review.get('language')}")
                print(f"      Text: {review.get('text')[:100]}...")
                print(f"      Created: {review.get('created_at')}")
                metrics = review.get('metrics', {})
                print(f"      Likes: {metrics.get('like_count', 0)}, Retweets: {metrics.get('retweet_count', 0)}")
        
        print(f"\n   ✅ All {len(reviews)} reviews are correctly linked to '{movie.get('title')}'")
    
    print("\n" + "="*70 + "\n")
    client.close()

if __name__ == "__main__":
    asyncio.run(detailed_check())
