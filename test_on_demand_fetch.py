"""
Test the on-demand movie fetching feature
Tests searching for a movie not in database and auto-fetching from TMDB
"""
import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from app.ml.recommendation_engine import get_recommendation_engine
from app.config import settings

# Enable logging
logging.basicConfig(level=logging.INFO)

async def test_on_demand_fetch():
    """Test fetching a movie that's not in database"""
    
    print("\n" + "="*80)
    print("🧪 TESTING ON-DEMAND MOVIE FETCHING")
    print("="*80)
    
    # Get database connection
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]
    movies_collection = db.movies
    
    # Test movie that's likely NOT in database (a 2023 blockbuster)
    test_movie = "Inception"  # Classic with lots of reviews
    
    # Check if already in database
    existing = await movies_collection.find_one(
        {"title": {"$regex": test_movie, "$options": "i"}}
    )
    
    if existing:
        print(f"\n⚠️  Movie '{test_movie}' already in database.")
        print(f"   Deleting it first to test on-demand fetch...")
        await movies_collection.delete_one({"_id": existing["_id"]})
        print(f"   ✅ Deleted '{existing['title']}'")
    
    print(f"\n🔍 Searching for: '{test_movie}'")
    print(f"   Expected: Not in database → Auto-fetch from TMDB")
    print("\n" + "-"*80)
    
    # Get recommendation engine
    engine = get_recommendation_engine()
    
    # This should trigger auto-fetch from TMDB
    print(f"\n🚀 Requesting recommendations for '{test_movie}'...")
    print("   (This will auto-fetch from TMDB if not found)")
    print("\n" + "-"*80)
    
    recommendations = await engine.get_recommendations(
        movie_name=test_movie,
        max_results=5,
        min_sentiment_score=0.6
    )
    
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    
    if recommendations:
        print(f"\n✅ SUCCESS! Got {len(recommendations)} recommendations")
        
        # Verify movie is now in database
        stored_movie = await movies_collection.find_one(
            {"title": {"$regex": test_movie, "$options": "i"}}
        )
        
        if stored_movie:
            print(f"\n✅ Movie '{stored_movie['title']}' now stored in database!")
            print(f"   • TMDB ID: {stored_movie.get('tmdb_id')}")
            print(f"   • Genres: {', '.join(stored_movie.get('genres', []))}")
            print(f"   • Rating: {stored_movie.get('rating')}/10")
            print(f"   • Reviews: {stored_movie.get('review_count')} (Telugu translated)")
            print(f"   • Auto-fetched: {stored_movie.get('added_on_demand', False)}")
            
            # Show sample Telugu review
            if stored_movie.get('reviews'):
                sample = stored_movie['reviews'][0]
                print(f"\n📝 Sample Telugu Review:")
                print(f"   Telugu: {sample.get('text', '')[:100]}...")
                print(f"   English: {sample.get('original_text', '')[:100]}...")
        
        print(f"\n🎬 Top Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"\n{i}. {rec['title']}")
            print(f"   Score: {rec['recommendation_score']:.2f}")
            print(f"   Genres: {', '.join(rec['genres'])}")
            print(f"   Reason: {rec['reasoning']}")
    else:
        print(f"\n❌ FAILED: No recommendations generated")
        print(f"   Movie might not have reviews on TMDB")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(test_on_demand_fetch())
