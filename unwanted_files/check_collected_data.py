"""Quick check of collected real data"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

async def check_data():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]
    
    print("\n" + "="*70)
    print("  📊 COLLECTED REAL DATA STATUS")
    print("="*70)
    
    # Get all movies
    movies = await db.movies.find({}).to_list(None)
    
    print(f"\n✅ Total movies in database: {len(movies)}")
    
    total_reviews = 0
    for movie in movies:
        reviews = movie.get('reviews', [])
        review_count = len(reviews)
        total_reviews += review_count
        print(f"\n📽️ {movie.get('title')} (⭐ {movie.get('rating', 0)})")
        print(f"   Genre: {movie.get('genre', 'Unknown')}")
        print(f"   Reviews: {review_count}")
        if review_count > 0:
            print(f"   Sample: {reviews[0].get('text', '')[:80]}...")
    
    print("\n" + "="*70)
    print(f"📊 TOTAL REVIEWS COLLECTED: {total_reviews}")
    print("="*70)
    
    client.close()

if __name__ == "__main__":
    asyncio.run(check_data())
