"""Quick database statistics check."""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_stats():
    """Check database statistics."""
    mongo_uri = os.getenv("MONGODB_URI")
    client = AsyncIOMotorClient(mongo_uri)
    db = client["telugu_movie_db"]
    
    movies_count = await db.movies.count_documents({})
    reviews_count = await db.reviews.count_documents({})
    
    print(f"Movies: {movies_count}")
    print(f"Reviews: {reviews_count}")
    
    # Get sample review if exists
    if reviews_count > 0:
        sample = await db.reviews.find_one()
        print(f"\nSample review keys: {list(sample.keys())}")
        print(f"Review text preview: {sample.get('text', '')[:100]}...")
    else:
        print("\nNo reviews found. We skipped Twitter collection due to rate limits.")
        print("For Task 11, we'll need to either:")
        print("1. Wait for Twitter rate limits to reset and collect reviews")
        print("2. Use synthetic data generation for testing")
        print("3. Find alternative review sources")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(check_stats())
