"""
Clear all movies and reviews from MongoDB database
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

async def clear_database():
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]
    
    print("🗑️  Clearing database...")
    
    # Delete all movies
    movies_result = await db.movies.delete_many({})
    print(f"✅ Deleted {movies_result.deleted_count} movies")
    
    # Delete all reviews (if separate collection exists)
    try:
        reviews_result = await db.reviews.delete_many({})
        print(f"✅ Deleted {reviews_result.deleted_count} reviews")
    except:
        print("ℹ️  No separate reviews collection")
    
    # Verify
    count = await db.movies.count_documents({})
    print(f"\n📊 Remaining movies in database: {count}")
    
    if count == 0:
        print("✅ Database cleared successfully!")
    else:
        print("⚠️  Some documents still remain")
    
    client.close()

if __name__ == "__main__":
    print("⚠️  WARNING: This will delete ALL data from the database!")
    confirm = input("Type 'YES' to confirm: ")
    
    if confirm == "YES":
        asyncio.run(clear_database())
    else:
        print("❌ Operation cancelled")
