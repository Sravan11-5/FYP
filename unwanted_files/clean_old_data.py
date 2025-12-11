"""Delete old static/synthetic data, keep only real Twitter reviews"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

async def clean_database():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]
    
    print("\n" + "="*70)
    print("  🧹 CLEANING OLD DATA")
    print("="*70)
    
    # Get all movies
    all_movies = await db.movies.find({}).to_list(None)
    print(f"\n📊 Total movies in database: {len(all_movies)}")
    
    # Identify movies to delete (0 reviews or no review_count field)
    to_delete = []
    to_keep = []
    
    for movie in all_movies:
        reviews = movie.get('reviews', [])
        review_count = len(reviews)
        
        if review_count == 0:
            to_delete.append(movie)
        else:
            to_keep.append(movie)
    
    print(f"\n🗑️  Movies to DELETE (0 reviews): {len(to_delete)}")
    for m in to_delete:
        print(f"   - {m.get('title')} (⭐ {m.get('rating', 0)})")
    
    print(f"\n✅ Movies to KEEP (with reviews): {len(to_keep)}")
    for m in to_keep:
        reviews = m.get('reviews', [])
        print(f"   - {m.get('title')} (⭐ {m.get('rating', 0)}) - {len(reviews)} reviews")
    
    # Confirm deletion
    print("\n" + "="*70)
    confirm = input("❓ Delete old movies with 0 reviews? (yes/no): ")
    
    if confirm.lower() == 'yes':
        # Delete movies with 0 reviews
        delete_ids = [m['_id'] for m in to_delete]
        result = await db.movies.delete_many({'_id': {'$in': delete_ids}})
        
        print(f"\n✅ Deleted {result.deleted_count} old movies")
        print(f"✅ Kept {len(to_keep)} movies with real reviews")
        
        # Show final status
        remaining = await db.movies.count_documents({})
        print(f"\n📊 Final database status: {remaining} movies")
    else:
        print("\n❌ Cancelled - no data deleted")
    
    print("="*70 + "\n")
    client.close()

if __name__ == "__main__":
    asyncio.run(clean_database())
