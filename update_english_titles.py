"""
Update existing movies in database to add English titles
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from app.collectors.tmdb_collector import TMDBDataCollector

async def update_movie_titles():
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]
    
    print("🔄 Updating existing movies with English titles...")
    
    # Get all movies
    movies = await db.movies.find({}).to_list(length=100)
    
    tmdb_collector = TMDBDataCollector()
    updated = 0
    
    for movie in movies:
        tmdb_id = movie.get('tmdb_id')
        
        if not tmdb_id:
            print(f"⚠️  Skipping {movie.get('title')} - no TMDB ID")
            continue
        
        # Get movie details from TMDB
        details = await tmdb_collector.get_movie_details(tmdb_id)
        
        if details:
            english_title = details.get('original_title', details.get('title'))
            
            # Update database
            await db.movies.update_one(
                {"_id": movie['_id']},
                {"$set": {"english_title": english_title}}
            )
            
            print(f"✅ Updated: {movie.get('title')} → English: {english_title}")
            updated += 1
        else:
            print(f"❌ Failed to get details for {movie.get('title')}")
    
    print(f"\n✅ Updated {updated}/{len(movies)} movies")
    client.close()

if __name__ == "__main__":
    asyncio.run(update_movie_titles())
