"""Check database format and structure."""
import asyncio
import json
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

async def check_format():
    """Check one movie's format."""
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]
    
    # Get one movie
    movie = await db.movies.find_one()
    
    if movie:
        print("="*60)
        print("MOVIE STRUCTURE:")
        print("="*60)
        print(json.dumps(movie, indent=2, default=str))
        
        print("\n" + "="*60)
        print("REVIEW STRUCTURE:")
        print("="*60)
        if movie.get("reviews"):
            print(json.dumps(movie["reviews"][0], indent=2, default=str))
        
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        print(f"Title: {movie.get('title')}")
        print(f"Original Title: {movie.get('original_title')}")
        print(f"Genre: {movie.get('genre')}")
        print(f"Review Count: {movie.get('review_count')}")
        print(f"Has Reviews: {movie.get('has_reviews')}")
        
        if movie.get("reviews"):
            review = movie["reviews"][0]
            print(f"\nFirst Review:")
            print(f"  - Text Length: {len(review.get('text', ''))} chars")
            print(f"  - Language: {review.get('language', 'N/A')}")
            print(f"  - Has Original Text: {'original_text' in review}")
            print(f"  - Source: {review.get('source', 'N/A')}")
    else:
        print("❌ No movies in database!")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(check_format())
