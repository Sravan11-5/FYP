"""
Quick Movie Collection (No Reviews)
Fetches movies from TMDB and stores them WITHOUT reviews
Reviews can be added later when rate limit resets
"""
import asyncio
import logging
from typing import List, Dict
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from app.collectors.tmdb_collector import TMDBDataCollector
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TMDB Genre IDs
TELUGU_GENRES = {
    "Action": 28,
    "Comedy": 35,
    "Drama": 18,
    "Romance": 10749,
    "Thriller": 53,
    "Crime": 80,
    "Family": 10751,
    "Adventure": 12
}


class QuickMovieCollector:
    """Quickly collects movies from TMDB"""
    
    def __init__(self):
        self.tmdb = TMDBDataCollector()
        self.db_client = None
        self.db = None
        
    async def connect_db(self):
        """Connect to MongoDB"""
        self.db_client = AsyncIOMotorClient(settings.MONGODB_URL)
        self.db = self.db_client[settings.MONGODB_DB_NAME]
        logger.info("✅ Connected to MongoDB")
        
    async def close_db(self):
        """Close MongoDB connection"""
        if self.db_client:
            self.db_client.close()
    
    async def collect_movies_by_genre(
        self,
        genre_name: str,
        genre_id: int,
        min_rating: float = 6.0,
        max_movies: int = 10
    ):
        """Collect and store movies for a genre"""
        logger.info(f"\n📂 Collecting {genre_name} movies...")
        
        movies = await self.tmdb.discover_movies_by_genre(
            genre_ids=[genre_id],
            min_vote_average=min_rating,
            max_results=max_movies
        )
        
        stored_count = 0
        for movie in movies:
            movie_doc = {
                "tmdb_id": movie.get('id'),
                "title": movie.get('title'),
                "original_title": movie.get('original_title'),
                "overview": movie.get('overview'),
                "release_date": movie.get('release_date'),
                "genre": genre_name,
                "rating": movie.get('vote_average'),
                "vote_count": movie.get('vote_count'),
                "popularity": movie.get('popularity'),
                "poster_path": movie.get('poster_path'),
                "backdrop_path": movie.get('backdrop_path'),
                "language": movie.get('original_language'),
                "reviews": [],
                "review_count": 0,
                "needs_reviews": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            await self.db.movies.update_one(
                {"tmdb_id": movie.get('id')},
                {"$set": movie_doc},
                upsert=True
            )
            stored_count += 1
            logger.info(f"   ✅ Stored: {movie.get('title')} (Rating: {movie.get('vote_average')})")
        
        logger.info(f"✅ Stored {stored_count} {genre_name} movies")
    
    async def collect_all(
        self,
        movies_per_genre: int = 10,
        min_rating: float = 6.0
    ):
        """Collect movies for all genres"""
        print("\n" + "="*70)
        print("  🎬 QUICK MOVIE COLLECTION (NO REVIEWS)")
        print("="*70)
        print(f"\n📊 Configuration:")
        print(f"   • Movies per genre: {movies_per_genre}")
        print(f"   • Minimum rating: {min_rating}")
        print(f"   • Total movies: ~{len(TELUGU_GENRES) * movies_per_genre}")
        
        await self.connect_db()
        
        try:
            for genre_name, genre_id in TELUGU_GENRES.items():
                await self.collect_movies_by_genre(
                    genre_name=genre_name,
                    genre_id=genre_id,
                    min_rating=min_rating,
                    max_movies=movies_per_genre
                )
                await asyncio.sleep(1)
        finally:
            await self.close_db()
        
        print("\n" + "="*70)
        print("  ✅ MOVIES COLLECTED!")
        print("  💡 Run 'add_reviews_to_movies.py' later to add reviews")
        print("="*70)


async def main():
    collector = QuickMovieCollector()
    await collector.collect_all(
        movies_per_genre=10,
        min_rating=6.0
    )


if __name__ == "__main__":
    asyncio.run(main())
