"""
Smart Data Collection System
Fetches movies from TMDB by genre and collects 5 Twitter reviews per movie
Stores data organized by genre and rating
"""
import asyncio
import logging
from typing import List, Dict
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from app.collectors.tmdb_collector import TMDBDataCollector
from app.collectors.twitter_collector import TwitterDataCollector
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TMDB Genre IDs for Telugu Cinema
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


class SmartDataCollector:
    """Collects and stores movie data with reviews by genre"""
    
    def __init__(self):
        self.tmdb = TMDBDataCollector()
        self.twitter = TwitterDataCollector()
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
            logger.info("✅ Closed MongoDB connection")
    
    async def fetch_movies_by_genre(
        self,
        genre_name: str,
        genre_id: int,
        min_rating: float = 6.0,
        max_movies: int = 10
    ) -> List[Dict]:
        """
        Fetch movies from TMDB for a specific genre
        
        Args:
            genre_name: Name of the genre (Action, Comedy, etc.)
            genre_id: TMDB genre ID
            min_rating: Minimum TMDB rating (0-10)
            max_movies: Maximum number of movies to fetch
            
        Returns:
            List of movie dictionaries
        """
        logger.info(f"\n📽️  Fetching {genre_name} movies (rating >= {min_rating})...")
        
        movies = await self.tmdb.discover_movies_by_genre(
            genre_ids=[genre_id],
            min_vote_average=min_rating,
            max_results=max_movies
        )
        
        logger.info(f"   Found {len(movies)} {genre_name} movies from TMDB")
        return movies
    
    async def collect_reviews_for_movie(
        self,
        movie: Dict,
        max_reviews: int = 5,
        skip_on_rate_limit: bool = True
    ) -> List[Dict]:
        """
        Collect Twitter reviews for a movie (limited to 5 to save tokens)
        
        Args:
            movie: TMDB movie dictionary
            max_reviews: Maximum reviews to collect (default: 5)
            skip_on_rate_limit: If True, skip on rate limit instead of waiting
            
        Returns:
            List of review dictionaries
        """
        movie_title = movie.get('title', movie.get('original_title', 'Unknown'))
        
        logger.info(f"   🐦 Fetching {max_reviews} reviews for: {movie_title}")
        
        try:
            # Add delay to respect rate limits
            await asyncio.sleep(4)
            
            tweets = await self.twitter.search_movie_reviews(
                movie_name=movie_title,
                max_results=max_reviews,
                language="te"
            )
            
            if not tweets:
                logger.warning(f"      ⚠️  No tweets found (might be rate limited)")
                return []
            
            # Format reviews
            reviews = []
            for tweet in tweets:
                review = {
                    "tweet_id": tweet.get('id'),
                    "text": tweet.get('text'),
                    "created_at": tweet.get('created_at'),
                    "language": tweet.get('lang'),
                    "author_id": tweet.get('author_id'),
                    "metrics": tweet.get('public_metrics', {}),
                    "collected_at": datetime.utcnow()
                }
                reviews.append(review)
            
            logger.info(f"      ✅ Collected {len(reviews)} reviews")
            return reviews
            
        except asyncio.CancelledError:
            logger.warning(f"      ⚠️  Rate limit hit, skipping this movie")
            return []
        except Exception as e:
            logger.error(f"      ❌ Error collecting reviews: {e}")
            return []
    
    async def store_movie_with_reviews(
        self,
        movie: Dict,
        reviews: List[Dict],
        genre_name: str
    ):
        """
        Store movie and reviews in MongoDB, organized by genre
        
        Args:
            movie: TMDB movie data
            reviews: List of Twitter reviews
            genre_name: Genre category
        """
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
            "reviews": reviews,
            "review_count": len(reviews),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Upsert (update if exists, insert if new)
        await self.db.movies.update_one(
            {"tmdb_id": movie.get('id')},
            {"$set": movie_doc},
            upsert=True
        )
        
        logger.info(f"      💾 Stored in database (Genre: {genre_name}, Rating: {movie.get('vote_average')})")
    
    async def collect_genre_data(
        self,
        genre_name: str,
        genre_id: int,
        movies_per_genre: int = 5,
        reviews_per_movie: int = 5,
        min_rating: float = 6.0
    ):
        """
        Collect data for one genre
        
        Args:
            genre_name: Genre name
            genre_id: TMDB genre ID
            movies_per_genre: Number of movies to fetch
            reviews_per_movie: Number of reviews per movie (max 5)
            min_rating: Minimum rating filter
        """
        print(f"\n{'='*70}")
        print(f"  📂 COLLECTING: {genre_name.upper()} MOVIES")
        print(f"{'='*70}")
        
        # Fetch movies from TMDB
        movies = await self.fetch_movies_by_genre(
            genre_name=genre_name,
            genre_id=genre_id,
            min_rating=min_rating,
            max_movies=movies_per_genre
        )
        
        if not movies:
            logger.warning(f"⚠️  No movies found for genre: {genre_name}")
            return
        
        # Process each movie
        for i, movie in enumerate(movies, 1):
            movie_title = movie.get('title', 'Unknown')
            movie_rating = movie.get('vote_average', 0)
            
            print(f"\n{i}. Processing: {movie_title} (Rating: {movie_rating})")
            
            # Collect reviews (only 5 to save tokens!)
            reviews = await self.collect_reviews_for_movie(
                movie=movie,
                max_reviews=reviews_per_movie
            )
            
            # Store in database (even if no reviews, store movie metadata)
            await self.store_movie_with_reviews(
                movie=movie,
                reviews=reviews,
                genre_name=genre_name
            )
            
            if not reviews:
                logger.warning(f"      ⚠️  No reviews found, but movie metadata stored")
        
        print(f"\n✅ Completed {genre_name} genre!")
    
    async def collect_all_genres(
        self,
        genres_to_collect: List[str] = None,
        movies_per_genre: int = 5,
        reviews_per_movie: int = 5,
        min_rating: float = 6.0
    ):
        """
        Collect data for multiple genres
        
        Args:
            genres_to_collect: List of genre names (None = all genres)
            movies_per_genre: Movies to fetch per genre
            reviews_per_movie: Reviews per movie (recommended: 5)
            min_rating: Minimum TMDB rating
        """
        if genres_to_collect is None:
            genres_to_collect = list(TELUGU_GENRES.keys())
        
        print("\n" + "="*70)
        print("  🎬 SMART DATA COLLECTION SYSTEM")
        print("="*70)
        print(f"\n📊 Configuration:")
        print(f"   • Genres: {', '.join(genres_to_collect)}")
        print(f"   • Movies per genre: {movies_per_genre}")
        print(f"   • Reviews per movie: {reviews_per_movie} (saves tokens!)")
        print(f"   • Minimum rating: {min_rating}")
        print(f"   • Total movies: {len(genres_to_collect) * movies_per_genre}")
        print(f"   • Total reviews: ~{len(genres_to_collect) * movies_per_genre * reviews_per_movie}")
        
        # Connect to database
        await self.connect_db()
        
        try:
            for genre_name in genres_to_collect:
                if genre_name not in TELUGU_GENRES:
                    logger.warning(f"⚠️  Unknown genre: {genre_name}, skipping")
                    continue
                
                genre_id = TELUGU_GENRES[genre_name]
                
                await self.collect_genre_data(
                    genre_name=genre_name,
                    genre_id=genre_id,
                    movies_per_genre=movies_per_genre,
                    reviews_per_movie=reviews_per_movie,
                    min_rating=min_rating
                )
                
                # Pause between genres to respect rate limits
                if genre_name != genres_to_collect[-1]:  # Not the last genre
                    logger.info("\n⏳ Waiting 15 seconds before next genre (rate limit protection)...")
                    await asyncio.sleep(15)
        
        finally:
            await self.close_db()
        
        print("\n" + "="*70)
        print("  ✅ DATA COLLECTION COMPLETE!")
        print("="*70)
    
    async def show_collection_stats(self):
        """Display statistics about collected data"""
        await self.connect_db()
        
        try:
            total_movies = await self.db.movies.count_documents({})
            
            print("\n" + "="*70)
            print("  📊 DATABASE STATISTICS")
            print("="*70)
            
            print(f"\n💾 Total Movies: {total_movies}")
            
            # Stats by genre
            pipeline = [
                {"$group": {
                    "_id": "$genre",
                    "count": {"$sum": 1},
                    "avg_rating": {"$avg": "$rating"},
                    "total_reviews": {"$sum": "$review_count"}
                }},
                {"$sort": {"count": -1}}
            ]
            
            genre_stats = await self.db.movies.aggregate(pipeline).to_list(None)
            
            if genre_stats:
                print("\n📂 By Genre:")
                for stat in genre_stats:
                    genre = stat['_id']
                    count = stat['count']
                    avg_rating = stat['avg_rating']
                    reviews = stat['total_reviews']
                    print(f"   • {genre}: {count} movies, "
                          f"avg rating {avg_rating:.1f}, "
                          f"{reviews} reviews")
            
            # Recent movies
            recent = await self.db.movies.find().sort("created_at", -1).limit(5).to_list(5)
            
            if recent:
                print("\n🆕 Recently Added:")
                for movie in recent:
                    print(f"   • {movie['title']} ({movie['genre']}) - "
                          f"{movie['review_count']} reviews")
        
        finally:
            await self.close_db()


async def main():
    """Main execution"""
    collector = SmartDataCollector()
    
    print("\n💡 Choose collection mode:")
    print("1. Quick Test (3 genres, 3 movies each, 5 reviews per movie)")
    print("2. Medium Collection (5 genres, 5 movies each, 5 reviews per movie)")
    print("3. Full Collection (All 8 genres, 10 movies each, 5 reviews per movie)")
    
    choice = input("\nEnter choice (1/2/3) or press Enter for Quick Test: ").strip()
    
    if choice == "3":
        # Full collection
        await collector.collect_all_genres(
            movies_per_genre=10,
            reviews_per_movie=5,
            min_rating=6.0
        )
    elif choice == "2":
        # Medium collection
        await collector.collect_all_genres(
            genres_to_collect=["Action", "Comedy", "Drama", "Romance", "Thriller"],
            movies_per_genre=5,
            reviews_per_movie=5,
            min_rating=6.0
        )
    else:
        # Quick test (default)
        await collector.collect_all_genres(
            genres_to_collect=["Action", "Comedy", "Drama"],
            movies_per_genre=3,
            reviews_per_movie=5,
            min_rating=6.5
        )
    
    # Show statistics
    await collector.show_collection_stats()


if __name__ == "__main__":
    print("\n🚀 Starting Smart Data Collection...")
    asyncio.run(main())
