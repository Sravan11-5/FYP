"""
Smart Movie & Review Collection System
Phase 1: Fetch 100-150 movies from TMDB by genre
Phase 2: Collect 10 Telugu Twitter reviews per movie
Phase 3: Store organized by genre and rating in MongoDB
"""
import asyncio
import logging
from typing import List, Dict
from datetime import datetime
import time
from motor.motor_asyncio import AsyncIOMotorClient
from app.collectors.tmdb_collector import TMDBDataCollector
from app.collectors.twitter_collector import TwitterDataCollector
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Telugu Movie Genres (TMDB Genre IDs)
TELUGU_GENRES = {
    "Action": 28,
    "Adventure": 12,
    "Comedy": 35,
    "Crime": 80,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "Horror": 27,
    "Mystery": 9648,
    "Romance": 10749,
    "Thriller": 53,
    "War": 10752
}


class SmartMovieCollector:
    """Collects movies from TMDB and reviews from Twitter"""
    
    def __init__(self):
        self.tmdb = TMDBDataCollector()
        self.twitter = TwitterDataCollector()
        self.db_client = None
        self.db = None
        
        # Rate limit tracking
        self.api_calls_made = 0
        self.window_start_time = time.time()
        self.max_calls_per_window = 15  # FREE tier limit
        self.window_duration = 900  # 15 minutes in seconds
        
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
    
    async def check_rate_limit(self):
        """Check and enforce Twitter API rate limits"""
        current_time = time.time()
        elapsed = current_time - self.window_start_time
        
        # Reset window if 15 minutes passed
        if elapsed >= self.window_duration:
            self.api_calls_made = 0
            self.window_start_time = current_time
            logger.info("🔄 Rate limit window reset")
            return
        
        # If we hit the limit, wait
        if self.api_calls_made >= self.max_calls_per_window:
            wait_time = self.window_duration - elapsed
            logger.warning(
                f"⏳ Rate limit reached! Waiting {wait_time/60:.1f} minutes..."
            )
            await asyncio.sleep(wait_time + 10)  # Add 10 sec buffer
            
            # Reset after waiting
            self.api_calls_made = 0
            self.window_start_time = time.time()
            logger.info("✅ Rate limit window reset after wait")
    
    async def fetch_movies_by_genre(
        self,
        genre_name: str,
        genre_id: int,
        movies_per_genre: int = 15,
        min_rating: float = 6.0
    ) -> List[Dict]:
        """
        Fetch movies from TMDB for a specific genre
        
        Args:
            genre_name: Genre name (Action, Drama, etc.)
            genre_id: TMDB genre ID
            movies_per_genre: Number of movies to fetch
            min_rating: Minimum TMDB rating
            
        Returns:
            List of movie dictionaries
        """
        logger.info(f"\n📂 Fetching {movies_per_genre} {genre_name} movies (rating >= {min_rating})...")
        
        movies = await self.tmdb.discover_movies_by_genre(
            genre_ids=[genre_id],
            min_vote_average=min_rating,
            max_results=movies_per_genre
        )
        
        logger.info(f"   ✅ Found {len(movies)} {genre_name} movies from TMDB")
        return movies
    
    async def collect_twitter_reviews(
        self,
        movie_title: str,
        max_reviews: int = 10
    ) -> List[Dict]:
        """
        Collect Telugu Twitter reviews for a movie
        
        Args:
            movie_title: Movie title to search
            max_reviews: Number of reviews to collect (default: 10)
            
        Returns:
            List of review dictionaries
        """
        try:
            # Check rate limit before making call
            await self.check_rate_limit()
            
            logger.info(f"      🐦 Fetching {max_reviews} Telugu reviews...")
            
            # Add delay between requests (conservative)
            await asyncio.sleep(5)
            
            # Make Twitter API call
            tweets = await self.twitter.search_movie_reviews(
                movie_name=movie_title,
                max_results=max_reviews,
                language="te"
            )
            
            # Increment counter
            self.api_calls_made += 1
            
            if not tweets:
                logger.warning(f"         ⚠️  No reviews found")
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
            
            logger.info(f"         ✅ Collected {len(reviews)} reviews "
                       f"(API calls: {self.api_calls_made}/{self.max_calls_per_window})")
            return reviews
            
        except Exception as e:
            logger.error(f"         ❌ Error: {e}")
            return []
    
    async def store_movie_with_reviews(
        self,
        movie: Dict,
        reviews: List[Dict],
        genre_name: str
    ):
        """
        Store movie and reviews in MongoDB
        
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
            "has_reviews": len(reviews) > 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Upsert (update if exists, insert if new)
        result = await self.db.movies.update_one(
            {"tmdb_id": movie.get('id')},
            {"$set": movie_doc},
            upsert=True
        )
        
        logger.info(f"         💾 Stored in DB (Genre: {genre_name}, Rating: {movie.get('vote_average')})")
    
    async def collect_genre_data(
        self,
        genre_name: str,
        genre_id: int,
        movies_per_genre: int = 15,
        reviews_per_movie: int = 10,
        min_rating: float = 6.0
    ):
        """
        Collect movies and reviews for one genre
        
        Args:
            genre_name: Genre name
            genre_id: TMDB genre ID
            movies_per_genre: Number of movies to fetch
            reviews_per_movie: Number of reviews per movie
            min_rating: Minimum rating filter
        """
        print(f"\n{'='*70}")
        print(f"  📂 GENRE: {genre_name.upper()}")
        print(f"{'='*70}")
        
        # Fetch movies from TMDB
        movies = await self.fetch_movies_by_genre(
            genre_name=genre_name,
            genre_id=genre_id,
            movies_per_genre=movies_per_genre,
            min_rating=min_rating
        )
        
        if not movies:
            logger.warning(f"⚠️  No movies found for {genre_name}")
            return
        
        # Process each movie
        movies_with_reviews = 0
        movies_without_reviews = 0
        
        for i, movie in enumerate(movies, 1):
            movie_title = movie.get('title', 'Unknown')
            movie_rating = movie.get('vote_average', 0)
            
            print(f"\n   {i}/{len(movies)}. {movie_title} (⭐ {movie_rating})")
            
            # Collect Twitter reviews
            reviews = await self.collect_twitter_reviews(
                movie_title=movie_title,
                max_reviews=reviews_per_movie
            )
            
            # Store in database (even if no reviews)
            await self.store_movie_with_reviews(
                movie=movie,
                reviews=reviews,
                genre_name=genre_name
            )
            
            if reviews:
                movies_with_reviews += 1
            else:
                movies_without_reviews += 1
        
        print(f"\n   ✅ {genre_name} Complete:")
        print(f"      • Movies with reviews: {movies_with_reviews}")
        print(f"      • Movies without reviews: {movies_without_reviews}")
    
    async def collect_all_data(
        self,
        genres_to_collect: List[str] = None,
        movies_per_genre: int = 12,
        reviews_per_movie: int = 10,
        min_rating: float = 6.0
    ):
        """
        Main collection process for all genres
        
        Args:
            genres_to_collect: List of genre names (None = all)
            movies_per_genre: Movies per genre
            reviews_per_movie: Reviews per movie
            min_rating: Minimum TMDB rating
        """
        if genres_to_collect is None:
            genres_to_collect = list(TELUGU_GENRES.keys())
        
        total_movies = len(genres_to_collect) * movies_per_genre
        total_reviews_target = total_movies * reviews_per_movie
        
        print("\n" + "="*70)
        print("  🎬 SMART MOVIE & REVIEW COLLECTION SYSTEM")
        print("="*70)
        print(f"\n📊 Configuration:")
        print(f"   • Genres: {len(genres_to_collect)} ({', '.join(genres_to_collect[:5])}...)")
        print(f"   • Movies per genre: {movies_per_genre}")
        print(f"   • Reviews per movie: {reviews_per_movie}")
        print(f"   • Minimum rating: {min_rating}")
        print(f"   • Total movies target: {total_movies}")
        print(f"   • Total reviews target: ~{total_reviews_target}")
        print(f"\n⏱️  Estimated time: ~{(total_movies * 5 / 60):.0f} minutes")
        print(f"   (with rate limit handling)")
        
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
        
        finally:
            await self.close_db()
        
        print("\n" + "="*70)
        print("  ✅ DATA COLLECTION COMPLETE!")
        print("="*70)
        
        # Show statistics
        await self.show_statistics()
    
    async def show_statistics(self):
        """Display collection statistics"""
        await self.connect_db()
        
        try:
            total_movies = await self.db.movies.count_documents({})
            movies_with_reviews = await self.db.movies.count_documents({"has_reviews": True})
            
            print(f"\n📊 DATABASE STATISTICS:")
            print(f"   • Total movies: {total_movies}")
            print(f"   • Movies with reviews: {movies_with_reviews}")
            print(f"   • Movies without reviews: {total_movies - movies_with_reviews}")
            
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
                print(f"\n📂 By Genre:")
                for stat in genre_stats:
                    print(f"   • {stat['_id']}: {stat['count']} movies, "
                          f"⭐ {stat['avg_rating']:.1f}, "
                          f"💬 {stat['total_reviews']} reviews")
        
        finally:
            await self.close_db()


async def main():
    """Main execution"""
    
    collector = SmartMovieCollector()
    
    # Full collection: 12 genres × 12 movies = 144 movies
    # 144 movies × 10 reviews = 1440 reviews target
    await collector.collect_all_data(
        genres_to_collect=list(TELUGU_GENRES.keys()),
        movies_per_genre=12,  # 12 movies per genre = ~144 total
        reviews_per_movie=10,  # 10 Telugu reviews per movie
        min_rating=6.0  # Only movies rated 6.0 or higher
    )


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  🚀 STARTING SMART COLLECTION SYSTEM")
    print("="*70)
    print("\n⚠️  This will take time due to Twitter API rate limits")
    print("   The script will automatically handle rate limits and wait")
    print("   You can stop anytime with Ctrl+C and resume later\n")
    
    input("Press Enter to start collection...")
    
    asyncio.run(main())
