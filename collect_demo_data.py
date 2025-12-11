"""
FINAL DEMO COLLECTION - 9 Movies × 10 Reviews
This is our last 90 tweets - every single one counts!
Ultra-careful implementation with progress saving.
"""
import asyncio
import logging
import time
import json
from typing import List, Dict
from datetime import datetime
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from app.collectors.tmdb_collector import TMDBDataCollector
from app.collectors.twitter_collector import TwitterDataCollector
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Target configuration - EXPANDED FOR BETTER RECOMMENDATIONS
TARGET_GENRES = {
    "Action": 28,
    "Drama": 18,
    "Thriller": 53,
    "Comedy": 35  # NEW: Adding Comedy genre
}
MOVIES_PER_GENRE = 5  # Increased from 3 to 5 per genre
REVIEWS_PER_MOVIE = 10
TOTAL_TWEETS_NEEDED = 60  # Remaining quota from API (100 - 40 already used)
SKIP_MOVIES = [579974, 256040, 350312]  # RRR, Baahubali 1, Baahubali 2 (already attempted)
FAILED_MOVIES = []  # Movies with 0 reviews - will be populated from progress file


class DemoCollector:
    """Ultra-careful collector for demo data"""
    
    def __init__(self):
        self.tmdb = TMDBDataCollector()
        self.twitter = TwitterDataCollector()
        self.db_client = None
        self.db = None
        
        # Progress tracking
        self.progress_file = Path("collection_progress.json")
        self.movies_collected = []
        self.tweets_used = 0
        self.failed_movies = []  # Track movies with 0 reviews
        self.load_progress()
        
    def load_progress(self):
        """Load previous progress if exists"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                self.movies_collected = data.get('movies_collected', [])
                self.tweets_used = data.get('tweets_used', 0)
                self.failed_movies = data.get('failed_movies', [])
                logger.info(f"📂 Loaded progress: {len(self.movies_collected)} movies, {self.tweets_used} tweets used, {len(self.failed_movies)} failed movies skipped")
    
    def save_progress(self):
        """Save progress after each movie"""
        with open(self.progress_file, 'w') as f:
            json.dump({
                'movies_collected': self.movies_collected,
                'tweets_used': self.tweets_used,
                'failed_movies': self.failed_movies,
                'last_update': datetime.utcnow().isoformat()
            }, f, indent=2)
        logger.info(f"💾 Progress saved: {len(self.movies_collected)} movies, {self.tweets_used} tweets")
    
    async def connect_db(self):
        """Connect to MongoDB"""
        self.db_client = AsyncIOMotorClient(settings.MONGODB_URL)
        self.db = self.db_client[settings.MONGODB_DB_NAME]
        logger.info("✅ Connected to MongoDB")
        
    async def close_db(self):
        """Close MongoDB connection"""
        if self.db_client:
            self.db_client.close()
    
    async def fetch_top_movies(
        self,
        genre_name: str,
        genre_id: int,
        count: int = 3
    ) -> List[Dict]:
        """Fetch top-rated, popular movies for genre (no duplicates)"""
        logger.info(f"\n📂 Fetching {count} best {genre_name} movies...")
        
        # Get MORE movies to ensure variety and quality
        movies = await self.tmdb.discover_movies_by_genre(
            genre_ids=[genre_id],
            min_vote_average=7.5,  # Higher quality threshold
            max_results=count * 5  # Get 5x more for better selection
        )
        
        # Filter out already collected, failed (0 reviews), AND previously attempted movies
        tmdb_ids_collected = [m['tmdb_id'] for m in self.movies_collected]
        failed_ids = [m['tmdb_id'] for m in self.failed_movies]
        movies = [m for m in movies 
                 if m.get('id') not in tmdb_ids_collected 
                 and m.get('id') not in failed_ids
                 and m.get('id') not in SKIP_MOVIES]
        
        # Sort by popularity and rating for best movies
        movies = sorted(movies, key=lambda x: (x.get('popularity', 0) + x.get('vote_average', 0) * 10), reverse=True)
        
        logger.info(f"   ✅ Found {len(movies[:count])} top {genre_name} movies (skipping already attempted)")
        return movies[:count]
    
    async def collect_reviews_for_movie(
        self,
        movie_title: str,
        max_reviews: int = 10
    ) -> List[Dict]:
        """Collect Telugu reviews with extreme care"""
        try:
            logger.info(f"      🐦 Fetching {max_reviews} Telugu reviews...")
            logger.info(f"      ⏳ Waiting 15 minutes for rate limit...")
            
            # CRITICAL: Wait full 15 minutes + 5 second buffer
            await asyncio.sleep(905)
            
            # Make the API call
            tweets = await self.twitter.search_movie_reviews(
                movie_name=movie_title,
                max_results=max_reviews,
                language="te"
            )
            
            if not tweets:
                logger.warning(f"         ⚠️  No reviews found for '{movie_title}'")
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
            
            logger.info(f"         ✅ Collected {len(reviews)} reviews!")
            return reviews
            
        except Exception as e:
            logger.error(f"         ❌ Error: {e}")
            return []
    
    async def store_movie(
        self,
        movie: Dict,
        reviews: List[Dict],
        genre_name: str
    ):
        """Store movie and reviews in database"""
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
        
        await self.db.movies.update_one(
            {"tmdb_id": movie.get('id')},
            {"$set": movie_doc},
            upsert=True
        )
        
        logger.info(f"         💾 Stored in database")
    
    async def collect_all(self):
        """Main collection process"""
        print("\n" + "="*70)
        print("  🎯 EXPANDED COLLECTION - MORE VARIETY FOR BETTER RECOMMENDATIONS")
        print("="*70)
        
        target_movies = MOVIES_PER_GENRE * len(TARGET_GENRES)  # 5 × 4 = 20 movies
        print(f"\n📊 Mission:")
        print(f"   • Target: {target_movies} movies (5 per genre) with reviews")
        print(f"   • Genres: Action (5), Drama (5), Thriller (5), Comedy (5)")
        print(f"   • Already collected: {len(self.movies_collected)} movies, {self.tweets_used} tweets")
        print(f"   • Remaining quota: {TOTAL_TWEETS_NEEDED} tweets")
        print(f"   • Estimated time: Variable (depends on review availability)")
        
        await self.connect_db()
        
        try:
            movie_counter = len(self.movies_collected)
            
            for genre_name, genre_id in TARGET_GENRES.items():
                print(f"\n{'='*70}")
                print(f"  📂 GENRE: {genre_name.upper()}")
                print(f"{'='*70}")
                
                # Get movies for this genre
                movies = await self.fetch_top_movies(
                    genre_name=genre_name,
                    genre_id=genre_id,
                    count=MOVIES_PER_GENRE
                )
                
                for i, movie in enumerate(movies, 1):
                    movie_counter += 1
                    movie_title = movie.get('title', 'Unknown')
                    movie_rating = movie.get('vote_average', 0)
                    
                    print(f"\n   Movie {movie_counter}/20: {movie_title} (⭐ {movie_rating})")
                    
                    # Check if already collected
                    if movie.get('id') in [m['tmdb_id'] for m in self.movies_collected]:
                        logger.info(f"      ⏭️  Already collected, skipping...")
                        continue
                    
                    # Use ENGLISH title for Twitter search (Telugu script doesn't work!)
                    search_title = movie.get('original_title', movie_title)
                    # If still Telugu script, try to use a common English name
                    if any(ord(c) > 127 for c in search_title):  # Has non-ASCII (Telugu)
                        # Try alternative title or original_title
                        search_title = movie.get('title', search_title)
                    
                    logger.info(f"      🔍 Searching Twitter with: '{search_title}'")
                    
                    # Collect reviews
                    reviews = await self.collect_reviews_for_movie(
                        movie_title=search_title,
                        max_reviews=REVIEWS_PER_MOVIE
                    )
                    
                    if reviews:
                        # Store in database
                        await self.store_movie(
                            movie=movie,
                            reviews=reviews,
                            genre_name=genre_name
                        )
                        
                        # Update progress
                        self.movies_collected.append({
                            'tmdb_id': movie.get('id'),
                            'title': movie_title,
                            'genre': genre_name,
                            'review_count': len(reviews)
                        })
                        self.tweets_used += len(reviews)
                        self.save_progress()
                        
                        print(f"      ✅ SUCCESS! Total progress: {len(self.movies_collected)}/20 movies, {self.tweets_used}/{TOTAL_TWEETS_NEEDED} tweets")
                    else:
                        # Add to failed movies list to skip in future
                        self.failed_movies.append({
                            'tmdb_id': movie.get('id'),
                            'title': movie_title,
                            'genre': genre_name
                        })
                        self.tweets_used += 1  # Still used 1 API call
                        self.save_progress()
                        logger.warning(f"      ⚠️  No reviews found - added to skip list. Trying next movie...")
                        movie_counter -= 1  # Don't count failed movies
            
            print(f"\n{'='*70}")
            print(f"  🎉 COLLECTION COMPLETE!")
            print(f"{'='*70}")
            print(f"\n📊 Final Stats:")
            print(f"   • Movies collected: {len(self.movies_collected)}")
            print(f"   • Total tweets used: {self.tweets_used}")
            print(f"   • Remaining quota: {100 - 10 - self.tweets_used}")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Collection paused by user")
            print(f"   Progress saved: {len(self.movies_collected)} movies, {self.tweets_used} tweets")
            print(f"   Run again to resume from where you left off!")
            
        finally:
            await self.close_db()
    
    async def show_stats(self):
        """Show database statistics"""
        await self.connect_db()
        
        try:
            total = await self.db.movies.count_documents({})
            with_reviews = await self.db.movies.count_documents({"has_reviews": True})
            
            print(f"\n{'='*70}")
            print(f"  📊 DATABASE STATISTICS")
            print(f"{'='*70}")
            print(f"\n   • Total movies: {total}")
            print(f"   • Movies with reviews: {with_reviews}")
            
            # By genre
            pipeline = [
                {"$group": {
                    "_id": "$genre",
                    "count": {"$sum": 1},
                    "avg_rating": {"$avg": "$rating"},
                    "total_reviews": {"$sum": "$review_count"}
                }},
                {"$sort": {"_id": 1}}
            ]
            
            stats = await self.db.movies.aggregate(pipeline).to_list(None)
            
            if stats:
                print(f"\n   By Genre:")
                for s in stats:
                    print(f"      • {s['_id']}: {s['count']} movies, "
                          f"⭐ {s['avg_rating']:.1f}, "
                          f"💬 {s['total_reviews']} reviews")
        
        finally:
            await self.close_db()


async def main():
    collector = DemoCollector()
    
    print("\n" + "="*70)
    print("  🚀 EXPANDED DATA COLLECTION - MORE VARIETY!")
    print("="*70)
    print("\n⚠️  IMPORTANT:")
    print("   • Target: 20 movies (5 per genre: Action, Drama, Thriller, Comedy)")
    print("   • Will use ~60 tweets (you have 60 remaining)")
    print("   • Takes ~3-5 hours (15 min wait per movie)")
    print("   • Progress is saved - can pause/resume anytime")
    print("   • Press Ctrl+C to pause safely")
    
    proceed = input("\n👉 Ready to start? (yes/no): ").strip().lower()
    
    if proceed != 'yes':
        print("\n❌ Collection cancelled")
        return
    
    # Collect data
    await collector.collect_all()
    
    # Show final stats
    await collector.show_stats()


if __name__ == "__main__":
    asyncio.run(main())
