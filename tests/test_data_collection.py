"""
Test Data Collection Script
Tests the data collection pipeline with sample Telugu movies
"""
import asyncio
import logging
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.collectors import get_tmdb_collector, get_twitter_collector
from app.services import get_storage_service
from app.database import connect_to_mongo, close_mongo_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Sample Telugu movies for testing (popular and recent)
SAMPLE_MOVIES = [
    "RRR",
    "Baahubali",
    "Pushpa",
    "Ala Vaikunthapurramuloo",
    "Rangasthalam",
    "Sita Ramam",
    "Eega",
    "Arjun Reddy",
    "Uppena",
    "KGF Chapter 2"
]


async def test_single_movie_collection(movie_name: str, max_reviews: int = 10) -> Dict[str, Any]:
    """
    Test data collection for a single movie
    
    Args:
        movie_name: Name of the movie
        max_reviews: Maximum reviews to collect
        
    Returns:
        Dict with collection results
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing collection for: {movie_name}")
        logger.info(f"{'='*60}")
        
        # Initialize services
        tmdb_collector = get_tmdb_collector()
        twitter_collector = get_twitter_collector()
        storage_service = get_storage_service()
        
        # Step 1: Search movie on TMDB
        logger.info(f"[1/4] Searching TMDB for: {movie_name}")
        movie_results = await tmdb_collector.search_movie(movie_name, language="te")
        
        if not movie_results:
            logger.warning(f"❌ No movie found on TMDB: {movie_name}")
            return {
                "movie_name": movie_name,
                "success": False,
                "error": "Movie not found on TMDB"
            }
        
        logger.info(f"✓ Found {len(movie_results)} results")
        
        # Step 2: Get detailed movie info
        movie_data = movie_results[0]
        tmdb_id = movie_data.get('id')
        logger.info(f"[2/4] Getting details for TMDB ID: {tmdb_id}")
        
        detailed_movie = await tmdb_collector.get_movie_details(tmdb_id)
        
        if not detailed_movie:
            logger.error(f"❌ Failed to get movie details")
            return {
                "movie_name": movie_name,
                "success": False,
                "error": "Failed to get movie details"
            }
        
        # Step 3: Store movie
        logger.info(f"[3/4] Storing movie in database")
        parsed_movie = tmdb_collector.parse_movie_data(detailed_movie)
        movie_id = await storage_service.store_movie(parsed_movie)
        
        logger.info(f"✓ Movie stored with ID: {movie_id}")
        logger.info(f"   Title: {parsed_movie.get('title')}")
        logger.info(f"   Original: {parsed_movie.get('original_title')}")
        logger.info(f"   Release: {parsed_movie.get('release_date')}")
        logger.info(f"   Rating: {parsed_movie.get('rating')}/10")
        logger.info(f"   Genres: {', '.join(parsed_movie.get('genres', []))}")
        
        # Step 4: Collect Twitter reviews
        logger.info(f"[4/4] Collecting Twitter reviews (max: {max_reviews})")
        reviews = await twitter_collector.search_movie_reviews(
            movie_name=movie_name,
            max_results=max_reviews,
            language="te"
        )
        
        reviews_stored = 0
        if reviews:
            logger.info(f"✓ Found {len(reviews)} reviews on Twitter")
            
            # Parse and store reviews
            parsed_reviews = [
                twitter_collector.parse_tweet_data(review, str(movie_id))
                for review in reviews
            ]
            
            await storage_service.store_reviews_batch(parsed_reviews)
            reviews_stored = len(parsed_reviews)
            
            logger.info(f"✓ Stored {reviews_stored} reviews")
            
            # Show sample review
            if parsed_reviews:
                sample = parsed_reviews[0]
                logger.info(f"\n   Sample Review:")
                logger.info(f"   Text: {sample.get('text', '')[:100]}...")
                logger.info(f"   Author: @{sample.get('author_username', 'unknown')}")
                logger.info(f"   Likes: {sample.get('like_count', 0)}")
        else:
            logger.warning(f"⚠️  No reviews found on Twitter")
        
        logger.info(f"\n✅ Successfully completed collection for: {movie_name}")
        logger.info(f"   Movie ID: {movie_id}")
        logger.info(f"   Reviews: {reviews_stored}")
        
        return {
            "movie_name": movie_name,
            "success": True,
            "movie_id": str(movie_id),
            "tmdb_id": parsed_movie.get('tmdb_id'),
            "title": parsed_movie.get('title'),
            "reviews_collected": reviews_stored
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing {movie_name}: {str(e)}", exc_info=True)
        return {
            "movie_name": movie_name,
            "success": False,
            "error": str(e)
        }


async def test_batch_collection(
    movie_names: List[str],
    max_reviews_per_movie: int = 10,
    delay_between_movies: int = 3
) -> Dict[str, Any]:
    """
    Test batch collection for multiple movies
    
    Args:
        movie_names: List of movie names
        max_reviews_per_movie: Max reviews per movie
        delay_between_movies: Delay in seconds between movies
        
    Returns:
        Dict with batch results
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"BATCH COLLECTION TEST")
    logger.info(f"Movies: {len(movie_names)}")
    logger.info(f"Max reviews per movie: {max_reviews_per_movie}")
    logger.info(f"{'#'*60}\n")
    
    results = []
    successful = 0
    total_reviews = 0
    
    for i, movie_name in enumerate(movie_names, 1):
        logger.info(f"\nProcessing movie {i}/{len(movie_names)}")
        
        result = await test_single_movie_collection(movie_name, max_reviews_per_movie)
        results.append(result)
        
        if result.get('success'):
            successful += 1
            total_reviews += result.get('reviews_collected', 0)
        
        # Add delay between movies (except for the last one)
        if i < len(movie_names):
            logger.info(f"\n⏳ Waiting {delay_between_movies}s before next movie...")
            await asyncio.sleep(delay_between_movies)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH COLLECTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total movies: {len(movie_names)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(movie_names) - successful}")
    logger.info(f"Total reviews collected: {total_reviews}")
    logger.info(f"Average reviews per movie: {total_reviews / successful if successful > 0 else 0:.1f}")
    
    # Show failed movies
    failed_movies = [r for r in results if not r.get('success')]
    if failed_movies:
        logger.warning(f"\n⚠️  Failed movies:")
        for movie in failed_movies:
            logger.warning(f"   - {movie['movie_name']}: {movie.get('error', 'Unknown error')}")
    
    return {
        "total_movies": len(movie_names),
        "successful": successful,
        "failed": len(movie_names) - successful,
        "total_reviews": total_reviews,
        "results": results
    }


async def test_storage_stats():
    """Display current storage statistics"""
    try:
        storage_service = get_storage_service()
        stats = await storage_service.get_storage_stats()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STORAGE STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total movies: {stats.get('total_movies', 0)}")
        logger.info(f"Total reviews: {stats.get('total_reviews', 0)}")
        logger.info(f"Avg reviews per movie: {stats.get('avg_reviews_per_movie', 0):.2f}")
        logger.info(f"Movies with reviews: {stats.get('movies_with_reviews', 0)}")
        logger.info(f"{'='*60}\n")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting storage stats: {str(e)}")
        return {}


async def main():
    """Main test function"""
    logger.info("Starting Data Collection Test Script")
    
    # Connect to MongoDB
    logger.info("Connecting to MongoDB...")
    try:
        await connect_to_mongo()
        logger.info("✓ MongoDB connected")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.error("Cannot proceed without database connection")
        return
    
    # Show initial stats
    logger.info("\n📊 Initial Storage State:")
    await test_storage_stats()
    
    # Test single movie (detailed)
    logger.info("\n🎬 TEST 1: Single Movie Collection (Detailed)")
    await test_single_movie_collection("RRR", max_reviews=10)
    
    # Show stats after single test
    await test_storage_stats()
    
    # Test batch collection (conservative)
    logger.info("\n🎬 TEST 2: Batch Collection (Conservative)")
    sample_movies = SAMPLE_MOVIES[:3]  # First 3 movies (conservative for free tier)
    await test_batch_collection(
        movie_names=sample_movies,
        max_reviews_per_movie=10,  # Twitter API minimum
        delay_between_movies=5     # 5 seconds between movies (safe for rate limits)
    )
    
    # Show final stats
    logger.info("\n📊 Final Storage State:")
    await test_storage_stats()
    
    logger.info("\n✅ Data Collection Test Completed!")
    logger.info("\nNote: If Twitter API rate limits are reached, the script will")
    logger.info("automatically wait. Be patient with the free tier limits.")
    
    # Close database connection
    logger.info("\nClosing MongoDB connection...")
    await close_mongo_connection()
    logger.info("✓ MongoDB connection closed")


if __name__ == "__main__":
    asyncio.run(main())
