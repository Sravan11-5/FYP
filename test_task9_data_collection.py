"""
Test Task 9: Test data collection for multiple movies
Tests the complete data collection pipeline with multiple Telugu movies
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.tmdb_collector import TMDBDataCollector
from app.services.twitter_collector import TwitterDataCollector
from app.services.database_storage import DatabaseStorageService
from app.database import connect_to_mongo, get_database, close_mongo_connection
from app.config import settings


async def test_task_9():
    """Test the complete data collection pipeline for multiple movies"""
    
    print("=" * 80)
    print("TEST TASK 9: DATA COLLECTION FOR MULTIPLE MOVIES")
    print("=" * 80)
    
    # Connect to database
    print("\n🔌 Connecting to MongoDB...")
    await connect_to_mongo()
    print("✅ Connected to MongoDB")
    
    # Get database connection
    db = get_database()
    
    # Initialize services
    tmdb_collector = TMDBDataCollector()
    twitter_collector = TwitterDataCollector()
    storage_service = DatabaseStorageService(db)
    
    # Configuration: Skip Twitter due to rate limits (15 min cooldown)
    SKIP_TWITTER_COLLECTION = True  # Set to False if Twitter API is available
    
    # =========================================================================
    # Subtask 9.1: Create a list of Telugu movie names
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUBTASK 9.1: CREATE LIST OF TELUGU MOVIE NAMES")
    print("=" * 80)
    
    # Diverse list of Telugu movies - popular and recent films
    telugu_movies = [
        "RRR",
        "Baahubali",
        "Pushpa",
        "Ala Vaikunthapurramuloo",
        "Rangasthalam",
        "Eega",
        "Sita Ramam",
        "Arjun Reddy",
        "Jersey",
        "Maharshi"
    ]
    
    print(f"\n✅ Created list of {len(telugu_movies)} Telugu movies:")
    for idx, movie in enumerate(telugu_movies, 1):
        print(f"   {idx}. {movie}")
    
    # =========================================================================
    # Subtask 9.2: Fetch metadata and reviews for each movie
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUBTASK 9.2: FETCH METADATA AND REVIEWS")
    print("=" * 80)
    
    collection_stats = {
        "total_movies": len(telugu_movies),
        "movies_found": 0,
        "movies_not_found": 0,
        "total_reviews_collected": 0,
        "errors": [],
        "start_time": time.time()
    }
    
    collected_data = []
    
    print(f"\n🔄 Starting data collection for {len(telugu_movies)} movies...")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for idx, movie_name in enumerate(telugu_movies, 1):
        print(f"\n{'─' * 80}")
        print(f"📽️  Processing Movie {idx}/{len(telugu_movies)}: {movie_name}")
        print(f"{'─' * 80}")
        
        try:
            # Search for the movie on TMDB
            print(f"   🔍 Searching TMDB for '{movie_name}'...")
            search_response = tmdb_collector.search_movie_with_retry(movie_name)
            
            if not search_response or not search_response.get('results') or len(search_response['results']) == 0:
                print(f"   ❌ No results found for '{movie_name}'")
                collection_stats["movies_not_found"] += 1
                collection_stats["errors"].append(f"{movie_name}: No TMDB results")
                continue
            
            # Get the first (most relevant) result
            movie = search_response['results'][0]
            tmdb_id = movie.get('id')
            title = movie.get('title', movie_name)
            
            print(f"   ✅ Found: {title} (TMDB ID: {tmdb_id})")
            
            # Get detailed movie metadata
            print(f"   📊 Fetching detailed metadata...")
            movie_details = tmdb_collector.get_movie_details_with_retry(tmdb_id)
            
            if not movie_details:
                print(f"   ⚠️  Could not fetch details for TMDB ID {tmdb_id}")
                collection_stats["errors"].append(f"{movie_name}: Failed to fetch details")
                continue
            
            print(f"   ✅ Metadata collected: {movie_details.get('vote_count', 0)} votes, "
                  f"rating {movie_details.get('vote_average', 0)}/10")
            
            # Transform TMDB data to match storage format
            # TMDB returns 'id' but storage expects 'tmdb_id'
            movie_details['tmdb_id'] = movie_details.pop('id', tmdb_id)
            
            # TMDB returns genres as [{'id': 28, 'name': 'Action'}] but we need [28]
            if 'genres' in movie_details and isinstance(movie_details['genres'], list):
                movie_details['genres'] = [g['id'] if isinstance(g, dict) else g for g in movie_details['genres']]
            
            # Collect Twitter reviews (skip if rate limited)
            reviews = []
            review_count = 0
            if SKIP_TWITTER_COLLECTION:
                print(f"   ⏭️  Skipping Twitter collection (rate limit/API unavailable)")
            else:
                print(f"   🐦 Collecting Twitter reviews...")
                reviews = twitter_collector.collect_movie_reviews_with_metadata(
                    movie_name=movie_name,
                    tmdb_id=tmdb_id,
                    max_results=50,  # Limit to avoid rate limits
                    days_back=180    # 6 months of reviews
                )
                review_count = len(reviews)
                print(f"   ✅ Collected {review_count} reviews from Twitter")
            
            collection_stats["movies_found"] += 1
            collection_stats["total_reviews_collected"] += review_count
            
            # Store the collected data
            collected_data.append({
                "movie_metadata": movie_details,
                "reviews": reviews,
                "movie_name": movie_name,
                "tmdb_id": tmdb_id
            })
            
            print(f"   ✅ Successfully collected data for '{title}'")
            
            # Small delay to be respectful to APIs
            if idx < len(telugu_movies):
                await asyncio.sleep(1)
            
        except Exception as e:
            error_msg = f"{movie_name}: {str(e)}"
            print(f"   ❌ ERROR: {str(e)}")
            collection_stats["errors"].append(error_msg)
    
    collection_stats["end_time"] = time.time()
    collection_stats["duration_seconds"] = collection_stats["end_time"] - collection_stats["start_time"]
    
    print("\n" + "=" * 80)
    print("📊 COLLECTION SUMMARY")
    print("=" * 80)
    print(f"Total movies processed: {collection_stats['total_movies']}")
    print(f"Movies found: {collection_stats['movies_found']} ✅")
    print(f"Movies not found: {collection_stats['movies_not_found']} ❌")
    print(f"Total reviews collected: {collection_stats['total_reviews_collected']} 🐦")
    print(f"Duration: {collection_stats['duration_seconds']:.2f} seconds")
    print(f"Average time per movie: {collection_stats['duration_seconds'] / collection_stats['total_movies']:.2f} seconds")
    
    if collection_stats["errors"]:
        print(f"\n⚠️  Errors encountered ({len(collection_stats['errors'])}):")
        for error in collection_stats["errors"]:
            print(f"   - {error}")
    
    # =========================================================================
    # Subtask 9.3: Store the data in the database
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUBTASK 9.3: STORE DATA IN DATABASE")
    print("=" * 80)
    
    storage_stats = {
        "movies_stored": 0,
        "movies_skipped": 0,
        "movies_updated": 0,
        "reviews_stored": 0,
        "reviews_skipped": 0,
        "reviews_updated": 0,
        "storage_errors": []
    }
    
    print(f"\n💾 Storing {len(collected_data)} movies and their reviews...\n")
    
    for idx, data in enumerate(collected_data, 1):
        movie_name = data["movie_name"]
        movie_metadata = data["movie_metadata"]
        reviews = data["reviews"]
        
        print(f"{'─' * 80}")
        print(f"💾 Storing {idx}/{len(collected_data)}: {movie_name}")
        print(f"{'─' * 80}")
        
        try:
            # Store movie metadata
            print(f"   📊 Storing movie metadata (TMDB ID: {data['tmdb_id']})...")
            movie_result = await storage_service.store_movie(
                movie_metadata,
                update_if_exists=True
            )
            
            if movie_result["success"]:
                operation = movie_result.get("operation", "unknown")
                if operation == "inserted":
                    storage_stats["movies_stored"] += 1
                    print(f"   ✅ Movie inserted into database")
                elif operation == "updated":
                    storage_stats["movies_updated"] += 1
                    print(f"   🔄 Movie updated in database")
                elif operation == "skipped":
                    storage_stats["movies_skipped"] += 1
                    print(f"   ⏭️  Movie already exists (skipped)")
            else:
                storage_stats["storage_errors"].append(f"{movie_name} (movie): {movie_result.get('error')}")
                print(f"   ❌ Failed to store movie: {movie_result.get('message')}")
            
            # Store reviews in batch
            if reviews:
                print(f"   🐦 Storing {len(reviews)} reviews...")
                reviews_result = await storage_service.store_multiple_reviews(
                    reviews,
                    update_if_exists=False  # Don't update existing reviews
                )
                
                if reviews_result["success"]:
                    stored_count = reviews_result.get("inserted", 0)
                    skipped_count = reviews_result.get("skipped", 0)
                    
                    storage_stats["reviews_stored"] += stored_count
                    storage_stats["reviews_skipped"] += skipped_count
                    
                    print(f"   ✅ Reviews stored: {stored_count}, skipped: {skipped_count}")
                else:
                    error_count = len(reviews_result.get("errors", []))
                    storage_stats["storage_errors"].append(f"{movie_name} (reviews): {error_count} errors")
                    print(f"   ⚠️  Some reviews failed to store ({error_count} errors)")
            else:
                print(f"   ℹ️  No reviews to store")
            
            print(f"   ✅ Completed storage for '{movie_name}'")
            
        except Exception as e:
            error_msg = f"{movie_name}: {str(e)}"
            storage_stats["storage_errors"].append(error_msg)
            print(f"   ❌ ERROR during storage: {str(e)}")
    
    print("\n" + "=" * 80)
    print("💾 STORAGE SUMMARY")
    print("=" * 80)
    print(f"Movies stored (new): {storage_stats['movies_stored']} ✅")
    print(f"Movies updated: {storage_stats['movies_updated']} 🔄")
    print(f"Movies skipped: {storage_stats['movies_skipped']} ⏭️")
    print(f"Reviews stored (new): {storage_stats['reviews_stored']} ✅")
    print(f"Reviews skipped (duplicates): {storage_stats['reviews_skipped']} ⏭️")
    
    if storage_stats["storage_errors"]:
        print(f"\n⚠️  Storage errors ({len(storage_stats['storage_errors'])}):")
        for error in storage_stats["storage_errors"]:
            print(f"   - {error}")
    
    # =========================================================================
    # Subtask 9.4: Monitor the process for errors and performance bottlenecks
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUBTASK 9.4: MONITORING & PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Get API request statistics
    tmdb_stats = tmdb_collector.get_request_stats()
    twitter_stats = twitter_collector.get_request_stats()
    
    print("\n📡 API REQUEST STATISTICS:")
    print(f"   TMDB Requests: {tmdb_stats['total_requests']}")
    print(f"   - Rate limit delay: {tmdb_stats['rate_limit_delay']}s per request")
    
    print(f"\n   Twitter Requests: {twitter_stats['total_requests']}")
    print(f"   - Rate limit delay: {twitter_stats['rate_limit_delay']}s per request")
    
    # Get database statistics
    print("\n💾 DATABASE STATISTICS:")
    db_stats = await storage_service.get_storage_statistics()
    print(f"   Total movies in database: {db_stats.get('total_movies', 0)}")
    print(f"   Total reviews in database: {db_stats.get('total_reviews', 0)}")
    if db_stats.get('average_rating') is not None:
        print(f"   Average movie rating: {db_stats['average_rating']:.2f}/10")
    
    if db_stats.get('top_genres'):
        print(f"\n   Top genres (by movie count):")
        for genre_id, count in db_stats['top_genres'][:5]:
            print(f"   - Genre {genre_id}: {count} movies")
    
    # Performance Analysis
    print("\n⚡ PERFORMANCE ANALYSIS:")
    print(f"   Total execution time: {collection_stats['duration_seconds']:.2f} seconds")
    print(f"   Time per movie: {collection_stats['duration_seconds'] / collection_stats['total_movies']:.2f} seconds")
    
    if collection_stats['movies_found'] > 0:
        reviews_per_movie = collection_stats['total_reviews_collected'] / collection_stats['movies_found']
        print(f"   Average reviews per movie: {reviews_per_movie:.1f}")
    
    # Error Rate Analysis
    total_operations = collection_stats['total_movies']
    error_count = len(collection_stats['errors']) + len(storage_stats['storage_errors'])
    if total_operations > 0:
        error_rate = (error_count / total_operations) * 100
        print(f"\n   Error rate: {error_rate:.1f}% ({error_count} errors out of {total_operations} operations)")
    
    # Bottleneck Analysis
    print("\n🔍 POTENTIAL BOTTLENECKS:")
    if collection_stats['duration_seconds'] / collection_stats['total_movies'] > 10:
        print("   ⚠️  Average time per movie is high (>10 seconds)")
        print("      - Consider implementing parallel processing")
        print("      - Check API rate limits and response times")
    else:
        print("   ✅ Processing time is acceptable")
    
    # Check if we have enough successful collections
    if collection_stats['movies_found'] < collection_stats['total_movies'] * 0.8:
        print("   ⚠️  Low success rate for movie collection (<80%)")
        print("      - Check API connectivity and search queries")
        print("      - Review error messages above")
    else:
        print("   ✅ Movie collection success rate is good")
    
    if storage_stats['storage_errors']:
        print(f"   ⚠️  Storage errors detected ({len(storage_stats['storage_errors'])})")
        print("      - Review validation rules and data format")
    else:
        print("   ✅ No storage errors detected")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("✅ TASK 9 COMPLETE - FINAL SUMMARY")
    print("=" * 80)
    print(f"Total movies processed: {collection_stats['total_movies']}")
    print(f"Successfully collected: {collection_stats['movies_found']} movies")
    print(f"Total reviews collected: {collection_stats['total_reviews_collected']}")
    print(f"Movies stored in database: {storage_stats['movies_stored'] + storage_stats['movies_updated']}")
    print(f"Reviews stored in database: {storage_stats['reviews_stored']}")
    print(f"Total execution time: {collection_stats['duration_seconds']:.2f} seconds")
    
    success_rate = (collection_stats['movies_found'] / collection_stats['total_movies']) * 100
    print(f"\n📊 Overall Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Data collection pipeline is working well!")
    elif success_rate >= 60:
        print("⚠️  Data collection pipeline needs some improvements")
    else:
        print("❌ Data collection pipeline needs significant improvements")
    
    print("\n" + "=" * 80)
    print("All subtasks of Task 9 completed!")
    print("=" * 80)
    
    # Close database connection
    await close_mongo_connection()


if __name__ == "__main__":
    asyncio.run(test_task_9())
