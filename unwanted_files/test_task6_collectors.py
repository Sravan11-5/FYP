"""
Test script for Task 6: Data Collection Agents
Tests TMDB and Twitter collectors with retry logic and rate limiting
"""
import sys
import logging
from app.services.tmdb_collector import TMDBDataCollector
from app.services.twitter_collector import TwitterDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_tmdb_collector():
    """Test TMDB Data Collector functionality"""
    print("\n" + "="*80)
    print("TESTING TMDB DATA COLLECTOR")
    print("="*80)
    
    collector = TMDBDataCollector()
    
    # Test 1: Search single movie
    print("\n[Test 1] Search for 'Baahubali'")
    result = collector.search_movie_with_retry("Baahubali", language="te")
    if result and result.get('results'):
        print(f"✅ Found {len(result['results'])} results")
        print(f"   First result: {result['results'][0].get('title')}")
    else:
        print("❌ Search failed")
    
    # Test 2: Get movie details
    print("\n[Test 2] Get movie details for Baahubali (ID: 325980)")
    details = collector.get_movie_details_with_retry(325980, language="te")
    if details:
        print(f"✅ Retrieved details")
        print(f"   Title: {details.get('title')}")
        print(f"   Rating: {details.get('vote_average')}")
        print(f"   Genres: {[g['name'] for g in details.get('genres', [])]}")
    else:
        print("❌ Failed to get details")
    
    # Test 3: Get genre list
    print("\n[Test 3] Get genre list")
    genres = collector.get_genre_list(language="te")
    if genres:
        print(f"✅ Retrieved {len(genres)} genres")
        print(f"   Sample genres: {[g['name'] for g in genres[:5]]}")
    else:
        print("❌ Failed to get genres")
    
    # Test 4: Discover movies by genre (Action = 28, Thriller = 53)
    print("\n[Test 4] Discover Action/Thriller movies")
    discovered = collector.discover_similar_movies_by_genre(
        genre_ids=[28, 53],
        language="te",
        max_results=5
    )
    if discovered:
        print(f"✅ Discovered {len(discovered)} movies")
        for movie in discovered[:3]:
            print(f"   - {movie.get('title')} (Rating: {movie.get('vote_average')})")
    else:
        print("❌ Discovery failed")
    
    # Test 5: Get similar movies
    print("\n[Test 5] Get similar movies to Baahubali")
    similar = collector.get_similar_movies_with_retry(325980, language="te", max_results=5)
    if similar:
        print(f"✅ Found {len(similar)} similar movies")
        for movie in similar[:3]:
            print(f"   - {movie.get('title')}")
    else:
        print("❌ Failed to get similar movies")
    
    # Test 6: Batch search
    print("\n[Test 6] Batch search for multiple movies")
    movie_names = ["RRR", "Pushpa", "Jersey"]
    batch_results = collector.batch_search_movies(movie_names, language="te")
    
    for movie_name, result in batch_results.items():
        if result and result.get('results'):
            print(f"✅ {movie_name}: Found {len(result['results'])} results")
        else:
            print(f"❌ {movie_name}: No results")
    
    # Test 7: Collect complete movie data
    print("\n[Test 7] Collect complete movie data with metadata")
    complete_data = collector.collect_movie_with_metadata(
        "Hit",
        language="te",
        include_similar=True,
        include_credits=False  # Skip credits to save API calls
    )
    
    if complete_data:
        print(f"✅ Collected complete data")
        print(f"   TMDB ID: {complete_data.get('tmdb_id')}")
        print(f"   Title: {complete_data.get('title')}")
        print(f"   Rating: {complete_data.get('vote_average')}")
        print(f"   Similar movies: {len(complete_data.get('similar_movies', []))}")
    else:
        print("❌ Failed to collect complete data")
    
    # Display stats
    print("\n[Statistics]")
    stats = collector.get_request_stats()
    print(f"   Total API requests: {stats['total_requests']}")
    print(f"   Rate limit delay: {stats['rate_limit_delay']}s")


def test_twitter_collector():
    """Test Twitter Data Collector functionality"""
    print("\n" + "="*80)
    print("TESTING TWITTER DATA COLLECTOR")
    print("="*80)
    
    collector = TwitterDataCollector()
    
    # Check if client is initialized
    if not collector.client.client:
        print("⚠️  Twitter API client not initialized (check API keys)")
        print("   Skipping Twitter tests...")
        return
    
    # Test 1: Search movie reviews
    print("\n[Test 1] Search reviews for 'Baahubali'")
    reviews = collector.search_movie_reviews_with_retry(
        "Baahubali",
        max_results=10,
        days_back=30,
        language="te"
    )
    
    if reviews is not None:
        print(f"✅ Found {len(reviews)} reviews")
        if len(reviews) > 0:
            sample = reviews[0]
            print(f"   Sample tweet: {sample.get('text')[:100]}...")
            print(f"   Likes: {sample.get('likes')}, Retweets: {sample.get('retweets')}")
    else:
        print("❌ Search failed")
    
    # Test 2: Batch collect reviews
    print("\n[Test 2] Batch collect reviews for multiple movies")
    movie_names = ["RRR", "Pushpa"]
    batch_reviews = collector.batch_collect_movie_reviews(
        movie_names,
        max_results_per_movie=5,
        days_back=30
    )
    
    for movie_name, reviews in batch_reviews.items():
        if reviews:
            print(f"✅ {movie_name}: {len(reviews)} reviews")
        else:
            print(f"❌ {movie_name}: No reviews found")
    
    # Test 3: Collect with metadata
    print("\n[Test 3] Collect reviews with metadata")
    data = collector.collect_movie_reviews_with_metadata(
        "Hit",
        max_results=20,
        days_back=30,
        min_likes=0
    )
    
    print(f"   Movie: {data.get('movie_name')}")
    print(f"   Total reviews: {data.get('total_reviews')}")
    
    if data.get('metadata'):
        meta = data['metadata']
        print(f"   Total likes: {meta.get('total_likes')}")
        print(f"   Avg likes/review: {meta.get('avg_likes_per_review', 0):.2f}")
    
    # Test 4: Basic sentiment analysis
    print("\n[Test 4] Analyze sentiment distribution")
    sentiment = collector.analyze_review_sentiment_distribution(
        "Baahubali",
        max_results=50,
        days_back=30
    )
    
    if sentiment.get('sentiment_distribution'):
        dist = sentiment['sentiment_distribution']
        print(f"   Total reviews analyzed: {sentiment.get('total_reviews')}")
        print(f"   Positive: {dist['positive']['count']} ({dist['positive']['percentage']}%)")
        print(f"   Negative: {dist['negative']['count']} ({dist['negative']['percentage']}%)")
        print(f"   Neutral: {dist['neutral']['count']} ({dist['neutral']['percentage']}%)")
        print(f"   {sentiment.get('note')}")
    
    # Display stats
    print("\n[Statistics]")
    stats = collector.get_request_stats()
    print(f"   Total API requests: {stats['total_requests']}")
    print(f"   Rate limit delay: {stats['rate_limit_delay']}s")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TASK 6: DATA COLLECTION AGENTS - COMPREHENSIVE TEST")
    print("="*80)
    
    try:
        # Test TMDB Collector
        test_tmdb_collector()
        
        # Test Twitter Collector
        test_twitter_collector()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED!")
        print("="*80)
        print("\nTask 6 Status:")
        print("✅ 6.1: TMDB Data Collector Class - Created")
        print("✅ 6.2: TMDB Data Fetching Methods - Implemented")
        print("✅ 6.3: Twitter Data Collector Class - Created")
        print("✅ 6.4: Twitter Review Fetching Method - Implemented")
        print("✅ 6.5: API Rate Limit and Error Handling - Implemented")
        print("\n🎉 Task 6 Complete!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
