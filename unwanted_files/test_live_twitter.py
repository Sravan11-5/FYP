"""
Test Live Twitter Data Collection
Verifies Twitter API connection and fetches real Telugu movie reviews
"""
import asyncio
from app.collectors.twitter_collector import TwitterDataCollector
from app.config import settings


async def test_live_twitter_data():
    """Test fetching live Twitter data"""
    
    print("🐦 Testing Live Twitter Data Collection")
    print("=" * 60)
    
    # Check API key configuration
    if not settings.TWITTER_BEARER_TOKEN:
        print("❌ ERROR: Twitter Bearer Token not configured!")
        print("\nPlease add to your .env file:")
        print("TWITTER_BEARER_TOKEN='your_bearer_token_here'")
        return
    
    print(f"✅ Bearer Token configured: {settings.TWITTER_BEARER_TOKEN[:20]}...")
    print()
    
    # Initialize collector
    collector = TwitterDataCollector()
    
    # Test with popular Telugu movies
    test_movies = [
        "RRR",
        "Pushpa",
        "Salaar"
    ]
    
    for i, movie in enumerate(test_movies):
        print(f"\n📽️  Searching for: {movie}")
        print("-" * 60)
        
        # Add delay between requests to avoid rate limit
        if i > 0:
            print("⏳ Waiting 5 seconds (rate limit)...")
            await asyncio.sleep(5)
        
        try:
            # Fetch LIVE tweets (max 10 for free tier)
            tweets = await collector.search_movie_reviews(
                movie_name=movie,
                max_results=10,
                language="te"
            )
            
            if tweets:
                print(f"✅ Found {len(tweets)} LIVE tweets!")
                print("\nSample tweets:")
                
                for i, tweet in enumerate(tweets[:3], 1):
                    text = tweet.get('text', 'N/A')
                    created_at = tweet.get('created_at', 'N/A')
                    metrics = tweet.get('public_metrics', {})
                    likes = metrics.get('like_count', 0)
                    retweets = metrics.get('retweet_count', 0)
                    
                    print(f"\n  {i}. Tweet ID: {tweet.get('id')}")
                    print(f"     Created: {created_at}")
                    print(f"     Text: {text[:100]}...")
                    print(f"     Likes: {likes}, Retweets: {retweets}")
                
                print(f"\n  ... and {len(tweets) - 3} more tweets")
            else:
                print(f"⚠️  No tweets found for '{movie}'")
                print("   This could mean:")
                print("   1. No recent Telugu tweets about this movie")
                print("   2. API rate limit reached")
                print("   3. Invalid search query")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Live Twitter data test complete!")
    print("\nNote: Twitter Free Tier limits:")
    print("- 15 requests per 15-minute window")
    print("- 10-100 tweets per request")
    print("- Recent tweets only (last 7 days)")


async def test_specific_search():
    """Test with a specific custom search"""
    
    print("\n\n🔍 Custom Search Test")
    print("=" * 60)
    
    collector = TwitterDataCollector()
    
    # Custom search query
    movie = input("\nEnter a Telugu movie name to search: ").strip()
    
    if not movie:
        movie = "RRR"  # Default
    
    print(f"\n📡 Fetching LIVE tweets about '{movie}'...")
    
    tweets = await collector.search_movie_reviews(
        movie_name=movie,
        max_results=20,  # Try getting more
        language="te"
    )
    
    if tweets:
        print(f"\n✅ SUCCESS! Found {len(tweets)} live tweets!")
        
        # Show detailed info
        print("\n" + "=" * 60)
        print("LIVE TWEET DETAILS:")
        print("=" * 60)
        
        for i, tweet in enumerate(tweets, 1):
            print(f"\n{i}. {'-' * 50}")
            print(f"ID: {tweet.get('id')}")
            print(f"Author: {tweet.get('author_id')}")
            print(f"Created: {tweet.get('created_at')}")
            print(f"Language: {tweet.get('lang')}")
            
            text = tweet.get('text', '')
            print(f"Text: {text}")
            
            metrics = tweet.get('public_metrics', {})
            print(f"Engagement: {metrics.get('like_count', 0)} likes, "
                  f"{metrics.get('retweet_count', 0)} retweets, "
                  f"{metrics.get('reply_count', 0)} replies")
    else:
        print("\n⚠️  No tweets found!")
        print("Try a more popular movie or check API configuration.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LIVE TWITTER DATA TEST")
    print("=" * 60)
    
    # Run basic test
    asyncio.run(test_live_twitter_data())
    
    # Optional: Run custom search
    run_custom = input("\n\nRun custom search? (y/n): ").lower().strip()
    if run_custom == 'y':
        asyncio.run(test_specific_search())
    
    print("\n✅ All tests completed!")
