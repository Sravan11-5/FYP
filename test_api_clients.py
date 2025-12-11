"""
Test TMDB and Twitter API Clients
Run this script to verify API integration is working
"""
import asyncio
from app.services.tmdb_client import tmdb_client
from app.services.twitter_client import twitter_client
from app.config import settings


def test_tmdb_client():
    """Test TMDB API client"""
    print("=" * 60)
    print("TESTING TMDB API CLIENT")
    print("=" * 60)
    
    # Check API key
    if not settings.TMDB_API_KEY:
        print("❌ TMDB API KEY not configured in .env file")
        print("   Please add: TMDB_API_KEY='your_api_key_here'")
        return False
    
    print(f"✅ TMDB API Key configured: {settings.TMDB_API_KEY[:10]}...")
    
    # Test 1: Search for a Telugu movie
    print("\n1. Testing movie search...")
    try:
        results = tmdb_client.search_movie("Baahubali", language="te")
        if results and results.get('results'):
            movie = results['results'][0]
            print(f"   ✅ Found movie: {movie.get('title')} ({movie.get('original_title')})")
            print(f"      TMDB ID: {movie.get('id')}")
            print(f"      Release: {movie.get('release_date')}")
            print(f"      Rating: {movie.get('vote_average')}")
            
            # Test poster URL
            poster_url = tmdb_client.get_poster_url(movie.get('poster_path'))
            print(f"      Poster: {poster_url}")
        else:
            print("   ⚠️  No results found")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Get movie details
    print("\n2. Testing movie details...")
    try:
        details = tmdb_client.get_movie_details(19995)  # Avatar movie ID
        if details:
            print(f"   ✅ Movie: {details.get('title')}")
            print(f"      Budget: ${details.get('budget'):,}")
            print(f"      Revenue: ${details.get('revenue'):,}")
            print(f"      Runtime: {details.get('runtime')} minutes")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 3: Get genres
    print("\n3. Testing genre list...")
    try:
        genres = tmdb_client.get_movie_genres(language="te")
        if genres and genres.get('genres'):
            print(f"   ✅ Found {len(genres['genres'])} genres:")
            for genre in genres['genres'][:5]:
                print(f"      - {genre.get('name')} (ID: {genre.get('id')})")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n✅ TMDB API CLIENT: ALL TESTS PASSED")
    return True


def test_twitter_client():
    """Test Twitter API client"""
    print("\n" + "=" * 60)
    print("TESTING TWITTER API CLIENT")
    print("=" * 60)
    
    # Check API credentials
    if not settings.TWITTER_BEARER_TOKEN:
        print("❌ TWITTER BEARER TOKEN not configured in .env file")
        print("   Please add: TWITTER_BEARER_TOKEN='your_bearer_token_here'")
        print("\n   To get your Bearer Token:")
        print("   1. Go to https://developer.twitter.com/")
        print("   2. Create a project and app")
        print("   3. Generate Bearer Token")
        return False
    
    print(f"✅ Twitter Bearer Token configured: {settings.TWITTER_BEARER_TOKEN[:20]}...")
    
    # Test 1: Search for Telugu movie reviews
    print("\n1. Testing Telugu movie review search...")
    try:
        tweets = twitter_client.search_movie_reviews(
            movie_name="Baahubali",
            max_results=10,
            days_back=180  # Search last 6 months
        )
        
        if tweets:
            print(f"   ✅ Found {len(tweets)} Telugu reviews")
            if tweets:
                tweet = tweets[0]
                print(f"      Sample tweet:")
                print(f"      - Text: {tweet['text'][:100]}...")
                print(f"      - Author: @{tweet['author_username']}")
                print(f"      - Likes: {tweet['likes']}")
                print(f"      - Created: {tweet['created_at']}")
        else:
            print("   ⚠️  No tweets found (this is okay - try a more popular movie)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Search Telugu tweets
    print("\n2. Testing Telugu keyword search...")
    try:
        tweets = twitter_client.search_telugu_tweets(
            keywords=["సినిమా", "మూవీ"],
            max_results=10,
            days_back=7
        )
        
        if tweets:
            print(f"   ✅ Found {len(tweets)} Telugu tweets")
            if tweets:
                print(f"      Sample: {tweets[0]['text'][:80]}...")
        else:
            print("   ⚠️  No tweets found")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n✅ TWITTER API CLIENT: ALL TESTS PASSED")
    return True


def main():
    """Run all tests"""
    print("\n🎬 TESTING API CLIENTS FOR TELUGU MOVIE RECOMMENDATION SYSTEM\n")
    
    # Test TMDB
    tmdb_success = test_tmdb_client()
    
    # Test Twitter
    twitter_success = test_twitter_client()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"TMDB API Client: {'✅ PASSED' if tmdb_success else '❌ FAILED'}")
    print(f"Twitter API Client: {'✅ PASSED' if twitter_success else '❌ FAILED'}")
    
    if tmdb_success and twitter_success:
        print("\n🎉 ALL API CLIENTS WORKING! Ready for Task 5.")
    else:
        print("\n⚠️  Some tests failed. Please configure missing API keys in .env file.")
        print("\nRequired environment variables:")
        print("  - TMDB_API_KEY=your_tmdb_api_key")
        print("  - TWITTER_BEARER_TOKEN=your_twitter_bearer_token")
        print("  - TWITTER_API_KEY=your_twitter_api_key (optional)")
        print("  - TWITTER_API_SECRET=your_twitter_api_secret (optional)")


if __name__ == "__main__":
    main()
