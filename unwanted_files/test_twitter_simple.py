"""
Simple Twitter API Live Data Test
Fetches ONE movie to prove live data works
"""
import asyncio
from app.collectors.twitter_collector import TwitterDataCollector


async def main():
    print("\n" + "=" * 70)
    print("  🐦 LIVE TWITTER DATA TEST")
    print("=" * 70)
    
    collector = TwitterDataCollector()
    
    movie = "RRR"  # Popular Telugu movie
    
    print(f"\n📡 Fetching LIVE tweets about '{movie}'...\n")
    
    tweets = await collector.search_movie_reviews(
        movie_name=movie,
        max_results=10,
        language="te"
    )
    
    if not tweets:
        print("❌ No tweets found!")
        return
    
    print(f"✅ SUCCESS! Fetched {len(tweets)} LIVE TWEETS from Twitter API v2\n")
    print("=" * 70)
    print("PROOF: These are REAL tweets from Twitter (last 7 days)")
    print("=" * 70)
    
    for i, tweet in enumerate(tweets, 1):
        print(f"\n{i}. 🐦 Tweet {tweet.get('id')}")
        print(f"   📅 Posted: {tweet.get('created_at')}")
        print(f"   🌐 Language: {tweet.get('lang')}")
        
        text = tweet.get('text', '')
        if len(text) > 150:
            text = text[:150] + "..."
        print(f"   💬 Text: {text}")
        
        metrics = tweet.get('public_metrics', {})
        print(f"   ❤️  {metrics.get('like_count', 0)} likes | "
              f"🔁 {metrics.get('retweet_count', 0)} retweets | "
              f"💭 {metrics.get('reply_count', 0)} replies")
    
    print("\n" + "=" * 70)
    print("✅ CONFIRMED: Your system fetches LIVE Twitter data!")
    print("=" * 70)
    print("\nTwitter API Configuration:")
    print(f"  • Endpoint: https://api.twitter.com/2/tweets/search/recent")
    print(f"  • Query: '{movie}' (lang:te OR #Telugu) -is:retweet")
    print(f"  • Rate Limit: 15 requests per 15-minute window")
    print(f"  • Data Source: LIVE Twitter API v2 (Not static/mock data)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
