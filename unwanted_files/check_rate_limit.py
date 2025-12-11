"""
Check Twitter API Rate Limit Status
Shows when you can start collecting data
"""
import asyncio
import aiohttp
from app.config import settings
from datetime import datetime


async def check_rate_limit_status():
    """Check current Twitter API rate limit status"""
    
    print("\n" + "="*70)
    print("  🐦 TWITTER API RATE LIMIT STATUS CHECK")
    print("="*70)
    
    endpoint = "https://api.twitter.com/2/tweets/search/recent"
    headers = {
        "Authorization": f"Bearer {settings.TWITTER_BEARER_TOKEN}"
    }
    
    # Make a minimal test query
    params = {
        "query": "test",
        "max_results": 10
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, headers=headers, params=params) as response:
                
                # Get rate limit headers
                remaining = response.headers.get('x-rate-limit-remaining')
                limit = response.headers.get('x-rate-limit-limit')
                reset = response.headers.get('x-rate-limit-reset')
                
                if remaining and limit and reset:
                    reset_time = datetime.fromtimestamp(int(reset))
                    current_time = datetime.now()
                    time_until_reset = (reset_time - current_time).total_seconds()
                    
                    print(f"\n📊 Rate Limit Status:")
                    print(f"   • Limit: {limit} requests per 15-minute window")
                    print(f"   • Remaining: {remaining} requests")
                    print(f"   • Used: {int(limit) - int(remaining)} requests")
                    print(f"   • Reset time: {reset_time.strftime('%I:%M:%S %p')}")
                    
                    if int(remaining) > 0:
                        print(f"\n✅ You can make {remaining} more API calls RIGHT NOW!")
                        print(f"   Start collection immediately!")
                    else:
                        minutes_wait = time_until_reset / 60
                        print(f"\n⏳ Rate limit exhausted!")
                        print(f"   Wait {minutes_wait:.1f} minutes until {reset_time.strftime('%I:%M:%S %p')}")
                        print(f"   Then you'll have {limit} fresh API calls")
                    
                    if response.status == 200:
                        print(f"\n✅ API is working correctly!")
                    elif response.status == 429:
                        print(f"\n⚠️  Currently rate limited (429)")
                else:
                    print("\n⚠️  Could not read rate limit headers")
                    if response.status == 429:
                        print("   Status: Rate limited (wait 15 minutes)")
                    else:
                        print(f"   Status: {response.status}")
                
    except Exception as e:
        print(f"\n❌ Error checking rate limit: {e}")
    
    print("\n" + "="*70)
    print("💡 TIP: Wait for 'Remaining' to show 15, then start collection")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(check_rate_limit_status())
