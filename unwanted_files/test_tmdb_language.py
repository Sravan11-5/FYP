"""
Test TMDB Reviews in Different Languages
Check if TMDB provides Telugu reviews
"""
import asyncio
from app.collectors.tmdb_collector import TMDBDataCollector


async def test_language_reviews():
    collector = TMDBDataCollector()
    
    # Test with RRR (popular Telugu movie)
    movie_name = "RRR"
    
    print("\n" + "="*70)
    print(f"  Testing TMDB Reviews for: {movie_name}")
    print("="*70)
    
    # Search for movie
    movies = await collector.search_movie(movie_name, language="te")
    
    if not movies:
        print("❌ Movie not found")
        return
    
    movie = movies[0]
    tmdb_id = movie['id']
    title = movie['title']
    
    print(f"\n✅ Found: {title} (ID: {tmdb_id})")
    
    # Test different language parameters
    languages = [
        ("en", "English"),
        ("te", "Telugu"),
        ("hi", "Hindi"),
        ("ta", "Tamil")
    ]
    
    for lang_code, lang_name in languages:
        print(f"\n{'='*70}")
        print(f"  Testing {lang_name} Reviews (lang={lang_code})")
        print("="*70)
        
        # Modify the API call to test language parameter
        endpoint = f"https://api.themoviedb.org/3/movie/{tmdb_id}/reviews"
        params = {"language": lang_code}
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                endpoint,
                headers=collector.headers,
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    reviews = data.get('results', [])
                    
                    if reviews:
                        print(f"✅ Found {len(reviews)} reviews")
                        
                        # Show first review
                        review = reviews[0]
                        author = review.get('author', 'Unknown')
                        content = review.get('content', '')
                        rating = review.get('author_details', {}).get('rating', 'N/A')
                        
                        print(f"\nSample Review:")
                        print(f"  Author: {author}")
                        print(f"  Rating: {rating}/10")
                        print(f"  Language: {review.get('iso_639_1', 'unknown')}")
                        print(f"  Content: {content[:200]}...")
                    else:
                        print(f"⚠️  No reviews found for {lang_name}")
                else:
                    print(f"❌ API Error: {response.status}")


async def check_tmdb_language_support():
    """Check what languages TMDB API supports"""
    
    print("\n" + "="*70)
    print("  TMDB API Language Support Check")
    print("="*70)
    
    print("\n📋 Important TMDB API Facts:")
    print("   • TMDB reviews are USER-GENERATED globally")
    print("   • Most reviews are in ENGLISH (international audience)")
    print("   • 'language' parameter filters MOVIE metadata, NOT review language")
    print("   • Review language depends on what users wrote")
    print("   • Telugu movies may have English reviews from global users")
    
    print("\n💡 Recommendation:")
    print("   For Telugu-specific reviews, Twitter API is better because:")
    print("   1. You can filter by 'lang:te' to get Telugu language tweets")
    print("   2. Local Telugu audience discusses movies on Twitter")
    print("   3. More authentic Telugu sentiment")
    print("   4. But faces rate limits")
    
    print("\n   For English reviews, TMDB is better because:")
    print("   1. No rate limits")
    print("   2. High-quality, moderated reviews")
    print("   3. Structured data with ratings")
    print("   4. But mostly English language")


if __name__ == "__main__":
    asyncio.run(test_language_reviews())
    asyncio.run(check_tmdb_language_support())
