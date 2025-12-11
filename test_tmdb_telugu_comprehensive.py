"""
Test TMDB Telugu Reviews with Correct Language Code
Testing with te-IN and te codes
"""
import asyncio
import aiohttp
from app.collectors.tmdb_collector import TMDBDataCollector
from app.config import settings


async def test_telugu_reviews_comprehensive():
    """Test TMDB reviews with different language codes"""
    
    collector = TMDBDataCollector()
    
    # Test with popular Telugu movies
    test_movies = [
        ("RRR", "Most popular recent Telugu movie"),
        ("Baahubali", "Blockbuster Telugu movie"),
        ("Pushpa", "Recent hit Telugu movie"),
        ("Arjun Reddy", "Critically acclaimed Telugu movie"),
        ("Eega", "Award-winning Telugu movie")
    ]
    
    print("\n" + "="*70)
    print("  COMPREHENSIVE TMDB TELUGU REVIEW TEST")
    print("="*70)
    
    for movie_name, description in test_movies:
        print(f"\n{'='*70}")
        print(f"Testing: {movie_name} ({description})")
        print("="*70)
        
        # Search for movie
        movies = await collector.search_movie(movie_name, language="te")
        
        if not movies:
            print(f"❌ Movie '{movie_name}' not found")
            continue
        
        movie = movies[0]
        tmdb_id = movie['id']
        title = movie.get('title', movie.get('original_title'))
        
        print(f"✅ Found: {title} (ID: {tmdb_id})")
        
        # Test different language codes
        language_codes = [
            ("en", "English (default)"),
            ("te", "Telugu (ISO 639-1)"),
            ("te-IN", "Telugu India (ISO 639-1 + region)"),
            ("", "No language filter")
        ]
        
        for lang_code, lang_desc in language_codes:
            print(f"\n  Testing with language='{lang_code}' ({lang_desc}):")
            
            endpoint = f"https://api.themoviedb.org/3/movie/{tmdb_id}/reviews"
            params = {}
            if lang_code:
                params["language"] = lang_code
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        endpoint,
                        headers={
                            "Authorization": f"Bearer {settings.TMDB_API_KEY}",
                            "Content-Type": "application/json;charset=utf-8"
                        },
                        params=params
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            reviews = data.get('results', [])
                            
                            if reviews:
                                print(f"    ✅ Found {len(reviews)} reviews!")
                                
                                # Show first review details
                                review = reviews[0]
                                author = review.get('author', 'Unknown')
                                content = review.get('content', '')
                                rating = review.get('author_details', {}).get('rating')
                                iso_lang = review.get('iso_639_1', 'unknown')
                                
                                print(f"       Author: {author}")
                                print(f"       Rating: {rating}/10" if rating else "       Rating: Not rated")
                                print(f"       Language Code: {iso_lang}")
                                print(f"       Content (first 150 chars): {content[:150]}...")
                                
                                # Check if any review is actually in Telugu
                                telugu_reviews = [r for r in reviews if r.get('iso_639_1') == 'te']
                                if telugu_reviews:
                                    print(f"    🎯 Found {len(telugu_reviews)} Telugu language reviews!")
                                else:
                                    print(f"    ⚠️  All {len(reviews)} reviews are in other languages (mostly English)")
                            else:
                                print(f"    ❌ No reviews found")
                        else:
                            print(f"    ❌ API Error: {response.status}")
                            
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        print()


async def test_review_statistics():
    """Get statistics on review availability"""
    
    collector = TMDBDataCollector()
    
    print("\n" + "="*70)
    print("  TMDB REVIEW STATISTICS")
    print("="*70)
    
    # Test top Telugu movies
    movies_to_check = [
        "Baahubali", "RRR", "Pushpa", "KGF", "Arjun Reddy",
        "Eega", "Rangasthalam", "Ala Vaikunthapurramuloo", "Sarileru Neekevvaru"
    ]
    
    total_movies = 0
    movies_with_reviews = 0
    total_reviews = 0
    total_telugu_reviews = 0
    
    for movie_name in movies_to_check:
        movies = await collector.search_movie(movie_name, language="te")
        if not movies:
            continue
        
        total_movies += 1
        tmdb_id = movies[0]['id']
        
        # Get reviews (no language filter to get all)
        endpoint = f"https://api.themoviedb.org/3/movie/{tmdb_id}/reviews"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {settings.TMDB_API_KEY}",
                        "Content-Type": "application/json;charset=utf-8"
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        reviews = data.get('results', [])
                        
                        if reviews:
                            movies_with_reviews += 1
                            total_reviews += len(reviews)
                            
                            telugu_count = len([r for r in reviews if r.get('iso_639_1') == 'te'])
                            total_telugu_reviews += telugu_count
                            
                            print(f"  {movie_name}: {len(reviews)} reviews "
                                  f"({telugu_count} in Telugu)")
        except:
            pass
    
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"  Total movies checked: {total_movies}")
    print(f"  Movies with ANY reviews: {movies_with_reviews}")
    print(f"  Total reviews found: {total_reviews}")
    print(f"  Telugu reviews: {total_telugu_reviews}")
    print(f"  English/Other reviews: {total_reviews - total_telugu_reviews}")
    
    if total_reviews > 0:
        telugu_percentage = (total_telugu_reviews / total_reviews) * 100
        print(f"\n  📊 Telugu review percentage: {telugu_percentage:.1f}%")
    
    print("\n💡 CONCLUSION:")
    if total_telugu_reviews == 0:
        print("  ❌ TMDB has ZERO Telugu language reviews for these movies")
        print("  ⚠️  All reviews are in English or other languages")
        print("  ✅ TWITTER API IS REQUIRED for actual Telugu reviews")
    else:
        print(f"  ✅ Found some Telugu reviews, but only {telugu_percentage:.1f}% of total")
        print("  ⚠️  Still need Twitter API for more Telugu content")


if __name__ == "__main__":
    print("\n🔍 Testing TMDB Telugu Review Availability...")
    asyncio.run(test_telugu_reviews_comprehensive())
    asyncio.run(test_review_statistics())
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
