"""
Test TMDB Reviews for Dangal
"""
import asyncio
from app.collectors.tmdb_collector import TMDBDataCollector


async def test_dangal_reviews():
    tmdb = TMDBDataCollector()
    
    print("\n" + "="*70)
    print("  🎬 TESTING TMDB REVIEWS")
    print("="*70)
    
    # Test with popular movies
    test_movies = ["RRR", "Dangal", "Baahubali"]
    
    for movie_name in test_movies:
        print(f"\n{'='*70}")
        print(f"  🔍 TESTING: {movie_name}")
        print(f"{'='*70}")
        
        movies = await tmdb.search_movie(movie_name)
        
        if not movies:
            print("❌ Movie not found!")
            continue
        
        movie = movies[0]
        print(f"\n✅ Found: {movie['title']} (ID: {movie['id']})")
        print(f"   Release Date: {movie.get('release_date')}")
        print(f"   Rating: {movie.get('vote_average')}/10")
        
        # Get reviews
        print(f"\n📝 Fetching reviews from TMDB...")
        reviews = await tmdb.get_movie_reviews(movie['id'], max_results=5)
        
        if not reviews:
            print("   ❌ No reviews found!")
            continue
        
        print(f"\n   ✅ Found {len(reviews)} reviews!")
        
        for i, review in enumerate(reviews, 1):
            author = review.get('author', 'Unknown')
            content = review.get('content', '')
            rating = review.get('author_details', {}).get('rating', 'N/A')
            created_at = review.get('created_at', 'N/A')[:10]
            
            print(f"\n   {i}. 👤 {author} | ⭐ {rating}/10 | 📅 {created_at}")
            
            # Show first 200 characters
            if len(content) > 200:
                print(f"      {content[:200]}... ({len(content)} chars total)")
            else:
                print(f"      {content}")
        
        await asyncio.sleep(1)  # Rate limit respect


if __name__ == "__main__":
    asyncio.run(test_dangal_reviews())
