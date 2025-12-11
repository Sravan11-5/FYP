"""
Test Task 16: Recommendation System
Validates the context-aware recommendation engine
"""

import asyncio
import sys
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

async def test_recommendation_system():
    """Test the recommendation system with sample data"""
    
    print("=" * 70)
    print("TASK 16: TESTING RECOMMENDATION SYSTEM")
    print("=" * 70)
    print()
    
    # Connect to database
    print("1. Connecting to MongoDB...")
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]
    
    try:
        await client.admin.command('ping')
        print("   ✓ Connected to MongoDB")
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
        return False
    
    # Check movies
    print("\n2. Checking available movies...")
    movies_count = await db.movies.count_documents({})
    print(f"   Total movies in database: {movies_count}")
    
    if movies_count == 0:
        print("   ✗ No movies found in database")
        return False
    
    # Get sample movies
    movies = await db.movies.find().limit(5).to_list(length=5)
    print(f"\n   Sample movies:")
    for movie in movies:
        print(f"   - {movie['title']} (Genres: {', '.join(movie.get('genres', []))})")
    
    # Check reviews
    print("\n3. Checking reviews...")
    reviews_count = await db.reviews.count_documents({})
    print(f"   Total reviews in database: {reviews_count}")
    
    if reviews_count == 0:
        print("   ⚠ Warning: No reviews found. Recommendations will not work properly.")
        print("   Need to collect review data first.")
        return False
    
    # Test recommendation engine
    print("\n4. Testing recommendation engine...")
    
    try:
        from app.ml.recommendation_engine import get_recommendation_engine
        
        engine = get_recommendation_engine()
        print("   ✓ Recommendation engine initialized")
        
        # Try to get recommendations for first movie
        test_movie = movies[0]
        print(f"\n5. Generating recommendations for: {test_movie['title']}")
        
        recommendations = await engine.get_recommendations(
            movie_name=test_movie['title'],
            max_results=5,
            min_sentiment_score=0.5
        )
        
        if recommendations:
            print(f"   ✓ Generated {len(recommendations)} recommendations")
            print(f"\n   Top 3 recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec['title']}")
                print(f"      Score: {rec['recommendation_score']:.4f}")
                print(f"      Sentiment: {rec['sentiment_analysis']['avg_positive_score']:.4f}")
                print(f"      Similarity: {rec['similarity_score']:.4f}")
                print(f"      Reasoning: {rec['reasoning']}")
                print()
        else:
            print("   ✗ No recommendations generated")
            return False
        
        print("=" * 70)
        print("✓ TASK 16: RECOMMENDATION SYSTEM TEST PASSED")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        client.close()


async def check_system_requirements():
    """Check if system has necessary data for recommendations"""
    
    print("=" * 70)
    print("SYSTEM REQUIREMENTS CHECK")
    print("=" * 70)
    print()
    
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]
    
    try:
        # Check movies
        movies_count = await db.movies.count_documents({})
        print(f"Movies in database: {movies_count}")
        
        # Check reviews
        reviews_count = await db.reviews.count_documents({})
        print(f"Reviews in database: {reviews_count}")
        
        # Check movies with reviews
        pipeline = [
            {
                "$lookup": {
                    "from": "reviews",
                    "localField": "_id",
                    "foreignField": "movie_id",
                    "as": "reviews"
                }
            },
            {
                "$match": {
                    "reviews": {"$ne": []}
                }
            },
            {
                "$count": "movies_with_reviews"
            }
        ]
        
        result = await db.movies.aggregate(pipeline).to_list(length=1)
        movies_with_reviews = result[0]["movies_with_reviews"] if result else 0
        
        print(f"Movies with reviews: {movies_with_reviews}")
        print()
        
        if movies_count == 0:
            print("✗ CRITICAL: No movies in database")
            print("  Solution: Run data collection scripts to populate movies")
            return False
        
        if reviews_count == 0:
            print("✗ CRITICAL: No reviews in database")
            print("  Solution: Run review collection scripts to populate reviews")
            return False
        
        if movies_with_reviews < 2:
            print("✗ CRITICAL: Insufficient movies with reviews")
            print("  Solution: Need at least 2 movies with reviews for recommendations")
            return False
        
        print("✓ All requirements met")
        return True
        
    finally:
        client.close()


if __name__ == "__main__":
    print("\n")
    
    # Check requirements first
    requirements_met = asyncio.run(check_system_requirements())
    
    if not requirements_met:
        print("\n⚠ System requirements not met. Cannot test recommendations.")
        print("  Please run data collection first:")
        print("  1. Collect movies from TMDB")
        print("  2. Collect reviews from Twitter")
        print()
        sys.exit(1)
    
    print("\n")
    
    # Run test
    success = asyncio.run(test_recommendation_system())
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
