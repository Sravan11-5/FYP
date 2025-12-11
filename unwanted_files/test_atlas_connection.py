"""
Test MongoDB Atlas Cloud Connection
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

async def test_atlas_connection():
    """Test MongoDB Atlas connection"""
    print("=" * 60)
    print("TESTING MONGODB ATLAS CLOUD CONNECTION")
    print("=" * 60)
    
    connection_string = "mongodb+srv://ProjectRefresh:projectrefresh@cluster0.oj8eytc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    
    # Connect to MongoDB Atlas
    print("\n1. Connecting to MongoDB Atlas...")
    try:
        client = AsyncIOMotorClient(connection_string)
        # Test connection
        await client.admin.command('ping')
        print("   ✅ Successfully connected to MongoDB Atlas!")
        
        db = client["telugu_movie_recommender"]
        print(f"   ✅ Using database: telugu_movie_recommender")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return
    
    # List existing collections
    print("\n2. Checking existing collections...")
    try:
        collections = await db.list_collection_names()
        if collections:
            print(f"   ✅ Found {len(collections)} collection(s): {collections}")
        else:
            print("   ℹ️  No collections yet (will be created automatically)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Create a test movie
    print("\n3. Testing CREATE operation...")
    try:
        test_movie = {
            "tmdb_id": 888888,
            "title": "తెలుగు టెస్ట్ మూవీ",  # Telugu Test Movie
            "original_title": "Telugu Test Movie",
            "genres": ["డ్రామా", "యాక్షన్"],  # Drama, Action
            "rating": 8.0,
            "poster_url": "https://example.com/test.jpg",
            "overview": "మొంగోడీబీ అట్లాస్ కనెక్షన్ టెస్ట్",  # MongoDB Atlas connection test
            "release_date": "2024-12-05",
            "avg_sentiment_score": 0.0,
            "total_reviews": 0,
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "domain_scores": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = await db.movies.insert_one(test_movie)
        print(f"   ✅ Test movie created with ID: {result.inserted_id}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Read the movie
    print("\n4. Testing READ operation...")
    try:
        movie = await db.movies.find_one({"tmdb_id": 888888})
        if movie:
            print(f"   ✅ Movie retrieved successfully!")
            print(f"      - Title: {movie['title']}")
            print(f"      - Rating: {movie['rating']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Create indexes
    print("\n5. Creating database indexes...")
    try:
        # Movies collection indexes
        await db.movies.create_index("tmdb_id", unique=True)
        await db.movies.create_index("title")
        await db.movies.create_index("genres")
        await db.movies.create_index("rating")
        await db.movies.create_index([("genres", 1), ("rating", -1)])
        
        # Reviews collection indexes
        await db.reviews.create_index("movie_id")
        await db.reviews.create_index("tmdb_id")
        await db.reviews.create_index("tweet_id", unique=True)
        await db.reviews.create_index([("movie_id", 1), ("sentiment_score", -1)])
        
        # User searches indexes
        await db.user_searches.create_index("search_query")
        await db.user_searches.create_index("tmdb_id")
        await db.user_searches.create_index("searched_at")
        
        # Genres indexes
        await db.genres.create_index("tmdb_genre_id", unique=True)
        await db.genres.create_index("name")
        
        print("   ✅ All indexes created successfully!")
    except Exception as e:
        print(f"   ❌ Error creating indexes: {e}")
    
    # Verify indexes
    print("\n6. Verifying indexes...")
    collections_to_check = ['movies', 'reviews', 'user_searches', 'genres']
    for coll in collections_to_check:
        try:
            indexes = await db[coll].index_information()
            print(f"   ✅ {coll}: {len(indexes)} index(es)")
        except Exception as e:
            print(f"   ❌ Error for {coll}: {e}")
    
    # Clean up test data
    print("\n7. Cleaning up test data...")
    try:
        result = await db.movies.delete_one({"tmdb_id": 888888})
        print(f"   ✅ Test movie deleted: {result.deleted_count} document(s)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Close connection
    client.close()
    
    print("\n" + "=" * 60)
    print("✅ MONGODB ATLAS CONNECTION TEST COMPLETE!")
    print("=" * 60)
    print("\n🎉 Your cloud database is ready!")
    print("📊 Database: telugu_movie_recommender")
    print("🌐 Cloud: MongoDB Atlas")
    print("✅ All collections and indexes configured!")

if __name__ == "__main__":
    asyncio.run(test_atlas_connection())
