"""
Check for a specific movie that's likely NOT in database
"""
from pymongo import MongoClient
from app.config import settings

client = MongoClient(settings.MONGODB_URL)
db = client[settings.MONGODB_DB_NAME]

# Test these popular movies - at least one should be missing
test_movies = [
    "The Shawshank Redemption",
    "The Dark Knight", 
    "Forrest Gump",
    "The Matrix",
    "Interstellar",
    "Fight Club",
    "Pulp Fiction",
    "The Godfather"
]

print("\n" + "="*80)
print("🔍 CHECKING WHICH MOVIES ARE NOT IN DATABASE")
print("="*80)

not_in_db = []

for movie in test_movies:
    exists = db.movies.find_one({"title": {"$regex": movie, "$options": "i"}})
    if exists:
        print(f"❌ {movie:<30} - Already in database")
    else:
        print(f"✅ {movie:<30} - NOT in database (good for testing!)")
        not_in_db.append(movie)

print("\n" + "="*80)
print("📋 MOVIES NOT IN DATABASE (Pick any to test on-demand fetch):")
print("="*80)

for movie in not_in_db:
    print(f"  • {movie}")

if not_in_db:
    print(f"\n🎯 RECOMMENDED TEST: '{not_in_db[0]}'")
    print(f"   This will trigger on-demand fetch from TMDB!")
else:
    print("\n⚠️  All test movies are already in database!")
    print("   Try searching for: 'Avatar' or 'Titanic'")

print("\n" + "="*80)
