"""
Quick check of database status
"""
from pymongo import MongoClient
from app.config import settings

client = MongoClient(settings.MONGODB_URL)
db = client[settings.MONGODB_DB_NAME]

# Count movies
movie_count = db.movies.count_documents({})
movies_with_reviews = db.movies.count_documents({"has_reviews": True})

print("\n" + "="*80)
print("📊 DATABASE STATUS")
print("="*80)
print(f"\n✅ Total movies in database: {movie_count}")
print(f"✅ Movies with reviews: {movies_with_reviews}")

if movie_count > 0:
    print(f"\n📝 Sample movies:")
    for movie in db.movies.find().limit(10):
        title = movie.get('title', 'Unknown')
        reviews = movie.get('review_count', 0)
        year = movie.get('release_date', '')[:4]
        print(f"  • {title[:50]:<50} ({year}) - {reviews} reviews")
    
    # Show language distribution
    print(f"\n🌍 Movie language distribution:")
    pipeline = [
        {"$group": {"_id": "$original_language", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    for lang in db.movies.aggregate(pipeline):
        lang_code = lang['_id'] if lang['_id'] else 'unknown'
        count = lang['count']
        print(f"  • {lang_code}: {count} movies")

print("\n" + "="*80)
