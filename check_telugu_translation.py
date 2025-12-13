"""
Check if all reviews are translated to Telugu
"""
from pymongo import MongoClient
from app.config import settings

client = MongoClient(settings.MONGODB_URL)
db = client[settings.MONGODB_DB_NAME]

print("\n" + "="*80)
print("🔍 CHECKING TELUGU TRANSLATION STATUS")
print("="*80)

# Get all movies
movies = list(db.movies.find())
total_movies = len(movies)
total_reviews = 0
telugu_reviews = 0
english_reviews = 0
missing_translation = 0

print(f"\n📊 Total movies: {total_movies}")

for movie in movies:
    movie_title = movie.get('title', 'Unknown')
    reviews = movie.get('reviews', [])
    
    for review in reviews:
        total_reviews += 1
        
        # Check if review has Telugu text
        telugu_text = review.get('text', '')
        original_text = review.get('original_text', '')
        language = review.get('language', '')
        
        # Detect Telugu characters (Unicode range)
        has_telugu = any('\u0c00' <= char <= '\u0c7f' for char in telugu_text)
        
        if language == 'te' and has_telugu:
            telugu_reviews += 1
        elif language == 'en' or not has_telugu:
            english_reviews += 1
            print(f"  ⚠️  {movie_title}: Review NOT in Telugu (lang={language})")
        
        if not telugu_text or not original_text:
            missing_translation += 1
            print(f"  ❌ {movie_title}: Missing text field!")

print("\n" + "="*80)
print("📈 TRANSLATION SUMMARY")
print("="*80)
print(f"\n✅ Total reviews: {total_reviews}")
print(f"✅ Telugu reviews: {telugu_reviews} ({telugu_reviews/total_reviews*100:.1f}%)")
print(f"❌ English reviews: {english_reviews} ({english_reviews/total_reviews*100:.1f}%)")
print(f"❌ Missing translation: {missing_translation}")

if telugu_reviews == total_reviews:
    print("\n🎉 SUCCESS! All reviews are properly translated to Telugu!")
elif telugu_reviews > 0:
    print(f"\n⚠️  WARNING: {english_reviews} reviews are NOT translated to Telugu")
else:
    print("\n❌ CRITICAL: NO reviews are translated to Telugu!")

# Show sample Telugu review
print("\n" + "="*80)
print("📝 SAMPLE TELUGU REVIEW")
print("="*80)

sample_movie = db.movies.find_one({"has_reviews": True})
if sample_movie and sample_movie.get('reviews'):
    sample_review = sample_movie['reviews'][0]
    print(f"\nMovie: {sample_movie.get('title')}")
    print(f"\nTelugu Text ({len(sample_review.get('text', ''))} chars):")
    print(f"  {sample_review.get('text', '')[:200]}...")
    print(f"\nOriginal English ({len(sample_review.get('original_text', ''))} chars):")
    print(f"  {sample_review.get('original_text', '')[:200]}...")
    print(f"\nLanguage field: {sample_review.get('language')}")

print("\n" + "="*80)
