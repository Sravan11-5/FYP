"""
Check if Telugu movies have reviews in Telugu/other languages vs English
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
BASE_URL = "https://api.themoviedb.org/3"

def check_movie_reviews_all_languages(tmdb_id, movie_title):
    """Check reviews in ALL languages for a movie"""
    print(f"\n{'='*80}")
    print(f"🎬 Movie: {movie_title} (ID: {tmdb_id})")
    print('='*80)
    
    # Get reviews WITHOUT language filter (all languages)
    url = f"{BASE_URL}/movie/{tmdb_id}/reviews"
    headers = {
        'Authorization': f'Bearer {TMDB_API_KEY}',
        'accept': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    all_reviews = []
    en_reviews = []
    
    if response.status_code == 200:
        all_reviews = response.json().get('results', [])
        print(f"\n📊 TOTAL REVIEWS (ALL LANGUAGES): {len(all_reviews)}")
        
        if all_reviews:
            # Count by language
            language_counts = {}
            for review in all_reviews:
                # Get ISO language code from review author details
                lang = review.get('author_details', {}).get('rating') 
                # Actually, let's get the actual language from the content
                # TMDB doesn't explicitly mark review language, so check content
                content = review.get('content', '')[:100]
                author = review.get('author', 'Unknown')
                
                print(f"\n  👤 Author: {author}")
                print(f"  📝 Content preview: {content}...")
                print(f"  ⭐ Rating: {review.get('author_details', {}).get('rating', 'N/A')}")
            
            print(f"\n✅ This movie HAS {len(all_reviews)} review(s) total")
        else:
            print("\n❌ NO REVIEWS FOUND in any language")
    else:
        print(f"❌ API Error: {response.status_code}")
    
    # Now check ENGLISH reviews specifically
    url_en = f"{BASE_URL}/movie/{tmdb_id}/reviews"
    params_en = {
        'language': 'en-US'  # English only
    }
    
    response_en = requests.get(url_en, headers=headers, params=params_en)
    if response_en.status_code == 200:
        en_reviews = response_en.json().get('results', [])
        print(f"\n📊 ENGLISH REVIEWS ONLY: {len(en_reviews)}")
        
        if en_reviews:
            print(f"✅ This movie has {len(en_reviews)} English review(s)")
        else:
            print("❌ NO ENGLISH REVIEWS (but may have reviews in other languages)")
    
    return len(all_reviews), len(en_reviews)

def main():
    print("\n" + "="*80)
    print("🔍 CHECKING TELUGU MOVIE REVIEWS - ALL LANGUAGES vs ENGLISH ONLY")
    print("="*80)
    
    # Test with popular Telugu movies from different years
    test_movies = [
        (1010581, "Pushpa 2: The Rule", 2024),  # Very recent, no reviews found
        (951491, "Devara: Part 1", 2024),  # Found 1 review
        (1094844, "Kalki 2898 AD", 2024),  # Found 3 reviews
        (579974, "RRR", 2022),  # Very popular internationally
        (426426, "Baahubali 2: The Conclusion", 2017),  # Blockbuster
        (369885, "Baahubali: The Beginning", 2015),  # Blockbuster
    ]
    
    total_results = []
    
    for tmdb_id, title, year in test_movies:
        all_count, en_count = check_movie_reviews_all_languages(tmdb_id, f"{title} ({year})")
        total_results.append({
            'title': title,
            'year': year,
            'all_reviews': all_count,
            'english_reviews': en_count
        })
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY COMPARISON")
    print("="*80)
    print(f"\n{'Movie':<40} {'Year':<6} {'All Reviews':<12} {'English Reviews'}")
    print("-" * 80)
    
    for result in total_results:
        print(f"{result['title']:<40} {result['year']:<6} {result['all_reviews']:<12} {result['english_reviews']}")
    
    # Analysis
    print("\n" + "="*80)
    print("🎯 CONCLUSION")
    print("="*80)
    
    total_all = sum(r['all_reviews'] for r in total_results)
    total_en = sum(r['english_reviews'] for r in total_results)
    
    print(f"\n📈 Total reviews across {len(test_movies)} popular Telugu movies:")
    print(f"   • ALL LANGUAGES: {total_all} reviews")
    print(f"   • ENGLISH ONLY: {total_en} reviews")
    
    if total_all > total_en:
        print(f"\n💡 INSIGHT: Telugu movies DO have reviews, but {total_all - total_en} reviews are in NON-ENGLISH languages!")
        print("   This suggests reviews may be in Telugu, Hindi, or other Indian languages.")
    elif total_all == 0:
        print("\n💡 INSIGHT: These Telugu movies have NO reviews at all on TMDB")
    else:
        print("\n💡 INSIGHT: All reviews found are in English")

if __name__ == "__main__":
    main()
