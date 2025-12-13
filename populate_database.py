"""
Populate database with English movies from TMDB across different genres.
Fetches movies year-by-year (2024->2023->2022...) until 200-300 movies with reviews.
Reviews are translated from English to Telugu for the recommendation system.
"""
import asyncio
import sys
import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient
from app.collectors.tmdb_collector import TMDBDataCollector
from app.utils.translator import get_translator
from app.config import settings

# Genre mapping: TMDB genre IDs
GENRES = {
    "Action": 28,
    "Drama": 18,
    "Thriller": 53,
    "Comedy": 35,
    "Romance": 10749,
    "Science Fiction": 878,
    "Horror": 27,
    "Crime": 80,
    "Adventure": 12,
    "Family": 10751
}

# Target: 200-300 movies with reviews
TARGET_MIN = 200
TARGET_MAX = 300

# Year ranges to search (most recent first)
YEAR_RANGES = [
    (2023, 2024),  # 2023-2024
    (2021, 2022),  # 2021-2022
    (2019, 2020),  # 2019-2020
    (2017, 2018),  # 2017-2018
    (2015, 2016),  # 2015-2016
]

async def populate_database():
    """Fetch and store movies year-by-year until target reached."""
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]
    
    print("🎬 Starting year-by-year database population...")
    print(f"📊 Target: {TARGET_MIN}-{TARGET_MAX} movies with reviews\n")
    
    # Initialize collectors
    tmdb = TMDBDataCollector()
    translator = get_translator()
    
    total_added = 0
    total_skipped = 0
    
    # Iterate through year ranges (most recent first)
    for year_start, year_end in YEAR_RANGES:
        if total_added >= TARGET_MAX:
            print(f"\n✅ Target reached! {total_added} movies collected.")
            break
        
        print(f"\n{'='*70}")
        print(f"📅 Fetching movies from {year_start}-{year_end}")
        print(f"{'='*70}")
        
        # Try each genre in this year range
        for genre_name, genre_id in GENRES.items():
            if total_added >= TARGET_MAX:
                break
            
            print(f"\n🎭 Genre: {genre_name} ({year_start}-{year_end})")
            
            try:
                # Discover English movies in this genre and year range
                url = f"{tmdb.BASE_URL}/discover/movie"
                params = {
                    "with_original_language": "en",  # English movies
                    "with_genres": genre_id,
                    "primary_release_date.gte": f"{year_start}-01-01",
                    "primary_release_date.lte": f"{year_end}-12-31",
                    "sort_by": "popularity.desc",
                    "page": 1,
                    "vote_count.gte": 5  # At least 5 votes
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=tmdb.headers, params=params) as response:
                        if response.status != 200:
                            print(f"  ❌ Failed to fetch {genre_name} movies")
                            continue
                        
                        data = await response.json()
                        movies = data.get("results", [])[:20]  # Check up to 20 per genre
                    
                    if not movies:
                        print(f"  ⚠️  No movies found")
                        continue
                    
                    print(f"  📋 Found {len(movies)} movies to check")
                    
                    for idx, movie_data in enumerate(movies, 1):
                        if total_added >= TARGET_MAX:
                            break
                        
                        tmdb_id = movie_data.get("id")
                        title = movie_data.get("title") or movie_data.get("original_title")
                        
                        # Check if already exists
                        existing = await db.movies.find_one({"tmdb_id": tmdb_id})
                        if existing:
                            total_skipped += 1
                            continue
                        
                        print(f"    📥 [{idx}] Checking: {title}")
                        
                        try:
                            # Get movie details
                            movie_details = await tmdb.get_movie_details(tmdb_id)
                            
                            # Get reviews and translate
                            print(f"        🔄 Fetching reviews...")
                            reviews = await tmdb.get_reviews_and_translate(tmdb_id, max_reviews=20)
                            
                            # Skip if no reviews
                            if not reviews or len(reviews) == 0:
                                print(f"        ⚠️  No reviews - skipping")
                                total_skipped += 1
                                continue
                            
                            print(f"        ✅ Got {len(reviews)} review(s)")
                            
                            # Get primary genre
                            genres = movie_details.get("genres", [])
                            primary_genre = genres[0]["name"] if genres else genre_name
                            
                            # Prepare movie document with English title
                            english_title = movie_details.get("title") or movie_details.get("original_title")
                            
                            movie_doc = {
                                "tmdb_id": tmdb_id,
                                "title": english_title,  # English title
                                "original_title": movie_details.get("original_title"),
                                "genre": primary_genre,
                                "genres": [g["name"] for g in genres],
                                "rating": movie_details.get("vote_average"),
                                "overview": movie_details.get("overview"),
                                "release_date": movie_details.get("release_date"),
                                "poster_path": movie_details.get("poster_path"),
                                "backdrop_path": movie_details.get("backdrop_path"),
                                "reviews": reviews,
                                "review_count": len(reviews),
                                "has_reviews": True,
                                "popularity": movie_details.get("popularity"),
                                "vote_count": movie_details.get("vote_count")
                            }
                            
                            # Store in database
                            await db.movies.insert_one(movie_doc)
                            total_added += 1
                            
                            print(f"        💾 Stored ({total_added}/{TARGET_MAX}): {english_title}")
                            
                            # Small delay to avoid rate limiting
                            await asyncio.sleep(0.5)
                            
                        except Exception as e:
                            print(f"        ❌ Error: {str(e)}")
                            total_skipped += 1
                            continue
            
            except Exception as e:
                print(f"  ❌ Error fetching {genre_name}: {str(e)}")
                continue
        
        # Check if we have minimum required
        if total_added >= TARGET_MIN:
            print(f"\n✅ Minimum target reached! {total_added} movies collected.")
            break
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"🎉 DATABASE POPULATION COMPLETE!")
    print(f"{'='*70}")
    print(f"✅ Movies added: {total_added}")
    print(f"⏭️  Movies skipped: {total_skipped}")
    print(f"📊 Total in database: {await db.movies.count_documents({})}")
    print(f"{'='*70}\n")
    
    await client.close()
    
    # Close connection
    client.close()

if __name__ == "__main__":
    try:
        asyncio.run(populate_database())
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(0)
