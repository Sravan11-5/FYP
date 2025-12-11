"""Quick test to check TMDB title formats"""
import asyncio
from app.collectors.tmdb_collector import TMDBDataCollector

async def test():
    collector = TMDBDataCollector()
    movies = await collector.discover_movies_by_genre([28], 7.0, 5)
    
    print("\nTMDB Movie Titles:")
    for m in movies:
        print(f"ID: {m.get('id')}")
        print(f"  title: {m.get('title')}")
        print(f"  original_title: {m.get('original_title')}")
        print()

asyncio.run(test())
