import asyncio
from app.database import get_database
from app.dependencies import get_db

async def check_genres():
    # Get database instance
    db_gen = get_db()
    db = await db_gen.__anext__()
    
    # Get Killer movie
    killer = await db.movies.find_one({'title': 'Killer'})
    if killer:
        print(f"Killer genre: {killer.get('genre', 'No genre')}")
    
    # Get all movies
    print("\nAll movies in database:")
    movies = await db.movies.find().to_list(length=20)
    for movie in movies:
        print(f"  {movie.get('title')}: {movie.get('genre', 'No genre')} ({movie.get('review_count', 0)} reviews)")

if __name__ == "__main__":
    asyncio.run(check_genres())
