import asyncio
from app.database import database, connect_to_mongo, close_mongo_connection

async def check_my_fault():
    await connect_to_mongo()
    
    # Search for "My Fault" or "Culpa"
    result = await database.db.movies.find_one({
        'title': {'$regex': 'fault|culpa', '$options': 'i'}
    })
    
    if result:
        print(f"✅ Found in database: {result['title']} (ID: {result.get('tmdb_id')})")
    else:
        print("❌ Not found in database")
        
        # Check total movies count
        count = await database.db.movies.count_documents({})
        print(f"📊 Total movies in database: {count}")
    
    await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(check_my_fault())
