"""
MongoDB Database Configuration and Connection
"""
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class Database:
    """MongoDB Database Manager"""
    client: AsyncIOMotorClient = None
    db = None

# Global database instance
database = Database()

async def connect_to_mongo():
    """Connect to MongoDB"""
    try:
        logger.info(f"Connecting to MongoDB: {settings.MONGODB_URL}")
        database.client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            maxPoolSize=10,
            minPoolSize=1,
            serverSelectionTimeoutMS=5000
        )
        
        # Verify connection
        await database.client.admin.command('ping')
        
        database.db = database.client[settings.MONGODB_DB_NAME]
        logger.info(f"Successfully connected to MongoDB database: {settings.MONGODB_DB_NAME}")
        
        # Create indexes
        await create_indexes()
        
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during MongoDB connection: {e}")
        raise

async def close_mongo_connection():
    """Close MongoDB connection"""
    try:
        if database.client:
            database.client.close()
            logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}")

async def create_indexes():
    """Create database indexes for efficient queries"""
    try:
        logger.info("Creating database indexes...")
        
        # Movies collection indexes
        try:
            await database.db.movies.create_index("tmdb_id", unique=True)
        except Exception as e:
            logger.debug(f"Index tmdb_id already exists: {e}")
        
        try:
            await database.db.movies.create_index("title")
        except Exception as e:
            logger.debug(f"Index title already exists: {e}")
        
        try:
            await database.db.movies.create_index("genres")
        except Exception as e:
            logger.debug(f"Index genres already exists: {e}")
        
        try:
            await database.db.movies.create_index("rating")
        except Exception as e:
            logger.debug(f"Index rating already exists: {e}")
        
        try:
            await database.db.movies.create_index([("genres", 1), ("rating", -1)])
        except Exception as e:
            logger.debug(f"Compound index genres/rating already exists: {e}")
        
        # Reviews collection indexes
        try:
            await database.db.reviews.create_index("movie_id")
        except Exception as e:
            logger.debug(f"Index movie_id already exists: {e}")
        
        try:
            await database.db.reviews.create_index("tmdb_id")
        except Exception as e:
            logger.debug(f"Index tmdb_id already exists: {e}")
        
        try:
            await database.db.reviews.create_index("tweet_id", unique=True)
        except Exception as e:
            logger.debug(f"Index tweet_id already exists: {e}")
        
        try:
            await database.db.reviews.create_index([("movie_id", 1), ("sentiment_score", -1)])
        except Exception as e:
            logger.debug(f"Compound index movie_id/sentiment already exists: {e}")
        
        # User searches collection indexes
        try:
            await database.db.user_searches.create_index("search_query")
        except Exception as e:
            logger.debug(f"Index search_query already exists: {e}")
        
        try:
            await database.db.user_searches.create_index("tmdb_id")
        except Exception as e:
            logger.debug(f"Index tmdb_id already exists: {e}")
        
        try:
            await database.db.user_searches.create_index("searched_at")
        except Exception as e:
            logger.debug(f"Index searched_at already exists: {e}")
        
        # Genres collection indexes
        try:
            await database.db.genres.create_index("tmdb_genre_id", unique=True)
        except Exception as e:
            logger.debug(f"Index tmdb_genre_id already exists: {e}")
        
        try:
            await database.db.genres.create_index("name")
        except Exception as e:
            logger.debug(f"Index name already exists: {e}")
        
        logger.info("Database indexes created successfully")
        
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        # Don't raise - indexes might already exist

def get_database():
    """Get database instance"""
    return database.db
