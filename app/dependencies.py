"""
FastAPI Dependencies for dependency injection
"""
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.config import settings
from app.services.tmdb_client import TMDBClient
from app.services.twitter_client import TwitterClient
from app.database import database
import logging

logger = logging.getLogger(__name__)

async def get_database() -> AsyncIOMotorDatabase:
    """
    Dependency: Get database instance
    
    Returns:
        AsyncIOMotorDatabase: MongoDB database instance
    """
    return database.db

def get_tmdb_client() -> TMDBClient:
    """
    Dependency: Get TMDB API client
    
    Returns:
        TMDBClient: Initialized TMDB client
    """
    return TMDBClient()

def get_twitter_client() -> TwitterClient:
    """
    Dependency: Get Twitter API client
    
    Returns:
        TwitterClient: Initialized Twitter client
    """
    return TwitterClient()
