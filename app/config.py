"""
Configuration Management
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application Settings"""
    
    # API Info
    APP_NAME: str = "Telugu Movie Recommendation System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # MongoDB Database
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "telugu_movie_recommender"
    
    # External APIs
    TMDB_API_KEY: str = ""
    TWITTER_API_KEY: str = ""
    TWITTER_API_SECRET: str = ""
    TWITTER_BEARER_TOKEN: str = ""
    
    # Google API
    GOOGLE_API_KEY: str = ""
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
