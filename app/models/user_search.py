"""
User Search Model - MongoDB Document Schema
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from bson import ObjectId
from .movie import PyObjectId


class UserSearchModel(BaseModel):
    """User search history document schema for MongoDB"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    
    # Search Query
    search_query: str = Field(..., description="User's search query (movie name)")
    search_language: str = Field(default="mixed", description="Language of search query")
    
    # Matched Movie
    movie_id: Optional[PyObjectId] = Field(None, description="Reference to matched Movie document")
    tmdb_id: Optional[int] = Field(None, description="TMDB movie ID")
    matched_title: Optional[str] = Field(None, description="Matched movie title")
    
    # Search Results
    total_recommendations: int = Field(default=0, description="Number of recommendations generated")
    recommendations_shown: List[int] = Field(default_factory=list, description="List of TMDB IDs shown")
    
    # Performance Metrics
    search_time_ms: Optional[int] = Field(None, description="Search execution time in milliseconds")
    cache_hit: bool = Field(default=False, description="Whether results were from cache")
    
    # Session Info
    session_id: Optional[str] = Field(None, description="User session identifier")
    ip_address: Optional[str] = Field(None, description="User IP address (anonymized)")
    user_agent: Optional[str] = Field(None, description="User agent string")
    
    # Timestamp
    searched_at: datetime = Field(default_factory=datetime.utcnow, description="Search timestamp")
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "search_query": "KGF",
                "tmdb_id": 123456,
                "matched_title": "KGF Chapter 2",
                "total_recommendations": 15,
                "search_time_ms": 850,
                "cache_hit": False
            }
        }
