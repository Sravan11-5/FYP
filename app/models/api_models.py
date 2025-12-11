"""
API Request and Response Models
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

# Search Request Models
class MovieSearchRequest(BaseModel):
    """Request model for movie search"""
    movie_name: str = Field(
        ..., 
        min_length=1, 
        max_length=200,
        description="Name of the Telugu movie to search for"
    )
    language: Optional[str] = Field(
        default="te",
        description="Language code (default: te for Telugu)"
    )
    
    @validator('movie_name')
    def validate_movie_name(cls, v):
        """Validate movie name"""
        v = v.strip() if v else ""
        if not v:
            raise ValueError("Movie name cannot be empty")
        return v

# Search Response Models
class MovieBasicInfo(BaseModel):
    """Basic movie information"""
    tmdb_id: int
    title: str
    original_title: Optional[str] = None
    release_date: Optional[str] = None
    rating: Optional[float] = None
    poster_url: Optional[str] = None
    overview: Optional[str] = None

class MovieSearchResponse(BaseModel):
    """Response model for movie search"""
    success: bool
    message: str
    movies: List[MovieBasicInfo] = []
    total_results: int = 0

# Health Check Models
class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    version: str
    services: dict = {
        "database": "unknown",
        "tmdb_api": "unknown",
        "twitter_api": "unknown"
    }

# Error Response Models
class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None
