"""
Genre Model - MongoDB Document Schema
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from bson import ObjectId
from .movie import PyObjectId


class GenreModel(BaseModel):
    """Genre document schema for MongoDB"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    
    # Genre Information
    tmdb_genre_id: int = Field(..., description="TMDB genre ID")
    name: str = Field(..., description="Genre name in English")
    telugu_name: Optional[str] = Field(None, description="Genre name in Telugu")
    
    # Statistics
    movie_count: int = Field(default=0, description="Number of movies in this genre")
    avg_rating: Optional[float] = Field(None, description="Average rating of movies in this genre")
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "tmdb_genre_id": 28,
                "name": "Action",
                "telugu_name": "యాక్షన్",
                "movie_count": 150,
                "avg_rating": 7.5
            }
        }


class MovieGenreRelationship(BaseModel):
    """Many-to-many relationship between Movies and Genres"""
    movie_id: PyObjectId = Field(..., description="Reference to Movie document")
    genre_ids: List[int] = Field(..., description="List of TMDB genre IDs")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
