"""
Movie Model - MongoDB Document Schema
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


class MovieModel(BaseModel):
    """Movie document schema for MongoDB"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    tmdb_id: int = Field(..., description="TMDB movie ID")
    title: str = Field(..., description="Movie title")
    original_title: Optional[str] = Field(None, description="Original title in Telugu")
    genres: List[str] = Field(default_factory=list, description="Movie genres")
    rating: Optional[float] = Field(None, ge=0, le=10, description="TMDB rating")
    vote_count: Optional[int] = Field(None, description="Number of votes")
    release_date: Optional[str] = Field(None, description="Release date")
    overview: Optional[str] = Field(None, description="Movie overview/description")
    poster_path: Optional[str] = Field(None, description="Poster image path")
    backdrop_path: Optional[str] = Field(None, description="Backdrop image path")
    popularity: Optional[float] = Field(None, description="TMDB popularity score")
    runtime: Optional[int] = Field(None, description="Runtime in minutes")
    language: Optional[str] = Field(default="te", description="Primary language")
    
    # Sentiment Analysis Results
    avg_sentiment_score: Optional[float] = Field(None, description="Average sentiment score from reviews")
    sentiment_distribution: Optional[dict] = Field(None, description="Distribution of sentiment (positive, negative, neutral)")
    
    # Domain Classification
    domain_scores: Optional[dict] = Field(None, description="Domain-specific scores (acting, story, music, etc.)")
    
    # Metadata
    total_reviews: int = Field(default=0, description="Total number of reviews analyzed")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "tmdb_id": 123456,
                "title": "KGF Chapter 2",
                "original_title": "కేజీఎఫ్ చాప్టర్ 2",
                "genres": ["Action", "Drama"],
                "rating": 8.5,
                "vote_count": 50000,
                "release_date": "2022-04-14",
                "avg_sentiment_score": 0.85,
                "total_reviews": 1500
            }
        }
