"""
Review Model - MongoDB Document Schema
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from bson import ObjectId
from .movie import PyObjectId


class ReviewModel(BaseModel):
    """Review document schema for MongoDB"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    movie_id: PyObjectId = Field(..., description="Reference to Movie document")
    tmdb_id: int = Field(..., description="TMDB movie ID for quick lookup")
    
    # Twitter/Source Data
    tweet_id: Optional[str] = Field(None, description="Twitter tweet ID (unique)")
    source: str = Field(default="twitter", description="Review source platform")
    user_id: Optional[str] = Field(None, description="User ID from source platform")
    username: Optional[str] = Field(None, description="Username from source platform")
    
    # Review Content
    review_text: str = Field(..., description="Original review text in Telugu")
    language: str = Field(default="te", description="Language of the review")
    
    # Sentiment Analysis
    sentiment_label: Optional[str] = Field(None, description="Sentiment classification (positive, negative, neutral)")
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1, description="Sentiment score (-1 to 1)")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Model confidence score")
    
    # Domain Classification
    domains: Optional[list] = Field(default_factory=list, description="Detected domains (acting, story, music, etc.)")
    domain_scores: Optional[dict] = Field(None, description="Score for each domain")
    
    # Metadata
    likes_count: Optional[int] = Field(None, description="Number of likes/favorites")
    retweets_count: Optional[int] = Field(None, description="Number of retweets/shares")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Review creation timestamp")
    fetched_at: datetime = Field(default_factory=datetime.utcnow, description="Data fetch timestamp")
    analyzed_at: Optional[datetime] = Field(None, description="Analysis timestamp")
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "tmdb_id": 123456,
                "tweet_id": "1234567890123456789",
                "source": "twitter",
                "username": "movie_fan_123",
                "review_text": "అద్భుతమైన సినిమా! యాక్షన్ సీన్స్ చాలా బాగున్నాయి",
                "language": "te",
                "sentiment_label": "positive",
                "sentiment_score": 0.92,
                "domains": ["action", "cinematography"]
            }
        }
