"""
Recommendation API Endpoints
Provides movie recommendations based on sentiment analysis and similarity
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import logging

from app.ml.recommendation_engine import get_recommendation_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


# Request/Response Models
class RecommendationRequest(BaseModel):
    """Request model for movie recommendations"""
    movie_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Name of the movie to get recommendations for"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of recommendations to return"
    )
    min_sentiment_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum positive sentiment threshold (0-1)"
    )
    genre_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for genre matching (0-1)"
    )
    sentiment_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for sentiment score (0-1)"
    )
    similarity_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for review similarity (0-1)"
    )
    
    @validator('genre_weight', 'sentiment_weight', 'similarity_weight')
    def weights_sum_to_one(cls, v, values):
        """Validate that weights sum to approximately 1.0"""
        if 'genre_weight' in values and 'sentiment_weight' in values:
            total = v + values.get('genre_weight', 0) + values.get('sentiment_weight', 0)
            if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                raise ValueError(
                    "genre_weight, sentiment_weight, and similarity_weight must sum to 1.0"
                )
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "movie_name": "RRR",
                "max_results": 10,
                "min_sentiment_score": 0.6,
                "genre_weight": 0.3,
                "sentiment_weight": 0.4,
                "similarity_weight": 0.3
            }
        }


class MovieRecommendation(BaseModel):
    """Model for a single movie recommendation"""
    movie_id: str
    tmdb_id: Optional[int]
    title: str
    genres: List[str]
    vote_average: float
    release_date: Optional[str]
    overview: str
    poster_path: Optional[str]
    recommendation_score: float = Field(
        ...,
        description="Overall recommendation score (0-1, higher is better)"
    )
    sentiment_analysis: Dict = Field(
        ...,
        description="Sentiment analysis results for this movie"
    )
    similarity_score: float = Field(
        ...,
        description="Review similarity with input movie (0-1)"
    )
    genre_match_score: float = Field(
        ...,
        description="Genre match score (0-1)"
    )
    rating_similarity: float = Field(
        ...,
        description="Rating similarity with input movie (0-1)"
    )
    reasoning: str = Field(
        ...,
        description="Human-readable explanation for recommendation"
    )


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    input_movie: str
    recommendations: List[MovieRecommendation]
    count: int
    parameters: Dict = Field(
        ...,
        description="Parameters used for generating recommendations"
    )


class RecommendationStats(BaseModel):
    """Statistics about the recommendation system"""
    total_movies_analyzed: int
    average_recommendation_score: float
    sentiment_distribution: Dict[str, int]
    genre_coverage: List[str]


# Endpoints
@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Generate movie recommendations based on input movie.
    
    Uses Siamese Network for:
    - Sentiment analysis of movie reviews
    - Review similarity computation
    - Context-aware ranking
    
    **Algorithm:**
    1. Find input movie in database
    2. Analyze sentiment of input movie's reviews
    3. Find candidate movies (same genres, similar ratings)
    4. Filter by minimum sentiment threshold
    5. Compute review similarity using Siamese Network
    6. Calculate weighted recommendation score
    7. Rank and return top N movies
    
    **Returns:**
    - List of recommended movies with scores and reasoning
    - Sentiment analysis for each recommendation
    - Similarity and genre match scores
    """
    logger.info(f"Recommendation request for movie: {request.movie_name}")
    
    try:
        # Get recommendation engine
        engine = get_recommendation_engine()
        
        # Generate recommendations
        recommendations = await engine.get_recommendations(
            movie_name=request.movie_name,
            min_sentiment_score=request.min_sentiment_score,
            max_results=request.max_results,
            genre_weight=request.genre_weight,
            sentiment_weight=request.sentiment_weight,
            similarity_weight=request.similarity_weight
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for movie: {request.movie_name}. "
                       "Movie might not exist or have insufficient reviews."
            )
        
        # Build response
        response = RecommendationResponse(
            input_movie=request.movie_name,
            recommendations=recommendations,
            count=len(recommendations),
            parameters={
                "min_sentiment_score": request.min_sentiment_score,
                "max_results": request.max_results,
                "genre_weight": request.genre_weight,
                "sentiment_weight": request.sentiment_weight,
                "similarity_weight": request.similarity_weight
            }
        )
        
        logger.info(
            f"Generated {len(recommendations)} recommendations for {request.movie_name}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@router.get("/stats", response_model=Dict)
async def get_recommendation_stats():
    """
    Get statistics about the recommendation system.
    
    **Returns:**
    - System health status
    - Available movies count
    - Model information
    """
    try:
        from app.database import get_database
        from app.ml.inference import get_model_inference
        
        # Get database stats
        db = get_database()
        movies_count = await db.movies.count_documents({})
        reviews_count = await db.reviews.count_documents({})
        
        # Get model info
        model = get_model_inference()
        
        return {
            "status": "operational",
            "statistics": {
                "total_movies": movies_count,
                "total_reviews": reviews_count,
                "model_loaded": True,
                "model_device": model.device,
                "model_vocab_size": model.vocab_size
            },
            "capabilities": {
                "sentiment_analysis": True,
                "similarity_computation": True,
                "context_aware_ranking": True,
                "batch_processing": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get recommendation stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.get("/health")
async def recommendation_health_check():
    """
    Health check endpoint for recommendation system.
    
    **Returns:**
    - System health status
    - Component availability
    """
    try:
        # Check if recommendation engine can be initialized
        engine = get_recommendation_engine()
        
        # Check if model is loaded
        from app.ml.inference import get_model_inference
        model = get_model_inference()
        
        # Check database connection
        from app.database import get_database
        db = get_database()
        
        return {
            "status": "healthy",
            "components": {
                "recommendation_engine": True,
                "ml_model": True,
                "database": True
            },
            "device": model.device,
            "ready": True
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "components": {
                "recommendation_engine": False,
                "ml_model": False,
                "database": False
            },
            "error": str(e),
            "ready": False
        }
