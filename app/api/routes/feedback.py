"""
Feedback API Endpoints
Handles user feedback submission and analytics
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from app.services.feedback_service import get_feedback_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class FeedbackSubmission(BaseModel):
    """Feedback submission request model"""
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    movie_id: int = Field(..., description="TMDB movie ID")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field(None, description="Optional text comment")
    recommendation_id: Optional[str] = Field(None, description="Recommendation ID")


class FeedbackResponse(BaseModel):
    """Feedback submission response model"""
    feedback_id: str
    message: str
    timestamp: str


@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackSubmission) -> FeedbackResponse:
    """
    Submit user feedback on a movie recommendation.
    
    Args:
        feedback: Feedback data
        
    Returns:
        Feedback ID and confirmation
    """
    try:
        service = get_feedback_service()
        
        feedback_id = await service.submit_feedback(
            user_id=feedback.user_id,
            session_id=feedback.session_id,
            movie_id=feedback.movie_id,
            rating=feedback.rating,
            comment=feedback.comment,
            recommendation_id=feedback.recommendation_id
        )
        
        if not feedback_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to submit feedback"
            )
        
        return FeedbackResponse(
            feedback_id=feedback_id,
            message="Feedback submitted successfully",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in submit_feedback endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user_feedback(
    user_id: str,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get feedback submitted by a specific user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of results
        
    Returns:
        List of feedback entries
    """
    try:
        service = get_feedback_service()
        feedback = await service.get_user_feedback(user_id=user_id, limit=limit)
        
        return {
            "user_id": user_id,
            "feedback_count": len(feedback),
            "feedback": feedback
        }
        
    except Exception as e:
        logger.error(f"Error in get_user_feedback endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/movie/{movie_id}")
async def get_movie_feedback(movie_id: int) -> Dict[str, Any]:
    """
    Get aggregated feedback statistics for a movie.
    
    Args:
        movie_id: TMDB movie ID
        
    Returns:
        Movie feedback statistics
    """
    try:
        service = get_feedback_service()
        stats = await service.get_movie_stats(movie_id)
        
        return {
            "movie_id": movie_id,
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in get_movie_feedback endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendation/{recommendation_id}")
async def get_recommendation_feedback(recommendation_id: str) -> Dict[str, Any]:
    """
    Get performance metrics for a specific recommendation.
    
    Args:
        recommendation_id: Recommendation ID
        
    Returns:
        Recommendation performance metrics
    """
    try:
        service = get_feedback_service()
        performance = await service.get_recommendation_performance(recommendation_id)
        
        return {
            "recommendation_id": recommendation_id,
            "performance": performance,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in get_recommendation_feedback endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/overall")
async def get_overall_feedback_stats() -> Dict[str, Any]:
    """
    Get overall feedback statistics across all movies.
    
    Returns:
        Overall feedback statistics
    """
    try:
        service = get_feedback_service()
        stats = await service.get_overall_stats()
        
        return {
            "overall_statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in get_overall_feedback_stats endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/unprocessed")
async def get_unprocessed_feedback(limit: int = 1000) -> Dict[str, Any]:
    """
    Get unprocessed feedback for model retraining (admin endpoint).
    
    Args:
        limit: Maximum number of results
        
    Returns:
        List of unprocessed feedback
    """
    try:
        service = get_feedback_service()
        feedback = await service.get_unprocessed_feedback(limit=limit)
        
        return {
            "count": len(feedback),
            "feedback": feedback,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in get_unprocessed_feedback endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
