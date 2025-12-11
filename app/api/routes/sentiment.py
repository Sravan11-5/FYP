"""
Sentiment Analysis API Endpoints
=================================
FastAPI routes for ML model inference
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import logging

from app.ml.inference import get_model_inference

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/sentiment", tags=["Sentiment Analysis"])


# Request/Response Models
class SentimentRequest(BaseModel):
    """Single review sentiment prediction request."""
    text: str = Field(..., description="Telugu review text", min_length=1, max_length=5000)
    return_confidence: bool = Field(True, description="Include confidence scores")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Review text cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "text": "సూపర్ మూవీ! అద్భుతమైన కథ మరియు యాక్షన్ సీన్స్",
                "return_confidence": True
            }
        }


class BatchSentimentRequest(BaseModel):
    """Batch review sentiment prediction request."""
    texts: List[str] = Field(..., description="List of Telugu review texts", min_items=1, max_items=100)
    batch_size: int = Field(32, description="Batch size for processing", ge=1, le=64)
    return_confidence: bool = Field(True, description="Include confidence scores")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")
        # Filter out empty texts
        valid_texts = [t.strip() for t in v if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")
        return valid_texts
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "సూపర్ మూవీ! అద్భుతమైన కథ",
                    "బోరింగ్ సినిమా. టైం వేస్ట్",
                    "డిసెంట్ వాచ్. కథ ఓకే టైప్"
                ],
                "batch_size": 32,
                "return_confidence": True
            }
        }


class SentimentResponse(BaseModel):
    """Sentiment prediction response."""
    sentiment: str = Field(..., description="Predicted sentiment: positive, negative, or neutral")
    sentiment_code: int = Field(..., description="Sentiment code: 0=negative, 1=neutral, 2=positive")
    confidence: Optional[float] = Field(None, description="Prediction confidence (0-1)")
    probabilities: Optional[dict] = Field(None, description="Class probabilities")
    
    class Config:
        schema_extra = {
            "example": {
                "sentiment": "positive",
                "sentiment_code": 2,
                "confidence": 0.9876,
                "probabilities": {
                    "negative": 0.0012,
                    "neutral": 0.0112,
                    "positive": 0.9876
                }
            }
        }


class BatchSentimentResponse(BaseModel):
    """Batch sentiment prediction response."""
    predictions: List[SentimentResponse] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "sentiment": "positive",
                        "sentiment_code": 2,
                        "confidence": 0.9876,
                        "probabilities": {
                            "negative": 0.0012,
                            "neutral": 0.0112,
                            "positive": 0.9876
                        }
                    }
                ],
                "count": 1
            }
        }


class SimilarityRequest(BaseModel):
    """Review similarity comparison request."""
    text1: str = Field(..., description="First Telugu review text", min_length=1)
    text2: str = Field(..., description="Second Telugu review text", min_length=1)
    
    class Config:
        schema_extra = {
            "example": {
                "text1": "సూపర్ మూవీ! అద్భుతమైన కథ",
                "text2": "గొప్ప సినిమా! కథ చాలా బాగుంది"
            }
        }


class SimilarityResponse(BaseModel):
    """Similarity comparison response."""
    similarity_score: float = Field(..., description="Similarity score (0-1, higher is more similar)")
    text1_sentiment: str = Field(..., description="Sentiment of first review")
    text2_sentiment: str = Field(..., description="Sentiment of second review")
    
    class Config:
        schema_extra = {
            "example": {
                "similarity_score": 0.8765,
                "text1_sentiment": "positive",
                "text2_sentiment": "positive"
            }
        }


class ModelStatusResponse(BaseModel):
    """Model status information."""
    status: str = Field(..., description="Model status")
    device: str = Field(..., description="Device being used (cuda/cpu)")
    vocab_size: int = Field(..., description="Vocabulary size")
    model_loaded: bool = Field(..., description="Whether model is loaded")


# API Endpoints

@router.post("/predict", response_model=SentimentResponse, status_code=status.HTTP_200_OK)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment for a single Telugu movie review.
    
    - **text**: Telugu review text (required)
    - **return_confidence**: Include confidence scores (default: true)
    
    Returns sentiment prediction with optional confidence scores.
    """
    try:
        # Get model instance
        model = get_model_inference()
        
        # Predict
        result = model.predict_sentiment(
            text=request.text,
            return_confidence=request.return_confidence
        )
        
        logger.info(f"Predicted sentiment: {result['sentiment']} (confidence: {result.get('confidence', 'N/A')})")
        
        return SentimentResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict-batch", response_model=BatchSentimentResponse, status_code=status.HTTP_200_OK)
async def predict_sentiment_batch(request: BatchSentimentRequest):
    """
    Predict sentiment for multiple Telugu movie reviews (batch processing).
    
    - **texts**: List of Telugu review texts (1-100 items)
    - **batch_size**: Batch size for processing (default: 32)
    - **return_confidence**: Include confidence scores (default: true)
    
    Returns list of sentiment predictions with optional confidence scores.
    Optimized for processing multiple reviews efficiently.
    """
    try:
        # Get model instance
        model = get_model_inference()
        
        # Predict batch
        results = model.predict_sentiment_batch(
            texts=request.texts,
            batch_size=request.batch_size,
            return_confidence=request.return_confidence
        )
        
        logger.info(f"Batch prediction completed: {len(results)} reviews processed")
        
        # Convert to response models
        predictions = [SentimentResponse(**r) for r in results]
        
        return BatchSentimentResponse(
            predictions=predictions,
            count=len(predictions)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post("/similarity", response_model=SimilarityResponse, status_code=status.HTTP_200_OK)
async def compute_similarity(request: SimilarityRequest):
    """
    Compute similarity between two Telugu movie reviews.
    
    - **text1**: First review text
    - **text2**: Second review text
    
    Returns similarity score (0-1) and sentiments of both reviews.
    Higher scores indicate more similar reviews.
    """
    try:
        # Get model instance
        model = get_model_inference()
        
        # Compute similarity
        similarity_score = model.compute_similarity(request.text1, request.text2)
        
        # Get sentiments
        sent1 = model.predict_sentiment(request.text1, return_confidence=False)
        sent2 = model.predict_sentiment(request.text2, return_confidence=False)
        
        logger.info(f"Similarity computed: {similarity_score}")
        
        return SimilarityResponse(
            similarity_score=similarity_score,
            text1_sentiment=sent1['sentiment'],
            text2_sentiment=sent2['sentiment']
        )
    
    except Exception as e:
        logger.error(f"Similarity computation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity computation failed: {str(e)}"
        )


@router.get("/status", response_model=ModelStatusResponse, status_code=status.HTTP_200_OK)
async def get_model_status():
    """
    Get current model status and configuration.
    
    Returns information about the loaded model and device being used.
    """
    try:
        # Get model instance
        model = get_model_inference()
        
        return ModelStatusResponse(
            status="ready",
            device=model.device,
            vocab_size=model.vocab_size,
            model_loaded=True
        )
    
    except Exception as e:
        logger.error(f"Status check error: {str(e)}", exc_info=True)
        return ModelStatusResponse(
            status="error",
            device="unknown",
            vocab_size=0,
            model_loaded=False
        )
