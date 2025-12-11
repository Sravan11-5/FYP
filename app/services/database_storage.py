"""
Database Storage Functions
Comprehensive functions for storing movie metadata and reviews with validation
"""
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
import logging

from app.services.duplicate_prevention import DuplicatePreventionService

logger = logging.getLogger(__name__)


# Pydantic Models for Data Validation
class MovieMetadata(BaseModel):
    """Movie metadata model for validation"""
    tmdb_id: int = Field(..., gt=0, description="TMDB movie ID")
    title: str = Field(..., min_length=1, max_length=500)
    original_title: Optional[str] = Field(None, max_length=500)
    overview: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = Field(None, ge=0, le=10)
    vote_count: Optional[int] = Field(None, ge=0)
    popularity: Optional[float] = Field(None, ge=0)
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    genres: Optional[List[int]] = Field(default_factory=list)
    runtime: Optional[int] = Field(None, ge=0)
    budget: Optional[int] = Field(None, ge=0)
    revenue: Optional[int] = Field(None, ge=0)
    original_language: Optional[str] = None
    
    @validator('title', 'original_title')
    def strip_whitespace(cls, v):
        return v.strip() if v else v
    
    class Config:
        json_schema_extra = {
            "example": {
                "tmdb_id": 325980,
                "title": "Baahubali",
                "vote_average": 7.5,
                "genres": [28, 12]
            }
        }


class MovieReview(BaseModel):
    """Movie review model for validation"""
    tweet_id: str = Field(..., min_length=1, description="Twitter tweet ID")
    tmdb_id: int = Field(..., gt=0, description="TMDB movie ID")
    movie_title: Optional[str] = Field(None, max_length=500)
    text: str = Field(..., min_length=1, max_length=5000)
    author_id: Optional[str] = None
    author_name: Optional[str] = None
    author_username: Optional[str] = None
    author_verified: Optional[bool] = False
    likes: Optional[int] = Field(default=0, ge=0)
    retweets: Optional[int] = Field(default=0, ge=0)
    replies: Optional[int] = Field(default=0, ge=0)
    language: Optional[str] = "te"
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    sentiment_label: Optional[str] = None  # positive, negative, neutral
    created_at: Optional[datetime] = None
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Review text cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "tweet_id": "1234567890",
                "tmdb_id": 325980,
                "text": "Great movie!",
                "likes": 10,
                "sentiment_score": 0.8
            }
        }


class DatabaseStorageService:
    """Service for storing movie and review data in the database"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize storage service
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.duplicate_service = DuplicatePreventionService(db)
    
    async def store_movie(
        self,
        movie_data: Dict[str, Any],
        update_if_exists: bool = True
    ) -> Dict[str, Any]:
        """
        Store movie metadata in the database
        
        Args:
            movie_data: Movie metadata dictionary
            update_if_exists: Whether to update if movie exists
            
        Returns:
            Operation result dictionary
        """
        try:
            # Validate data using Pydantic model
            validated_movie = MovieMetadata(**movie_data)
            
            # Convert to dict for storage
            storage_data = validated_movie.model_dump(exclude_none=True)
            
            # Use duplicate prevention service
            result = await self.duplicate_service.insert_or_update_movie(
                storage_data,
                update_if_exists=update_if_exists
            )
            
            logger.info(f"Movie storage result: {result['operation']} for TMDB ID {validated_movie.tmdb_id}")
            
            return {
                "success": True,
                "operation": result["operation"],
                "tmdb_id": validated_movie.tmdb_id,
                "data": result
            }
            
        except ValueError as e:
            logger.error(f"Validation error storing movie: {e}")
            return {
                "success": False,
                "error": "validation_error",
                "message": str(e)
            }
        except Exception as e:
            logger.error(f"Error storing movie: {e}")
            return {
                "success": False,
                "error": "storage_error",
                "message": str(e)
            }
    
    async def store_review(
        self,
        review_data: Dict[str, Any],
        update_if_exists: bool = False
    ) -> Dict[str, Any]:
        """
        Store movie review in the database
        
        Args:
            review_data: Review data dictionary
            update_if_exists: Whether to update if review exists
            
        Returns:
            Operation result dictionary
        """
        try:
            # Validate data using Pydantic model
            validated_review = MovieReview(**review_data)
            
            # Convert to dict for storage
            storage_data = validated_review.model_dump(exclude_none=True)
            
            # Use duplicate prevention service
            result = await self.duplicate_service.insert_or_update_review(
                storage_data,
                update_if_exists=update_if_exists
            )
            
            logger.info(f"Review storage result: {result['operation']} for tweet ID {validated_review.tweet_id}")
            
            return {
                "success": True,
                "operation": result["operation"],
                "tweet_id": validated_review.tweet_id,
                "data": result
            }
            
        except ValueError as e:
            logger.error(f"Validation error storing review: {e}")
            return {
                "success": False,
                "error": "validation_error",
                "message": str(e)
            }
        except Exception as e:
            logger.error(f"Error storing review: {e}")
            return {
                "success": False,
                "error": "storage_error",
                "message": str(e)
            }
    
    async def store_multiple_movies(
        self,
        movies_data: List[Dict[str, Any]],
        update_if_exists: bool = True
    ) -> Dict[str, Any]:
        """
        Store multiple movies in batch
        
        Args:
            movies_data: List of movie data dictionaries
            update_if_exists: Whether to update existing movies
            
        Returns:
            Batch operation results
        """
        results = {
            "success": True,
            "total": len(movies_data),
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "validation_errors": 0,
            "details": []
        }
        
        for movie_data in movies_data:
            try:
                # Validate
                validated_movie = MovieMetadata(**movie_data)
                storage_data = validated_movie.model_dump(exclude_none=True)
                
                # Store
                result = await self.duplicate_service.insert_or_update_movie(
                    storage_data,
                    update_if_exists=update_if_exists
                )
                
                # Update counters
                if result["operation"] == "inserted":
                    results["inserted"] += 1
                elif result["operation"] == "updated":
                    results["updated"] += 1
                elif result["operation"] == "skipped":
                    results["skipped"] += 1
                
                results["details"].append({
                    "tmdb_id": validated_movie.tmdb_id,
                    "operation": result["operation"]
                })
                
            except ValueError as e:
                results["validation_errors"] += 1
                results["details"].append({
                    "error": "validation_error",
                    "message": str(e),
                    "data": movie_data
                })
                logger.error(f"Validation error in batch movie storage: {e}")
                
            except Exception as e:
                results["errors"] += 1
                results["details"].append({
                    "error": "storage_error",
                    "message": str(e),
                    "data": movie_data
                })
                logger.error(f"Error in batch movie storage: {e}")
        
        # Set overall success status
        results["success"] = results["errors"] == 0 and results["validation_errors"] == 0
        
        logger.info(
            f"Batch movie storage: {results['inserted']} inserted, "
            f"{results['updated']} updated, {results['skipped']} skipped, "
            f"{results['validation_errors']} validation errors, {results['errors']} storage errors"
        )
        
        return results
    
    async def store_multiple_reviews(
        self,
        reviews_data: List[Dict[str, Any]],
        update_if_exists: bool = False
    ) -> Dict[str, Any]:
        """
        Store multiple reviews in batch
        
        Args:
            reviews_data: List of review data dictionaries
            update_if_exists: Whether to update existing reviews
            
        Returns:
            Batch operation results
        """
        results = {
            "success": True,
            "total": len(reviews_data),
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "validation_errors": 0,
            "details": []
        }
        
        for review_data in reviews_data:
            try:
                # Validate
                validated_review = MovieReview(**review_data)
                storage_data = validated_review.model_dump(exclude_none=True)
                
                # Store
                result = await self.duplicate_service.insert_or_update_review(
                    storage_data,
                    update_if_exists=update_if_exists
                )
                
                # Update counters
                if result["operation"] == "inserted":
                    results["inserted"] += 1
                elif result["operation"] == "updated":
                    results["updated"] += 1
                elif result["operation"] == "skipped":
                    results["skipped"] += 1
                
                results["details"].append({
                    "tweet_id": validated_review.tweet_id,
                    "operation": result["operation"]
                })
                
            except ValueError as e:
                results["validation_errors"] += 1
                results["details"].append({
                    "error": "validation_error",
                    "message": str(e),
                    "data": review_data
                })
                logger.error(f"Validation error in batch review storage: {e}")
                
            except Exception as e:
                results["errors"] += 1
                results["details"].append({
                    "error": "storage_error",
                    "message": str(e),
                    "data": review_data
                })
                logger.error(f"Error in batch review storage: {e}")
        
        # Set overall success status
        results["success"] = results["errors"] == 0 and results["validation_errors"] == 0
        
        logger.info(
            f"Batch review storage: {results['inserted']} inserted, "
            f"{results['updated']} updated, {results['skipped']} skipped, "
            f"{results['validation_errors']} validation errors, {results['errors']} storage errors"
        )
        
        return results
    
    async def store_movie_with_reviews(
        self,
        movie_data: Dict[str, Any],
        reviews_data: List[Dict[str, Any]],
        update_movie_if_exists: bool = True,
        update_reviews_if_exist: bool = False
    ) -> Dict[str, Any]:
        """
        Store movie and its reviews together (transaction-like operation)
        
        Args:
            movie_data: Movie metadata
            reviews_data: List of review data
            update_movie_if_exists: Whether to update movie if exists
            update_reviews_if_exist: Whether to update reviews if exist
            
        Returns:
            Combined operation results
        """
        results = {
            "success": True,
            "movie": None,
            "reviews": None
        }
        
        try:
            # Store movie first
            movie_result = await self.store_movie(movie_data, update_movie_if_exists)
            results["movie"] = movie_result
            
            if not movie_result["success"]:
                logger.error("Failed to store movie, skipping reviews")
                results["success"] = False
                return results
            
            # Store reviews
            reviews_result = await self.store_multiple_reviews(reviews_data, update_reviews_if_exist)
            results["reviews"] = reviews_result
            
            if not reviews_result["success"]:
                logger.warning("Some reviews failed to store")
                results["success"] = False
            
            logger.info(
                f"Stored movie {movie_result.get('tmdb_id')} with "
                f"{reviews_result.get('inserted', 0)} new reviews"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in store_movie_with_reviews: {e}")
            results["success"] = False
            results["error"] = str(e)
            return results
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Database statistics
        """
        try:
            movies_count = await self.db.movies.count_documents({})
            reviews_count = await self.db.reviews.count_documents({})
            
            # Get genre statistics
            genre_pipeline = [
                {"$unwind": "$genres"},
                {"$group": {"_id": "$genres", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5}
            ]
            top_genres = await self.db.movies.aggregate(genre_pipeline).to_list(5)
            
            # Get rating statistics
            rating_stats = await self.db.movies.aggregate([
                {
                    "$group": {
                        "_id": None,
                        "avg_rating": {"$avg": "$vote_average"},
                        "max_rating": {"$max": "$vote_average"},
                        "min_rating": {"$min": "$vote_average"}
                    }
                }
            ]).to_list(1)
            
            stats = {
                "total_movies": movies_count,
                "total_reviews": reviews_count,
                "top_genres": top_genres,
                "rating_stats": rating_stats[0] if rating_stats else None
            }
            
            logger.info(f"Storage statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage statistics: {e}")
            return {"error": str(e)}


# Helper function to get service instance
def get_storage_service(db: AsyncIOMotorDatabase) -> DatabaseStorageService:
    """
    Get database storage service instance
    
    Args:
        db: MongoDB database instance
        
    Returns:
        DatabaseStorageService instance
    """
    return DatabaseStorageService(db)
