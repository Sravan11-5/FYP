"""
Duplicate Prevention Service
Handles checking and preventing duplicate movie and review entries
"""
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DuplicatePreventionService:
    """Service for preventing duplicate entries in the database"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize duplicate prevention service
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
    
    async def movie_exists(self, tmdb_id: int) -> bool:
        """
        Check if a movie already exists in the database
        
        Args:
            tmdb_id: TMDB movie ID
            
        Returns:
            True if movie exists, False otherwise
        """
        try:
            count = await self.db.movies.count_documents({"tmdb_id": tmdb_id}, limit=1)
            exists = count > 0
            
            if exists:
                logger.debug(f"Movie with TMDB ID {tmdb_id} already exists")
            
            return exists
            
        except Exception as e:
            logger.error(f"Error checking movie existence for TMDB ID {tmdb_id}: {e}")
            raise
    
    async def review_exists(self, tweet_id: str) -> bool:
        """
        Check if a review already exists in the database
        
        Args:
            tweet_id: Twitter tweet ID
            
        Returns:
            True if review exists, False otherwise
        """
        try:
            count = await self.db.reviews.count_documents({"tweet_id": tweet_id}, limit=1)
            exists = count > 0
            
            if exists:
                logger.debug(f"Review with tweet ID {tweet_id} already exists")
            
            return exists
            
        except Exception as e:
            logger.error(f"Error checking review existence for tweet ID {tweet_id}: {e}")
            raise
    
    async def get_existing_movie(self, tmdb_id: int) -> Optional[Dict[str, Any]]:
        """
        Get existing movie data from database
        
        Args:
            tmdb_id: TMDB movie ID
            
        Returns:
            Movie document or None if not found
        """
        try:
            movie = await self.db.movies.find_one({"tmdb_id": tmdb_id})
            return movie
            
        except Exception as e:
            logger.error(f"Error fetching movie with TMDB ID {tmdb_id}: {e}")
            raise
    
    async def get_existing_review(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """
        Get existing review data from database
        
        Args:
            tweet_id: Twitter tweet ID
            
        Returns:
            Review document or None if not found
        """
        try:
            review = await self.db.reviews.find_one({"tweet_id": tweet_id})
            return review
            
        except Exception as e:
            logger.error(f"Error fetching review with tweet ID {tweet_id}: {e}")
            raise
    
    async def insert_or_update_movie(
        self,
        movie_data: Dict[str, Any],
        update_if_exists: bool = True
    ) -> Dict[str, Any]:
        """
        Insert a new movie or update existing one
        
        Args:
            movie_data: Movie data to insert/update
            update_if_exists: Whether to update if movie exists
            
        Returns:
            Dictionary with operation result
        """
        try:
            tmdb_id = movie_data.get("tmdb_id")
            
            if not tmdb_id:
                raise ValueError("movie_data must contain 'tmdb_id'")
            
            # Check if movie exists
            exists = await self.movie_exists(tmdb_id)
            
            if exists:
                if update_if_exists:
                    # Update existing movie
                    movie_data["updated_at"] = datetime.utcnow()
                    
                    result = await self.db.movies.update_one(
                        {"tmdb_id": tmdb_id},
                        {"$set": movie_data}
                    )
                    
                    logger.info(f"Updated movie with TMDB ID {tmdb_id}")
                    
                    return {
                        "operation": "updated",
                        "tmdb_id": tmdb_id,
                        "matched_count": result.matched_count,
                        "modified_count": result.modified_count
                    }
                else:
                    logger.info(f"Movie with TMDB ID {tmdb_id} already exists, skipping")
                    
                    return {
                        "operation": "skipped",
                        "tmdb_id": tmdb_id,
                        "reason": "already_exists"
                    }
            else:
                # Insert new movie
                movie_data["created_at"] = datetime.utcnow()
                movie_data["updated_at"] = datetime.utcnow()
                
                result = await self.db.movies.insert_one(movie_data)
                
                logger.info(f"Inserted new movie with TMDB ID {tmdb_id}")
                
                return {
                    "operation": "inserted",
                    "tmdb_id": tmdb_id,
                    "inserted_id": str(result.inserted_id)
                }
                
        except Exception as e:
            logger.error(f"Error in insert_or_update_movie: {e}")
            raise
    
    async def insert_or_update_review(
        self,
        review_data: Dict[str, Any],
        update_if_exists: bool = False
    ) -> Dict[str, Any]:
        """
        Insert a new review or update existing one
        
        Args:
            review_data: Review data to insert/update
            update_if_exists: Whether to update if review exists
            
        Returns:
            Dictionary with operation result
        """
        try:
            tweet_id = review_data.get("tweet_id")
            
            if not tweet_id:
                raise ValueError("review_data must contain 'tweet_id'")
            
            # Check if review exists
            exists = await self.review_exists(tweet_id)
            
            if exists:
                if update_if_exists:
                    # Update existing review (e.g., update sentiment score)
                    review_data["updated_at"] = datetime.utcnow()
                    
                    result = await self.db.reviews.update_one(
                        {"tweet_id": tweet_id},
                        {"$set": review_data}
                    )
                    
                    logger.info(f"Updated review with tweet ID {tweet_id}")
                    
                    return {
                        "operation": "updated",
                        "tweet_id": tweet_id,
                        "matched_count": result.matched_count,
                        "modified_count": result.modified_count
                    }
                else:
                    logger.info(f"Review with tweet ID {tweet_id} already exists, skipping")
                    
                    return {
                        "operation": "skipped",
                        "tweet_id": tweet_id,
                        "reason": "already_exists"
                    }
            else:
                # Insert new review
                review_data["created_at"] = datetime.utcnow()
                
                result = await self.db.reviews.insert_one(review_data)
                
                logger.info(f"Inserted new review with tweet ID {tweet_id}")
                
                return {
                    "operation": "inserted",
                    "tweet_id": tweet_id,
                    "inserted_id": str(result.inserted_id)
                }
                
        except Exception as e:
            logger.error(f"Error in insert_or_update_review: {e}")
            raise
    
    async def batch_insert_or_update_movies(
        self,
        movies_data: list[Dict[str, Any]],
        update_if_exists: bool = True
    ) -> Dict[str, Any]:
        """
        Batch insert or update multiple movies
        
        Args:
            movies_data: List of movie data dictionaries
            update_if_exists: Whether to update existing movies
            
        Returns:
            Dictionary with batch operation results
        """
        results = {
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "details": []
        }
        
        for movie_data in movies_data:
            try:
                result = await self.insert_or_update_movie(movie_data, update_if_exists)
                
                if result["operation"] == "inserted":
                    results["inserted"] += 1
                elif result["operation"] == "updated":
                    results["updated"] += 1
                elif result["operation"] == "skipped":
                    results["skipped"] += 1
                
                results["details"].append(result)
                
            except Exception as e:
                results["errors"] += 1
                logger.error(f"Error processing movie: {e}")
                results["details"].append({
                    "operation": "error",
                    "error": str(e)
                })
        
        logger.info(
            f"Batch movie operation complete: "
            f"{results['inserted']} inserted, "
            f"{results['updated']} updated, "
            f"{results['skipped']} skipped, "
            f"{results['errors']} errors"
        )
        
        return results
    
    async def batch_insert_or_update_reviews(
        self,
        reviews_data: list[Dict[str, Any]],
        update_if_exists: bool = False
    ) -> Dict[str, Any]:
        """
        Batch insert or update multiple reviews
        
        Args:
            reviews_data: List of review data dictionaries
            update_if_exists: Whether to update existing reviews
            
        Returns:
            Dictionary with batch operation results
        """
        results = {
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "details": []
        }
        
        for review_data in reviews_data:
            try:
                result = await self.insert_or_update_review(review_data, update_if_exists)
                
                if result["operation"] == "inserted":
                    results["inserted"] += 1
                elif result["operation"] == "updated":
                    results["updated"] += 1
                elif result["operation"] == "skipped":
                    results["skipped"] += 1
                
                results["details"].append(result)
                
            except Exception as e:
                results["errors"] += 1
                logger.error(f"Error processing review: {e}")
                results["details"].append({
                    "operation": "error",
                    "error": str(e)
                })
        
        logger.info(
            f"Batch review operation complete: "
            f"{results['inserted']} inserted, "
            f"{results['updated']} updated, "
            f"{results['skipped']} skipped, "
            f"{results['errors']} errors"
        )
        
        return results
    
    async def verify_indexes(self) -> Dict[str, Any]:
        """
        Verify that required indexes exist
        
        Returns:
            Dictionary with index verification results
        """
        try:
            # Check movies collection indexes
            movies_indexes = await self.db.movies.index_information()
            
            # Check reviews collection indexes
            reviews_indexes = await self.db.reviews.index_information()
            
            # Verify required indexes (check index names)
            has_movie_tmdb_id = 'tmdb_id_1' in movies_indexes
            has_review_tweet_id = 'tweet_id_1' in reviews_indexes
            
            result = {
                "movies_tmdb_id_indexed": has_movie_tmdb_id,
                "reviews_tweet_id_indexed": has_review_tweet_id,
                "movies_indexes": list(movies_indexes.keys()),
                "reviews_indexes": list(reviews_indexes.keys())
            }
            
            logger.info(f"Index verification: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error verifying indexes: {e}")
            raise


# Helper function to get service instance
def get_duplicate_prevention_service(db: AsyncIOMotorDatabase) -> DuplicatePreventionService:
    """
    Get duplicate prevention service instance
    
    Args:
        db: MongoDB database instance
        
    Returns:
        DuplicatePreventionService instance
    """
    return DuplicatePreventionService(db)
