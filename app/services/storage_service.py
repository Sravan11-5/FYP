"""
Database Storage Service
Handles storing movies and reviews with duplicate prevention
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from bson import ObjectId
from app.database import get_database
from app.services.cache_service import get_cache_service

logger = logging.getLogger(__name__)


class DatabaseStorageService:
    """
    Service for storing movies and reviews with duplicate prevention
    """
    
    def __init__(self):
        """Initialize storage service"""
        self.db = get_database()
        self.cache = get_cache_service()
        logger.info("Database Storage Service initialized with caching")
    
    async def store_movie(self, movie_data: Dict) -> Optional[str]:
        """
        Store movie in database with duplicate prevention.
        
        Args:
            movie_data: Parsed movie data
            
        Returns:
            Movie ID (string) if stored/updated, None if failed
        """
        try:
            tmdb_id = movie_data.get('tmdb_id')
            
            if not tmdb_id:
                logger.error("Movie data missing tmdb_id")
                return None
            
            # Check if movie already exists (duplicate prevention)
            existing_movie = await self.db.movies.find_one({"tmdb_id": tmdb_id})
            
            if existing_movie:
                # Update existing movie
                movie_id = existing_movie['_id']
                
                await self.db.movies.update_one(
                    {"_id": movie_id},
                    {
                        "$set": {
                            **movie_data,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                
                logger.info(
                    f"Updated existing movie: {movie_data.get('title')} "
                    f"(TMDB ID: {tmdb_id})"
                )
                
                # Invalidate cache for updated movie
                self.cache.invalidate_movie(tmdb_id)
                
                return str(movie_id)
            
            else:
                # Insert new movie
                movie_data['created_at'] = datetime.utcnow()
                movie_data['updated_at'] = datetime.utcnow()
                
                result = await self.db.movies.insert_one(movie_data)
                movie_id = result.inserted_id
                
                logger.info(
                    f"Stored new movie: {movie_data.get('title')} "
                    f"(TMDB ID: {tmdb_id}, MongoDB ID: {movie_id})"
                )
                
                # Cache the new movie
                movie_data['_id'] = movie_id
                self.cache.set_movie(tmdb_id, movie_data, ttl_seconds=3600)
                
                return str(movie_id)
        
        except Exception as e:
            logger.error(f"Error storing movie: {e}", exc_info=True)
            return None
    
    async def store_review(self, review_data: Dict) -> Optional[str]:
        """
        Store review in database with duplicate prevention.
        
        Args:
            review_data: Parsed review data
            
        Returns:
            Review ID (string) if stored, None if duplicate or failed
        """
        try:
            tweet_id = review_data.get('tweet_id')
            
            if not tweet_id:
                logger.error("Review data missing tweet_id")
                return None
            
            # Check if review already exists (duplicate prevention)
            existing_review = await self.db.reviews.find_one({"tweet_id": tweet_id})
            
            if existing_review:
                logger.debug(
                    f"Duplicate review skipped (tweet_id: {tweet_id})"
                )
                return None
            
            # Insert new review
            review_data['created_at'] = datetime.utcnow()
            
            # Convert movie_id string to ObjectId if needed
            if 'movie_id' in review_data and isinstance(review_data['movie_id'], str):
                try:
                    review_data['movie_id'] = ObjectId(review_data['movie_id'])
                except:
                    pass  # Keep as string if conversion fails
            
            result = await self.db.reviews.insert_one(review_data)
            review_id = result.inserted_id
            
            logger.info(
                f"Stored new review (tweet_id: {tweet_id}, "
                f"MongoDB ID: {review_id})"
            )
            
            return str(review_id)
        
        except Exception as e:
            logger.error(f"Error storing review: {e}", exc_info=True)
            return None
    
    async def store_reviews_batch(
        self,
        reviews_data: List[Dict]
    ) -> Dict[str, int]:
        """
        Store multiple reviews efficiently with duplicate prevention.
        
        Args:
            reviews_data: List of parsed review data
            
        Returns:
            Dictionary with counts: {stored, duplicates, failed}
        """
        counts = {"stored": 0, "duplicates": 0, "failed": 0}
        
        for review_data in reviews_data:
            tweet_id = review_data.get('tweet_id')
            
            if not tweet_id:
                counts['failed'] += 1
                continue
            
            try:
                # Check for duplicate
                existing = await self.db.reviews.find_one({"tweet_id": tweet_id})
                
                if existing:
                    counts['duplicates'] += 1
                    continue
                
                # Store new review
                review_data['created_at'] = datetime.utcnow()
                
                # Convert movie_id if needed
                if 'movie_id' in review_data and isinstance(review_data['movie_id'], str):
                    try:
                        review_data['movie_id'] = ObjectId(review_data['movie_id'])
                    except:
                        pass
                
                await self.db.reviews.insert_one(review_data)
                counts['stored'] += 1
            
            except Exception as e:
                logger.error(f"Error storing review {tweet_id}: {e}")
                counts['failed'] += 1
        
        logger.info(
            f"Batch storage complete: {counts['stored']} stored, "
            f"{counts['duplicates']} duplicates, {counts['failed']} failed"
        )
        
        return counts
    
    async def check_movie_exists(self, tmdb_id: int) -> bool:
        """
        Check if movie already exists in database.
        
        Args:
            tmdb_id: TMDB movie ID
            
        Returns:
            True if exists, False otherwise
        """
        try:
            count = await self.db.movies.count_documents({"tmdb_id": tmdb_id})
            return count > 0
        except Exception as e:
            logger.error(f"Error checking movie existence: {e}")
            return False
    
    async def check_review_exists(self, tweet_id: str) -> bool:
        """
        Check if review already exists in database.
        
        Args:
            tweet_id: Twitter tweet ID
            
        Returns:
            True if exists, False otherwise
        """
        try:
            count = await self.db.reviews.count_documents({"tweet_id": tweet_id})
            return count > 0
        except Exception as e:
            logger.error(f"Error checking review existence: {e}")
            return False
    
    async def get_movie_by_tmdb_id(self, tmdb_id: int) -> Optional[Dict]:
        """
        Get movie from database by TMDB ID with caching.
        
        Args:
            tmdb_id: TMDB movie ID
            
        Returns:
            Movie document or None
        """
        try:
            # Check cache first
            cached_movie = self.cache.get_movie(tmdb_id)
            if cached_movie is not None:
                logger.debug(f"Cache HIT for movie {tmdb_id}")
                return cached_movie
            
            logger.debug(f"Cache MISS for movie {tmdb_id}")
            # Query database
            movie = await self.db.movies.find_one({"tmdb_id": tmdb_id})
            
            # Cache the result (even if None to prevent repeated lookups)
            if movie:
                self.cache.set_movie(tmdb_id, movie, ttl_seconds=3600)  # 1 hour
            
            return movie
        except Exception as e:
            logger.error(f"Error getting movie: {e}")
            return None
    
    async def get_reviews_for_movie(self, movie_id: str) -> List[Dict]:
        """
        Get all reviews for a specific movie.
        
        Args:
            movie_id: MongoDB movie ID (string)
            
        Returns:
            List of review documents
        """
        try:
            # Convert to ObjectId
            movie_object_id = ObjectId(movie_id)
            
            reviews = await self.db.reviews.find(
                {"movie_id": movie_object_id}
            ).to_list(length=1000)
            
            return reviews
        except Exception as e:
            logger.error(f"Error getting reviews: {e}")
            return []
    
    async def get_storage_stats(self) -> Dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with counts and stats
        """
        try:
            movies_count = await self.db.movies.count_documents({})
            reviews_count = await self.db.reviews.count_documents({})
            
            # Get movies with reviews
            pipeline = [
                {
                    "$lookup": {
                        "from": "reviews",
                        "localField": "_id",
                        "foreignField": "movie_id",
                        "as": "reviews"
                    }
                },
                {
                    "$project": {
                        "title": 1,
                        "review_count": {"$size": "$reviews"}
                    }
                },
                {
                    "$match": {
                        "review_count": {"$gt": 0}
                    }
                }
            ]
            
            movies_with_reviews = await self.db.movies.aggregate(pipeline).to_list(length=None)
            
            return {
                "total_movies": movies_count,
                "total_reviews": reviews_count,
                "movies_with_reviews": len(movies_with_reviews),
                "avg_reviews_per_movie": (
                    reviews_count / movies_count if movies_count > 0 else 0
                )
            }
        
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}


# Singleton instance
_storage_service = None

def get_storage_service() -> DatabaseStorageService:
    """
    Get singleton instance of storage service.
    
    Returns:
        DatabaseStorageService instance
    """
    global _storage_service
    if _storage_service is None:
        _storage_service = DatabaseStorageService()
    return _storage_service
