"""
User Feedback Service
Collects and analyzes user feedback on movie recommendations
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from bson import ObjectId
from app.database import get_database

logger = logging.getLogger(__name__)


class FeedbackService:
    """Service for collecting and analyzing user feedback on recommendations"""
    
    def __init__(self):
        """Initialize feedback service"""
        self.db = get_database()
        logger.info("Feedback Service initialized")
    
    async def submit_feedback(
        self,
        user_id: str,
        session_id: str,
        movie_id: int,
        rating: int,
        comment: Optional[str] = None,
        recommendation_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Submit user feedback on a movie recommendation.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            movie_id: TMDB movie ID
            rating: User rating (1-5 scale)
            comment: Optional text comment
            recommendation_id: ID of the recommendation that led to this feedback
            
        Returns:
            Feedback ID if successful, None otherwise
        """
        try:
            feedback_data = {
                'user_id': user_id,
                'session_id': session_id,
                'movie_id': movie_id,
                'rating': rating,
                'comment': comment,
                'recommendation_id': recommendation_id,
                'created_at': datetime.utcnow(),
                'processed': False
            }
            
            result = await self.db.user_feedback.insert_one(feedback_data)
            feedback_id = str(result.inserted_id)
            
            logger.info(
                f"Feedback submitted: user={user_id}, movie={movie_id}, "
                f"rating={rating}, feedback_id={feedback_id}"
            )
            
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}", exc_info=True)
            return None
    
    async def get_user_feedback(
        self,
        user_id: Optional[str] = None,
        movie_id: Optional[int] = None,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get user feedback based on filters.
        
        Args:
            user_id: Filter by user ID
            movie_id: Filter by movie ID
            session_id: Filter by session ID
            limit: Maximum number of results
            
        Returns:
            List of feedback documents
        """
        try:
            query = {}
            
            if user_id:
                query['user_id'] = user_id
            if movie_id:
                query['movie_id'] = movie_id
            if session_id:
                query['session_id'] = session_id
            
            feedback = await self.db.user_feedback.find(query).limit(limit).to_list(length=limit)
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error getting feedback: {e}", exc_info=True)
            return []
    
    async def get_movie_stats(self, movie_id: int) -> Dict:
        """
        Get aggregated feedback statistics for a movie.
        
        Args:
            movie_id: TMDB movie ID
            
        Returns:
            Statistics including average rating, count, distribution
        """
        try:
            pipeline = [
                {'$match': {'movie_id': movie_id}},
                {
                    '$group': {
                        '_id': '$movie_id',
                        'avg_rating': {'$avg': '$rating'},
                        'total_feedback': {'$sum': 1},
                        'ratings_distribution': {
                            '$push': '$rating'
                        }
                    }
                }
            ]
            
            result = await self.db.user_feedback.aggregate(pipeline).to_list(length=1)
            
            if not result:
                return {
                    'movie_id': movie_id,
                    'avg_rating': 0.0,
                    'total_feedback': 0,
                    'ratings_distribution': {}
                }
            
            stats = result[0]
            
            # Calculate rating distribution
            ratings = stats.get('ratings_distribution', [])
            distribution = {i: ratings.count(i) for i in range(1, 6)}
            
            return {
                'movie_id': movie_id,
                'avg_rating': round(stats.get('avg_rating', 0.0), 2),
                'total_feedback': stats.get('total_feedback', 0),
                'ratings_distribution': distribution
            }
            
        except Exception as e:
            logger.error(f"Error getting movie stats: {e}", exc_info=True)
            return {}
    
    async def get_recommendation_performance(
        self,
        recommendation_id: str
    ) -> Dict:
        """
        Get performance metrics for a specific recommendation.
        
        Args:
            recommendation_id: Recommendation ID
            
        Returns:
            Performance metrics
        """
        try:
            feedback = await self.db.user_feedback.find({
                'recommendation_id': recommendation_id
            }).to_list(length=None)
            
            if not feedback:
                return {
                    'recommendation_id': recommendation_id,
                    'feedback_count': 0,
                    'avg_rating': 0.0,
                    'positive_rate': 0.0
                }
            
            ratings = [f['rating'] for f in feedback]
            positive_count = sum(1 for r in ratings if r >= 4)
            
            return {
                'recommendation_id': recommendation_id,
                'feedback_count': len(feedback),
                'avg_rating': round(sum(ratings) / len(ratings), 2),
                'positive_rate': round(positive_count / len(feedback) * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendation performance: {e}", exc_info=True)
            return {}
    
    async def get_overall_stats(self) -> Dict:
        """
        Get overall feedback statistics across all movies.
        
        Returns:
            Overall statistics
        """
        try:
            pipeline = [
                {
                    '$group': {
                        '_id': None,
                        'total_feedback': {'$sum': 1},
                        'avg_rating': {'$avg': '$rating'},
                        'unique_users': {'$addToSet': '$user_id'},
                        'unique_movies': {'$addToSet': '$movie_id'}
                    }
                }
            ]
            
            result = await self.db.user_feedback.aggregate(pipeline).to_list(length=1)
            
            if not result:
                return {
                    'total_feedback': 0,
                    'avg_rating': 0.0,
                    'unique_users': 0,
                    'unique_movies': 0
                }
            
            stats = result[0]
            
            return {
                'total_feedback': stats.get('total_feedback', 0),
                'avg_rating': round(stats.get('avg_rating', 0.0), 2),
                'unique_users': len(stats.get('unique_users', [])),
                'unique_movies': len(stats.get('unique_movies', []))
            }
            
        except Exception as e:
            logger.error(f"Error getting overall stats: {e}", exc_info=True)
            return {}
    
    async def mark_feedback_processed(self, feedback_id: str) -> bool:
        """
        Mark feedback as processed for model retraining.
        
        Args:
            feedback_id: Feedback document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = await self.db.user_feedback.update_one(
                {'_id': ObjectId(feedback_id)},
                {'$set': {'processed': True, 'processed_at': datetime.utcnow()}}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error marking feedback as processed: {e}", exc_info=True)
            return False
    
    async def get_unprocessed_feedback(self, limit: int = 1000) -> List[Dict]:
        """
        Get unprocessed feedback for model retraining.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of unprocessed feedback documents
        """
        try:
            feedback = await self.db.user_feedback.find({
                'processed': False
            }).limit(limit).to_list(length=limit)
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error getting unprocessed feedback: {e}", exc_info=True)
            return []


def get_feedback_service() -> FeedbackService:
    """Get feedback service instance"""
    return FeedbackService()
