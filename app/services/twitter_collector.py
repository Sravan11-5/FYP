"""
Twitter Data Collector
Enhanced collector with batch processing, error handling, and rate limit management
"""
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from app.services.twitter_client import TwitterClient
import logging

logger = logging.getLogger(__name__)


class TwitterDataCollector:
    """Enhanced Twitter data collector with batch processing and error handling"""
    
    def __init__(
        self,
        bearer_token: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        """
        Initialize Twitter Data Collector
        
        Args:
            bearer_token: Twitter Bearer Token
            api_key: Twitter API Key
            api_secret: Twitter API Secret
        """
        self.client = TwitterClient(bearer_token, api_key, api_secret)
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # Conservative: 1 request per second
        
    def _handle_rate_limit(self):
        """Implement rate limiting to prevent API throttling"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def search_movie_reviews_with_retry(
        self,
        movie_name: str,
        max_results: int = 100,
        days_back: int = 30,
        language: str = "te",
        max_retries: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search for movie reviews with retry logic
        
        Args:
            movie_name: Name of the movie
            max_results: Maximum tweets to fetch (10-100)
            days_back: Days to search back
            language: Language code
            max_retries: Maximum retry attempts
            
        Returns:
            List of tweet data or None
        """
        for attempt in range(max_retries):
            try:
                self._handle_rate_limit()
                
                result = self.client.search_movie_reviews(
                    movie_name=movie_name,
                    max_results=max_results,
                    days_back=days_back,
                    language=language
                )
                
                if result is not None:  # Can be empty list
                    logger.info(f"Successfully searched reviews for: {movie_name}, found {len(result)} tweets")
                    return result
                
            except Exception as e:
                logger.error(f"Error searching reviews for '{movie_name}': {e}, attempt {attempt + 1}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached for: {movie_name}")
                    return None
        
        return None
    
    def search_telugu_tweets_with_retry(
        self,
        keywords: List[str],
        max_results: int = 100,
        days_back: int = 7,
        max_retries: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search Telugu tweets by keywords with retry logic
        
        Args:
            keywords: List of keywords to search
            max_results: Maximum tweets to fetch
            days_back: Days to search back
            max_retries: Maximum retry attempts
            
        Returns:
            List of tweets or None
        """
        for attempt in range(max_retries):
            try:
                self._handle_rate_limit()
                
                result = self.client.search_telugu_tweets(
                    keywords=keywords,
                    max_results=max_results,
                    days_back=days_back
                )
                
                if result is not None:
                    logger.info(f"Found {len(result)} Telugu tweets for keywords: {keywords}")
                    return result
                
            except Exception as e:
                logger.error(f"Error searching Telugu tweets: {e}, attempt {attempt + 1}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def batch_collect_movie_reviews(
        self,
        movie_names: List[str],
        max_results_per_movie: int = 50,
        days_back: int = 30,
        language: str = "te"
    ) -> Dict[str, Optional[List[Dict[str, Any]]]]:
        """
        Batch collect reviews for multiple movies
        
        Args:
            movie_names: List of movie names
            max_results_per_movie: Max tweets per movie
            days_back: Days to search back
            language: Language code
            
        Returns:
            Dictionary mapping movie names to their reviews
        """
        results = {}
        total = len(movie_names)
        
        logger.info(f"Batch collecting reviews for {total} movies")
        
        for idx, movie_name in enumerate(movie_names, 1):
            logger.info(f"Collecting reviews {idx}/{total}: {movie_name}")
            
            reviews = self.search_movie_reviews_with_retry(
                movie_name=movie_name,
                max_results=max_results_per_movie,
                days_back=days_back,
                language=language
            )
            
            results[movie_name] = reviews
            
            # Log progress
            if reviews:
                logger.info(f"  → Collected {len(reviews)} reviews for: {movie_name}")
            else:
                logger.warning(f"  → No reviews found for: {movie_name}")
        
        successful = sum(1 for r in results.values() if r is not None and len(r) > 0)
        logger.info(f"Batch collection complete: {successful}/{total} movies have reviews")
        
        return results
    
    def collect_movie_reviews_with_metadata(
        self,
        movie_name: str,
        tmdb_id: Optional[int] = None,
        max_results: int = 100,
        days_back: int = 30,
        min_likes: int = 0,
        language: str = "te"
    ) -> List[Dict[str, Any]]:
        """
        Collect movie reviews with additional metadata and filtering
        
        Args:
            movie_name: Name of the movie
            tmdb_id: TMDB movie ID (optional, will be added to each review)
            max_results: Maximum tweets to fetch
            days_back: Days to search back
            min_likes: Minimum likes filter
            language: Language code
            
        Returns:
            List of review dictionaries with metadata
        """
        try:
            # Fetch reviews
            reviews = self.search_movie_reviews_with_retry(
                movie_name=movie_name,
                max_results=max_results,
                days_back=days_back,
                language=language
            )
            
            if not reviews:
                logger.warning(f"No reviews collected for: {movie_name}")
                return []
            
            # Filter by likes if specified
            if min_likes > 0:
                filtered_reviews = [r for r in reviews if r.get('likes', 0) >= min_likes]
                logger.info(f"Filtered {len(filtered_reviews)}/{len(reviews)} reviews with min {min_likes} likes")
                reviews = filtered_reviews
            
            # Add tmdb_id to each review if provided
            if tmdb_id is not None:
                for review in reviews:
                    review['tmdb_id'] = tmdb_id
            
            logger.info(f"Collected {len(reviews)} reviews for {movie_name}")
            return reviews
            
        except Exception as e:
            logger.error(f"Error collecting reviews with metadata for '{movie_name}': {e}")
            return []
    
    def get_trending_movie_discussions(
        self,
        movie_names: List[str],
        days_back: int = 7,
        min_engagement: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get trending discussions for multiple movies
        
        Args:
            movie_names: List of movie names to check
            days_back: Days to look back
            min_engagement: Minimum likes + retweets for trending
            
        Returns:
            List of trending movie discussions sorted by engagement
        """
        try:
            all_discussions = []
            
            for movie_name in movie_names:
                logger.info(f"Checking trending discussions for: {movie_name}")
                
                reviews = self.search_movie_reviews_with_retry(
                    movie_name=movie_name,
                    max_results=100,
                    days_back=days_back
                )
                
                if not reviews:
                    continue
                
                # Calculate engagement score
                for review in reviews:
                    engagement = review.get('likes', 0) + review.get('retweets', 0)
                    
                    if engagement >= min_engagement:
                        all_discussions.append({
                            'movie_name': movie_name,
                            'tweet_id': review.get('tweet_id'),
                            'text': review.get('text'),
                            'author': review.get('author_username'),
                            'likes': review.get('likes', 0),
                            'retweets': review.get('retweets', 0),
                            'engagement_score': engagement,
                            'created_at': review.get('created_at')
                        })
            
            # Sort by engagement
            trending = sorted(
                all_discussions,
                key=lambda x: x['engagement_score'],
                reverse=True
            )
            
            logger.info(f"Found {len(trending)} trending discussions across {len(movie_names)} movies")
            return trending
            
        except Exception as e:
            logger.error(f"Error getting trending discussions: {e}")
            return []
    
    def analyze_review_sentiment_distribution(
        self,
        movie_name: str,
        max_results: int = 100,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze sentiment distribution for movie reviews (basic keyword analysis)
        
        Args:
            movie_name: Name of the movie
            max_results: Maximum tweets to analyze
            days_back: Days to look back
            
        Returns:
            Dictionary with sentiment distribution
        """
        try:
            reviews = self.search_movie_reviews_with_retry(
                movie_name=movie_name,
                max_results=max_results,
                days_back=days_back
            )
            
            if not reviews:
                return {
                    'movie_name': movie_name,
                    'total_reviews': 0,
                    'sentiment_distribution': {}
                }
            
            # Simple keyword-based sentiment (will be replaced with ML model later)
            positive_keywords = ['బాగుంది', 'చాలా బాగుంది', 'excellent', 'superb', 'awesome', 'great', 'best', '👍', '❤️']
            negative_keywords = ['చెత్త', 'worst', 'boring', 'bad', 'waste', 'flop', '👎']
            
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for review in reviews:
                text = review.get('text', '').lower()
                
                has_positive = any(keyword in text for keyword in positive_keywords)
                has_negative = any(keyword in text for keyword in negative_keywords)
                
                if has_positive and not has_negative:
                    positive_count += 1
                elif has_negative and not has_positive:
                    negative_count += 1
                else:
                    neutral_count += 1
            
            total = len(reviews)
            
            return {
                'movie_name': movie_name,
                'total_reviews': total,
                'sentiment_distribution': {
                    'positive': {
                        'count': positive_count,
                        'percentage': round((positive_count / total) * 100, 2) if total > 0 else 0
                    },
                    'negative': {
                        'count': negative_count,
                        'percentage': round((negative_count / total) * 100, 2) if total > 0 else 0
                    },
                    'neutral': {
                        'count': neutral_count,
                        'percentage': round((neutral_count / total) * 100, 2) if total > 0 else 0
                    }
                },
                'note': 'Basic keyword analysis - will be replaced with ML model'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment distribution: {e}")
            return {
                'movie_name': movie_name,
                'error': str(e)
            }
    
    def get_request_stats(self) -> Dict[str, Any]:
        """
        Get collector statistics
        
        Returns:
            Dictionary with request count and other stats
        """
        return {
            'total_requests': self.request_count,
            'rate_limit_delay': self.rate_limit_delay,
            'last_request_time': self.last_request_time
        }


# Create singleton instance
twitter_collector = TwitterDataCollector()
