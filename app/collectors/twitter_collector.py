"""
Twitter Data Collector
Fetches Telugu movie reviews from Twitter API v2
CONSERVATIVE MODE: Minimal API calls for free tier
"""

import aiohttp
import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)


class TwitterDataCollector:
    """
    Collects Telugu movie reviews from Twitter API v2
    Designed for FREE TIER with strict rate limiting
    """
    
    BASE_URL = "https://api.twitter.com/2"
    
    # FREE TIER LIMITS (per 15-minute window)
    MAX_REQUESTS_PER_WINDOW = 15  # Conservative limit
    REQUEST_DELAY = 2  # 2 seconds between requests
    
    def __init__(self):
        """Initialize Twitter collector with API credentials"""
        self.bearer_token = settings.TWITTER_BEARER_TOKEN
        self.headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        self.request_count = 0
        self.window_start = datetime.now()
        logger.info("Twitter Data Collector initialized (FREE TIER MODE)")
    
    async def _check_rate_limit(self):
        """
        Check and enforce rate limits for free tier.
        Resets counter every 15 minutes.
        """
        now = datetime.now()
        elapsed = (now - self.window_start).total_seconds()
        
        # Reset counter every 15 minutes
        if elapsed >= 900:  # 15 minutes
            self.request_count = 0
            self.window_start = now
            logger.info("Rate limit window reset")
        
        # Check if we've hit the limit
        if self.request_count >= self.MAX_REQUESTS_PER_WINDOW:
            wait_time = 900 - elapsed
            if wait_time > 0:
                logger.warning(
                    f"Rate limit reached. Waiting {wait_time:.0f} seconds..."
                )
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.window_start = datetime.now()
        
        # Add delay between requests
        if self.request_count > 0:
            await asyncio.sleep(self.REQUEST_DELAY)
        
        self.request_count += 1
    
    async def search_movie_reviews(
        self,
        movie_name: str,
        max_results: int = 10,  # Conservative default
        language: str = "te"
    ) -> List[Dict]:
        """
        Search for Telugu movie reviews on Twitter.
        
        Args:
            movie_name: Name of the movie
            max_results: Maximum tweets to fetch (10-100, default: 10)
            language: Language filter (te=Telugu)
            
        Returns:
            List of tweet data
        """
        await self._check_rate_limit()
        
        # Twitter API requires minimum 10 results
        max_results = max(10, min(max_results, 100))
        
        endpoint = f"{self.BASE_URL}/tweets/search/recent"
        
        # Build search query for Telugu reviews
        # Format: movie_name (lang:te OR #Telugu) -is:retweet
        query = f'"{movie_name}" (lang:{language} OR #Telugu OR #తెలుగు) -is:retweet'
        
        params = {
            "query": query,
            "max_results": max_results,
            "tweet.fields": "created_at,public_metrics,lang,text,author_id",
            "expansions": "author_id",
            "user.fields": "username,verified"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    endpoint,
                    headers=self.headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        tweets = data.get('data', [])
                        
                        logger.info(
                            f"Found {len(tweets)} reviews for '{movie_name}' "
                            f"(Request {self.request_count}/{self.MAX_REQUESTS_PER_WINDOW})"
                        )
                        
                        return tweets
                    
                    elif response.status == 429:
                        # Rate limit hit
                        logger.error("Rate limit exceeded! Waiting...")
                        reset_time = response.headers.get('x-rate-limit-reset')
                        if reset_time:
                            wait_time = int(reset_time) - int(datetime.now().timestamp())
                            await asyncio.sleep(max(wait_time, 900))
                        return []
                    
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Twitter search failed: {response.status} - {error_text}"
                        )
                        return []
        
        except Exception as e:
            logger.error(f"Error searching Twitter: {e}", exc_info=True)
            return []
    
    async def get_user_timeline_reviews(
        self,
        username: str,
        movie_name: str,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Get reviews from a specific user's timeline.
        
        Args:
            username: Twitter username
            movie_name: Movie name to filter by
            max_results: Maximum tweets
            
        Returns:
            List of tweets
        """
        await self._check_rate_limit()
        
        # First, get user ID
        user_endpoint = f"{self.BASE_URL}/users/by/username/{username}"
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get user ID
                async with session.get(
                    user_endpoint,
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get user ID for {username}")
                        return []
                    
                    user_data = await response.json()
                    user_id = user_data.get('data', {}).get('id')
                    
                    if not user_id:
                        return []
                
                # Get user's tweets
                await self._check_rate_limit()
                
                timeline_endpoint = f"{self.BASE_URL}/users/{user_id}/tweets"
                params = {
                    "max_results": max_results,
                    "tweet.fields": "created_at,public_metrics,lang,text"
                }
                
                async with session.get(
                    timeline_endpoint,
                    headers=self.headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        tweets = data.get('data', [])
                        
                        # Filter tweets mentioning the movie
                        filtered = [
                            tweet for tweet in tweets
                            if movie_name.lower() in tweet.get('text', '').lower()
                        ]
                        
                        logger.info(
                            f"Found {len(filtered)} reviews from @{username}"
                        )
                        return filtered
                    
                    return []
        
        except Exception as e:
            logger.error(f"Error getting user timeline: {e}", exc_info=True)
            return []
    
    def parse_tweet_data(self, tweet: Dict, movie_id: str) -> Dict:
        """
        Parse tweet data into our database format.
        
        Args:
            tweet: Raw Twitter API tweet data
            movie_id: Associated movie ID
            
        Returns:
            Parsed review data for database storage
        """
        metrics = tweet.get('public_metrics', {})
        
        return {
            "tweet_id": tweet.get('id'),
            "movie_id": movie_id,
            "text": tweet.get('text'),
            "created_at": tweet.get('created_at'),
            "language": tweet.get('lang'),
            "author_id": tweet.get('author_id'),
            "likes": metrics.get('like_count', 0),
            "retweets": metrics.get('retweet_count', 0),
            "replies": metrics.get('reply_count', 0),
            "quotes": metrics.get('quote_count', 0),
            "engagement_score": (
                metrics.get('like_count', 0) * 1 +
                metrics.get('retweet_count', 0) * 2 +
                metrics.get('reply_count', 0) * 1.5
            )
        }
    
    async def search_reviews_batch(
        self,
        movie_names: List[str],
        max_per_movie: int = 10  # Twitter API minimum is 10
    ) -> Dict[str, List[Dict]]:
        """
        Search reviews for multiple movies (BATCH MODE).
        Use sparingly to conserve API quota.
        
        Args:
            movie_names: List of movie names
            max_per_movie: Maximum reviews per movie (minimum 10 per Twitter API)
            
        Returns:
            Dictionary mapping movie names to review lists
        """
        logger.info(
            f"Batch collection started for {len(movie_names)} movies "
            f"(max {max_per_movie} reviews each)"
        )
        
        results = {}
        
        for i, movie_name in enumerate(movie_names, 1):
            logger.info(
                f"Collecting reviews for {movie_name} ({i}/{len(movie_names)})"
            )
            
            reviews = await self.search_movie_reviews(
                movie_name,
                max_results=max_per_movie
            )
            
            results[movie_name] = reviews
            
            # Log progress
            logger.info(
                f"Collected {len(reviews)} reviews for {movie_name}. "
                f"Total API calls: {self.request_count}/{self.MAX_REQUESTS_PER_WINDOW}"
            )
            
            # Stop if we're close to limit
            if self.request_count >= self.MAX_REQUESTS_PER_WINDOW - 2:
                logger.warning(
                    f"Stopping batch early to preserve rate limit. "
                    f"Processed {i}/{len(movie_names)} movies."
                )
                break
        
        total_reviews = sum(len(reviews) for reviews in results.values())
        logger.info(
            f"Batch collection complete: {total_reviews} reviews from "
            f"{len(results)} movies"
        )
        
        return results


# Singleton instance
_twitter_collector = None

def get_twitter_collector() -> TwitterDataCollector:
    """
    Get singleton instance of Twitter collector.
    
    Returns:
        TwitterDataCollector instance
    """
    global _twitter_collector
    if _twitter_collector is None:
        _twitter_collector = TwitterDataCollector()
    return _twitter_collector
