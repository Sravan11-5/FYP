"""
Twitter API Client
Handles all interactions with Twitter API v2 for fetching Telugu movie reviews
"""
import tweepy
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class TwitterClient:
    """Client for interacting with Twitter API v2"""
    
    def __init__(
        self, 
        bearer_token: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        """
        Initialize Twitter API client
        
        Args:
            bearer_token: Twitter API Bearer Token (OAuth 2.0)
            api_key: Twitter API Key
            api_secret: Twitter API Secret
        """
        self.bearer_token = bearer_token or settings.TWITTER_BEARER_TOKEN
        self.api_key = api_key or settings.TWITTER_API_KEY
        self.api_secret = api_secret or settings.TWITTER_API_SECRET
        
        if not self.bearer_token:
            logger.warning("Twitter Bearer Token not configured")
        
        try:
            # Initialize Twitter API v2 client with Bearer Token
            self.client = tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                wait_on_rate_limit=True
            )
            logger.info("Twitter API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API client: {e}")
            self.client = None
    
    def search_movie_reviews(
        self,
        movie_name: str,
        max_results: int = 100,
        days_back: int = 30,
        language: str = "te"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search for Telugu movie reviews on Twitter
        
        Args:
            movie_name: Name of the movie to search for
            max_results: Maximum number of tweets to return (10-100)
            days_back: Number of days to search back
            language: Language code (default: 'te' for Telugu)
            
        Returns:
            List of tweet data with text, metrics, and metadata
        """
        if not self.client:
            logger.error("Twitter client not initialized")
            return None
        
        try:
            # Build search query for Telugu movie reviews
            # Include common Telugu review keywords
            query = f'({movie_name} OR #{movie_name.replace(" ", "")}) '
            query += 'lang:te '  # Telugu language tweets
            query += '(సినిమా OR మూవీ OR review OR రివ్యూ OR అభిప్రాయం) '
            query += '-is:retweet'  # Exclude retweets
            
            # Calculate start time (Twitter API requires RFC 3339 format)
            start_time = datetime.utcnow() - timedelta(days=days_back)
            
            logger.info(f"Searching Twitter for: '{query}'")
            
            # Search recent tweets
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),  # API limit is 100
                start_time=start_time,
                tweet_fields=[
                    'created_at', 
                    'public_metrics', 
                    'author_id',
                    'lang',
                    'possibly_sensitive'
                ],
                expansions=['author_id'],
                user_fields=['name', 'username', 'verified']
            )
            
            if not response.data:
                logger.info(f"No tweets found for movie: {movie_name}")
                return []
            
            # Format tweet data
            tweets = []
            users = {user.id: user for user in response.includes.get('users', [])}
            
            for tweet in response.data:
                author = users.get(tweet.author_id)
                
                tweet_data = {
                    'tweet_id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'language': tweet.lang,
                    'author_id': tweet.author_id,
                    'author_name': author.name if author else None,
                    'author_username': author.username if author else None,
                    'author_verified': author.verified if author else False,
                    'likes': tweet.public_metrics.get('like_count', 0),
                    'retweets': tweet.public_metrics.get('retweet_count', 0),
                    'replies': tweet.public_metrics.get('reply_count', 0),
                    'possibly_sensitive': tweet.possibly_sensitive
                }
                tweets.append(tweet_data)
            
            logger.info(f"Found {len(tweets)} tweets for movie: {movie_name}")
            return tweets
            
        except tweepy.errors.TweepyException as e:
            logger.error(f"Twitter API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error searching Twitter: {e}")
            return None
    
    def get_tweet_by_id(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific tweet by ID
        
        Args:
            tweet_id: Twitter tweet ID
            
        Returns:
            Tweet data or None if not found
        """
        if not self.client:
            logger.error("Twitter client not initialized")
            return None
        
        try:
            response = self.client.get_tweet(
                tweet_id,
                tweet_fields=['created_at', 'public_metrics', 'lang'],
                expansions=['author_id'],
                user_fields=['name', 'username', 'verified']
            )
            
            if not response.data:
                return None
            
            tweet = response.data
            author = response.includes['users'][0] if response.includes.get('users') else None
            
            return {
                'tweet_id': tweet.id,
                'text': tweet.text,
                'created_at': tweet.created_at,
                'language': tweet.lang,
                'author_id': tweet.author_id,
                'author_name': author.name if author else None,
                'author_username': author.username if author else None,
                'author_verified': author.verified if author else False,
                'likes': tweet.public_metrics.get('like_count', 0),
                'retweets': tweet.public_metrics.get('retweet_count', 0),
                'replies': tweet.public_metrics.get('reply_count', 0)
            }
            
        except tweepy.errors.TweepyException as e:
            logger.error(f"Twitter API error: {e}")
            return None
    
    def search_telugu_tweets(
        self,
        keywords: List[str],
        max_results: int = 100,
        days_back: int = 7
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search for Telugu tweets by keywords
        
        Args:
            keywords: List of keywords to search for
            max_results: Maximum number of tweets to return
            days_back: Number of days to search back
            
        Returns:
            List of tweet data
        """
        if not self.client:
            logger.error("Twitter client not initialized")
            return None
        
        try:
            # Build query with keywords
            keyword_query = ' OR '.join(keywords)
            query = f'({keyword_query}) lang:te -is:retweet'
            
            start_time = datetime.utcnow() - timedelta(days=days_back)
            
            logger.info(f"Searching Telugu tweets with keywords: {keywords}")
            
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                start_time=start_time,
                tweet_fields=['created_at', 'public_metrics', 'lang']
            )
            
            if not response.data:
                return []
            
            tweets = [
                {
                    'tweet_id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'language': tweet.lang,
                    'likes': tweet.public_metrics.get('like_count', 0),
                    'retweets': tweet.public_metrics.get('retweet_count', 0)
                }
                for tweet in response.data
            ]
            
            logger.info(f"Found {len(tweets)} Telugu tweets")
            return tweets
            
        except tweepy.errors.TweepyException as e:
            logger.error(f"Twitter API error: {e}")
            return None


# Create singleton instance
twitter_client = TwitterClient()
