"""
TMDB Data Collector
Enhanced collector with batch processing, error handling, and rate limit management
"""
import asyncio
import time
from typing import List, Dict, Any, Optional
from app.services.tmdb_client import TMDBClient
import logging

logger = logging.getLogger(__name__)


class TMDBDataCollector:
    """Enhanced TMDB data collector with batch processing and error handling"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TMDB Data Collector
        
        Args:
            api_key: TMDB API key (optional, uses settings if not provided)
        """
        self.client = TMDBClient(api_key)
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit_delay = 0.25  # 4 requests per second (TMDB limit)
        
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
    
    def search_movie_with_retry(
        self,
        query: str,
        language: str = "te",
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Search for a movie with retry logic
        
        Args:
            query: Movie name to search
            language: Language code
            max_retries: Maximum number of retry attempts
            
        Returns:
            Movie search results or None
        """
        for attempt in range(max_retries):
            try:
                self._handle_rate_limit()
                result = self.client.search_movie(query, language)
                
                if result:
                    logger.info(f"Successfully searched for: {query}")
                    return result
                
                logger.warning(f"Empty result for: {query}, attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Error searching movie '{query}': {e}, attempt {attempt + 1}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached for: {query}")
                    return None
        
        return None
    
    def get_movie_details_with_retry(
        self,
        movie_id: int,
        language: str = "te",
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Get movie details with retry logic
        
        Args:
            movie_id: TMDB movie ID
            language: Language code
            max_retries: Maximum number of retry attempts
            
        Returns:
            Movie details or None
        """
        for attempt in range(max_retries):
            try:
                self._handle_rate_limit()
                result = self.client.get_movie_details(movie_id, language)
                
                if result:
                    logger.info(f"Successfully fetched details for movie ID: {movie_id}")
                    return result
                
            except Exception as e:
                logger.error(f"Error fetching movie details {movie_id}: {e}, attempt {attempt + 1}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
        
        return None
    
    def discover_similar_movies_by_genre(
        self,
        genre_ids: List[int],
        language: str = "te",
        min_vote_count: int = 10,
        max_results: int = 20,
        sort_by: str = "vote_average.desc"
    ) -> List[Dict[str, Any]]:
        """
        Discover movies by genre with filtering
        
        Args:
            genre_ids: List of TMDB genre IDs
            language: Language code
            min_vote_count: Minimum number of votes
            max_results: Maximum number of movies to return
            sort_by: Sort criteria (e.g., 'vote_average.desc', 'popularity.desc')
            
        Returns:
            List of discovered movies
        """
        try:
            self._handle_rate_limit()
            
            result = self.client.discover_movies(
                with_genres=genre_ids,
                language=language,
                sort_by=sort_by,
                with_original_language="te"
            )
            
            if not result or 'results' not in result:
                logger.warning(f"No movies found for genres: {genre_ids}")
                return []
            
            # Filter by vote count and limit results
            movies = [
                movie for movie in result['results']
                if movie.get('vote_count', 0) >= min_vote_count
            ][:max_results]
            
            logger.info(f"Found {len(movies)} movies for genres: {genre_ids}")
            return movies
            
        except Exception as e:
            logger.error(f"Error discovering movies by genre: {e}")
            return []
    
    def get_similar_movies_with_retry(
        self,
        movie_id: int,
        language: str = "te",
        max_results: int = 10,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get similar movies with retry logic
        
        Args:
            movie_id: TMDB movie ID
            language: Language code
            max_results: Maximum number of similar movies
            max_retries: Maximum retry attempts
            
        Returns:
            List of similar movies
        """
        for attempt in range(max_retries):
            try:
                self._handle_rate_limit()
                result = self.client.get_similar_movies(movie_id, language)
                
                if result and 'results' in result:
                    movies = result['results'][:max_results]
                    logger.info(f"Found {len(movies)} similar movies for ID: {movie_id}")
                    return movies
                
            except Exception as e:
                logger.error(f"Error getting similar movies for {movie_id}: {e}, attempt {attempt + 1}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return []
    
    def batch_search_movies(
        self,
        movie_names: List[str],
        language: str = "te"
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Batch search multiple movies
        
        Args:
            movie_names: List of movie names to search
            language: Language code
            
        Returns:
            Dictionary mapping movie names to search results
        """
        results = {}
        total = len(movie_names)
        
        logger.info(f"Batch searching {total} movies")
        
        for idx, movie_name in enumerate(movie_names, 1):
            logger.info(f"Searching movie {idx}/{total}: {movie_name}")
            results[movie_name] = self.search_movie_with_retry(movie_name, language)
            
        successful = sum(1 for r in results.values() if r is not None)
        logger.info(f"Batch search complete: {successful}/{total} successful")
        
        return results
    
    def batch_get_movie_details(
        self,
        movie_ids: List[int],
        language: str = "te"
    ) -> Dict[int, Optional[Dict[str, Any]]]:
        """
        Batch fetch movie details
        
        Args:
            movie_ids: List of TMDB movie IDs
            language: Language code
            
        Returns:
            Dictionary mapping movie IDs to details
        """
        results = {}
        total = len(movie_ids)
        
        logger.info(f"Batch fetching details for {total} movies")
        
        for idx, movie_id in enumerate(movie_ids, 1):
            logger.info(f"Fetching details {idx}/{total}: Movie ID {movie_id}")
            results[movie_id] = self.get_movie_details_with_retry(movie_id, language)
        
        successful = sum(1 for r in results.values() if r is not None)
        logger.info(f"Batch fetch complete: {successful}/{total} successful")
        
        return results
    
    def collect_movie_with_metadata(
        self,
        movie_name: str,
        language: str = "te",
        include_similar: bool = True,
        include_credits: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Collect comprehensive movie data including metadata
        
        Args:
            movie_name: Movie name to search
            language: Language code
            include_similar: Whether to include similar movies
            include_credits: Whether to include cast/crew
            
        Returns:
            Complete movie data package or None
        """
        try:
            # Search for movie
            search_result = self.search_movie_with_retry(movie_name, language)
            if not search_result or not search_result.get('results'):
                logger.warning(f"No results found for: {movie_name}")
                return None
            
            # Get first result
            movie = search_result['results'][0]
            movie_id = movie['id']
            
            # Get detailed info
            details = self.get_movie_details_with_retry(movie_id, language)
            
            # Combine data
            movie_data = {
                'tmdb_id': movie_id,
                'title': movie.get('title'),
                'original_title': movie.get('original_title'),
                'overview': movie.get('overview'),
                'release_date': movie.get('release_date'),
                'vote_average': movie.get('vote_average'),
                'vote_count': movie.get('vote_count'),
                'popularity': movie.get('popularity'),
                'poster_path': movie.get('poster_path'),
                'backdrop_path': movie.get('backdrop_path'),
                'genre_ids': movie.get('genre_ids', []),
                'details': details
            }
            
            # Add similar movies if requested
            if include_similar:
                similar = self.get_similar_movies_with_retry(movie_id, language)
                movie_data['similar_movies'] = similar
            
            # Add credits if requested
            if include_credits:
                self._handle_rate_limit()
                credits = self.client.get_movie_credits(movie_id)
                movie_data['credits'] = credits
            
            logger.info(f"Collected complete data for: {movie_name}")
            return movie_data
            
        except Exception as e:
            logger.error(f"Error collecting movie data for '{movie_name}': {e}")
            return None
    
    def get_genre_list(self, language: str = "te") -> List[Dict[str, Any]]:
        """
        Get list of all movie genres
        
        Args:
            language: Language code
            
        Returns:
            List of genres with IDs and names
        """
        try:
            self._handle_rate_limit()
            result = self.client.get_movie_genres(language)
            
            if result and 'genres' in result:
                logger.info(f"Fetched {len(result['genres'])} genres")
                return result['genres']
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching genre list: {e}")
            return []
    
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
tmdb_collector = TMDBDataCollector()
