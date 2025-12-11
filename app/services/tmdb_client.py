"""
TMDB API Client
Handles all interactions with The Movie Database (TMDB) API
"""
import requests
from typing import Optional, List, Dict, Any
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class TMDBClient:
    """Client for interacting with TMDB API"""
    
    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://image.tmdb.org/t/p"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TMDB API client
        
        Args:
            api_key: TMDB API key or Bearer token (defaults to settings.TMDB_API_KEY)
        """
        self.api_key = api_key or settings.TMDB_API_KEY
        if not self.api_key:
            logger.warning("TMDB API key not configured")
        
        self.session = requests.Session()
        
        # Check if it's a Bearer token (JWT format) or API key
        if self.api_key and self.api_key.startswith('eyJ'):
            # Bearer token format (v4 auth)
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'
        else:
            # API key format (v3 auth)
            self.session.params = {"api_key": self.api_key}
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make a request to TMDB API
        
        Args:
            endpoint: API endpoint (e.g., '/search/movie')
            params: Additional query parameters
            
        Returns:
            JSON response data or None if request fails
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"TMDB API request failed: {e}")
            return None
    
    def search_movie(
        self, 
        query: str, 
        language: str = "te",
        page: int = 1,
        include_adult: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Search for movies by title
        
        Args:
            query: Movie title to search for
            language: Language code (default: 'te' for Telugu)
            page: Page number for pagination
            include_adult: Whether to include adult content
            
        Returns:
            Search results containing movie data
        """
        params = {
            "query": query,
            "language": language,
            "page": page,
            "include_adult": include_adult
        }
        
        logger.info(f"Searching TMDB for movie: '{query}'")
        return self._make_request("/search/movie", params)
    
    def get_movie_details(
        self, 
        movie_id: int, 
        language: str = "te"
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a movie
        
        Args:
            movie_id: TMDB movie ID
            language: Language code (default: 'te' for Telugu)
            
        Returns:
            Detailed movie data
        """
        params = {"language": language}
        logger.info(f"Fetching TMDB movie details for ID: {movie_id}")
        return self._make_request(f"/movie/{movie_id}", params)
    
    def get_movie_credits(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """
        Get cast and crew information for a movie
        
        Args:
            movie_id: TMDB movie ID
            
        Returns:
            Cast and crew data
        """
        logger.info(f"Fetching TMDB movie credits for ID: {movie_id}")
        return self._make_request(f"/movie/{movie_id}/credits")
    
    def get_similar_movies(
        self, 
        movie_id: int, 
        language: str = "te",
        page: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Get movies similar to a given movie
        
        Args:
            movie_id: TMDB movie ID
            language: Language code (default: 'te' for Telugu)
            page: Page number for pagination
            
        Returns:
            Similar movies data
        """
        params = {"language": language, "page": page}
        logger.info(f"Fetching similar movies for ID: {movie_id}")
        return self._make_request(f"/movie/{movie_id}/similar", params)
    
    def get_movie_genres(self, language: str = "te") -> Optional[Dict[str, Any]]:
        """
        Get list of all movie genres
        
        Args:
            language: Language code (default: 'te' for Telugu)
            
        Returns:
            List of genres
        """
        params = {"language": language}
        logger.info("Fetching TMDB movie genres")
        return self._make_request("/genre/movie/list", params)
    
    def discover_movies(
        self,
        with_genres: Optional[List[int]] = None,
        language: str = "te",
        sort_by: str = "popularity.desc",
        page: int = 1,
        with_original_language: str = "te"
    ) -> Optional[Dict[str, Any]]:
        """
        Discover movies by different criteria
        
        Args:
            with_genres: List of genre IDs
            language: Language code
            sort_by: Sort order (e.g., 'popularity.desc', 'vote_average.desc')
            page: Page number
            with_original_language: Filter by original language
            
        Returns:
            Discovered movies data
        """
        params = {
            "language": language,
            "sort_by": sort_by,
            "page": page,
            "with_original_language": with_original_language
        }
        
        if with_genres:
            params["with_genres"] = ",".join(map(str, with_genres))
        
        logger.info(f"Discovering movies with params: {params}")
        return self._make_request("/discover/movie", params)
    
    def get_poster_url(
        self, 
        poster_path: Optional[str], 
        size: str = "w500"
    ) -> Optional[str]:
        """
        Get full URL for movie poster
        
        Args:
            poster_path: Poster path from TMDB API response
            size: Image size (w92, w154, w185, w342, w500, w780, original)
            
        Returns:
            Full poster URL or None if poster_path is None
        """
        if not poster_path:
            return None
        return f"{self.IMAGE_BASE_URL}/{size}{poster_path}"
    
    def get_backdrop_url(
        self, 
        backdrop_path: Optional[str], 
        size: str = "w1280"
    ) -> Optional[str]:
        """
        Get full URL for movie backdrop
        
        Args:
            backdrop_path: Backdrop path from TMDB API response
            size: Image size (w300, w780, w1280, original)
            
        Returns:
            Full backdrop URL or None if backdrop_path is None
        """
        if not backdrop_path:
            return None
        return f"{self.IMAGE_BASE_URL}/{size}{backdrop_path}"


# Create singleton instance
tmdb_client = TMDBClient()
