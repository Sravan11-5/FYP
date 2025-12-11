"""
TMDB Data Collector
Fetches movie metadata from The Movie Database (TMDB) API
"""

import aiohttp
import logging
from typing import List, Dict, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class TMDBDataCollector:
    """
    Collects movie data from TMDB API
    Handles movie search, details, and genre-based discovery
    """
    
    BASE_URL = "https://api.themoviedb.org/3"
    
    def __init__(self):
        """Initialize TMDB collector with API credentials"""
        self.api_key = settings.TMDB_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json;charset=utf-8"
        }
        logger.info("TMDB Data Collector initialized")
    
    async def search_movie(self, movie_name: str, language: str = "te") -> List[Dict]:
        """
        Search for movies by name.
        
        Args:
            movie_name: Name of the movie to search for
            language: Language for results (te=Telugu, en=English)
            
        Returns:
            List of movie results with basic info
        """
        endpoint = f"{self.BASE_URL}/search/movie"
        params = {
            "query": movie_name,
            "language": language,
            "include_adult": "false"
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
                        results = data.get('results', [])
                        logger.info(
                            f"Found {len(results)} movies for query: {movie_name}"
                        )
                        return results
                    else:
                        logger.error(
                            f"TMDB search failed: {response.status} - {await response.text()}"
                        )
                        return []
        
        except Exception as e:
            logger.error(f"Error searching movie on TMDB: {e}", exc_info=True)
            return []
    
    async def get_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """
        Get detailed information for a specific movie.
        
        Args:
            tmdb_id: TMDB movie ID
            
        Returns:
            Dictionary with detailed movie information
        """
        endpoint = f"{self.BASE_URL}/movie/{tmdb_id}"
        params = {"language": "te"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    endpoint,
                    headers=self.headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        movie_data = await response.json()
                        logger.info(
                            f"Retrieved details for movie: {movie_data.get('title')}"
                        )
                        return movie_data
                    else:
                        logger.error(
                            f"Failed to get movie details: {response.status}"
                        )
                        return None
        
        except Exception as e:
            logger.error(f"Error getting movie details: {e}", exc_info=True)
            return None
    
    async def discover_movies_by_genre(
        self,
        genre_ids: List[int],
        min_vote_average: float = 6.0,
        max_results: int = 20
    ) -> List[Dict]:
        """
        Discover movies by genre with rating filter.
        
        Args:
            genre_ids: List of TMDB genre IDs
            min_vote_average: Minimum rating (0-10)
            max_results: Maximum number of results
            
        Returns:
            List of movie results
        """
        endpoint = f"{self.BASE_URL}/discover/movie"
        
        # TMDB returns 20 movies per page
        pages_needed = (max_results + 19) // 20
        all_movies = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for page in range(1, min(pages_needed + 1, 4)):  # Max 3 pages (60 movies)
                    params = {
                        "with_genres": ",".join(map(str, genre_ids)),
                        "with_original_language": "te",  # Telugu movies only!
                        "vote_average.gte": min_vote_average,
                        "sort_by": "popularity.desc",
                        "page": page,
                        "language": "en"  # Get ENGLISH titles for Twitter search!
                    }
                    
                    async with session.get(
                        endpoint,
                        headers=self.headers,
                        params=params
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            movies = data.get('results', [])
                            all_movies.extend(movies)
                            
                            if len(all_movies) >= max_results:
                                break
                        else:
                            logger.error(
                                f"Discovery failed on page {page}: {response.status}"
                            )
                            break
            
            logger.info(
                f"Discovered {len(all_movies[:max_results])} movies with genres {genre_ids}"
            )
            return all_movies[:max_results]
        
        except Exception as e:
            logger.error(f"Error discovering movies: {e}", exc_info=True)
            return []
    
    async def get_movie_reviews(
        self,
        tmdb_id: int,
        max_results: int = 5
    ) -> List[Dict]:
        """
        Get user reviews for a specific movie from TMDB.
        
        Args:
            tmdb_id: TMDB movie ID
            max_results: Maximum number of reviews to fetch
            
        Returns:
            List of review dictionaries with author, content, rating
        """
        endpoint = f"{self.BASE_URL}/movie/{tmdb_id}/reviews"
        params = {"language": "en"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    endpoint,
                    headers=self.headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        reviews = data.get('results', [])[:max_results]
                        logger.info(
                            f"Retrieved {len(reviews)} reviews for movie ID: {tmdb_id}"
                        )
                        return reviews
                    else:
                        logger.error(
                            f"Failed to get reviews: {response.status}"
                        )
                        return []
        
        except Exception as e:
            logger.error(f"Error getting reviews: {e}", exc_info=True)
            return []
    
    async def get_similar_movies(
        self,
        tmdb_id: int,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Get movies similar to a given movie.
        
        Args:
            tmdb_id: TMDB movie ID
            max_results: Maximum number of results
            
        Returns:
            List of similar movies
        """
        endpoint = f"{self.BASE_URL}/movie/{tmdb_id}/similar"
        params = {
            "language": "te",
            "page": 1
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
                        movies = data.get('results', [])[:max_results]
                        logger.info(
                            f"Found {len(movies)} similar movies for TMDB ID: {tmdb_id}"
                        )
                        return movies
                    else:
                        logger.error(
                            f"Failed to get similar movies: {response.status}"
                        )
                        return []
        
        except Exception as e:
            logger.error(f"Error getting similar movies: {e}", exc_info=True)
            return []
    
    def parse_movie_data(self, movie_data: Dict) -> Dict:
        """
        Parse TMDB movie data into our database format.
        
        Args:
            movie_data: Raw TMDB movie data
            
        Returns:
            Parsed movie data for database storage
        """
        return {
            "tmdb_id": movie_data.get("id"),
            "title": movie_data.get("title") or movie_data.get("original_title"),
            "original_title": movie_data.get("original_title"),
            "overview": movie_data.get("overview", ""),
            "release_date": movie_data.get("release_date"),
            "vote_average": movie_data.get("vote_average", 0),
            "vote_count": movie_data.get("vote_count", 0),
            "popularity": movie_data.get("popularity", 0),
            "poster_path": movie_data.get("poster_path"),
            "backdrop_path": movie_data.get("backdrop_path"),
            "genres": [
                genre.get("name") 
                for genre in movie_data.get("genres", [])
            ] if "genres" in movie_data else [
                str(genre_id) 
                for genre_id in movie_data.get("genre_ids", [])
            ],
            "original_language": movie_data.get("original_language"),
            "adult": movie_data.get("adult", False)
        }


# Singleton instance
_tmdb_collector = None

def get_tmdb_collector() -> TMDBDataCollector:
    """
    Get singleton instance of TMDB collector.
    
    Returns:
        TMDBDataCollector instance
    """
    global _tmdb_collector
    if _tmdb_collector is None:
        _tmdb_collector = TMDBDataCollector()
    return _tmdb_collector
