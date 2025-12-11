"""
Enhanced Data Collection with Circuit Breaker Integration
Example showing how to integrate circuit breaker pattern with existing collectors
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.tmdb_collector import TMDBDataCollector
from app.services.twitter_collector import TwitterDataCollector
from app.utils.error_handler import get_circuit_breaker, CircuitBreakerError, log_error


class EnhancedTMDBCollector(TMDBDataCollector):
    """TMDB Collector with circuit breaker protection"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.circuit_breaker = get_circuit_breaker(
            name="tmdb_api",
            failure_threshold=5,
            recovery_timeout=30,
            success_threshold=2
        )
    
    def search_movie_with_circuit_breaker(self, query: str, language: str = "te"):
        """
        Search movie with circuit breaker protection
        
        Args:
            query: Movie name to search
            language: Language code
            
        Returns:
            Search results or None if circuit is open
        """
        try:
            # Call through circuit breaker
            result = self.circuit_breaker.call(
                self.search_movie_with_retry,
                query,
                language
            )
            return result
            
        except CircuitBreakerError as e:
            log_error(
                "circuit_breaker_open",
                f"TMDB circuit breaker is open: {e}",
                context={"service": "tmdb", "query": query}
            )
            return None
            
        except Exception as e:
            log_error(
                "tmdb_search_error",
                f"Error searching TMDB: {e}",
                exception=e,
                context={"query": query, "language": language}
            )
            return None


class EnhancedTwitterCollector(TwitterDataCollector):
    """Twitter Collector with circuit breaker protection"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.circuit_breaker = get_circuit_breaker(
            name="twitter_api",
            failure_threshold=3,
            recovery_timeout=60,
            success_threshold=2
        )
    
    def collect_reviews_with_circuit_breaker(
        self,
        movie_name: str,
        tmdb_id: int = None,
        max_results: int = 50
    ):
        """
        Collect reviews with circuit breaker protection
        
        Args:
            movie_name: Movie name
            tmdb_id: TMDB movie ID
            max_results: Maximum reviews to collect
            
        Returns:
            List of reviews or empty list if circuit is open
        """
        try:
            # Call through circuit breaker
            result = self.circuit_breaker.call(
                self.collect_movie_reviews_with_metadata,
                movie_name,
                tmdb_id,
                max_results
            )
            return result
            
        except CircuitBreakerError as e:
            log_error(
                "circuit_breaker_open",
                f"Twitter circuit breaker is open: {e}",
                context={"service": "twitter", "movie": movie_name}
            )
            return []
            
        except Exception as e:
            log_error(
                "twitter_collection_error",
                f"Error collecting Twitter reviews: {e}",
                exception=e,
                context={"movie": movie_name, "max_results": max_results}
            )
            return []


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED DATA COLLECTION WITH CIRCUIT BREAKER")
    print("=" * 70)
    
    # Create enhanced collectors
    tmdb = EnhancedTMDBCollector()
    twitter = EnhancedTwitterCollector()
    
    # Test TMDB with circuit breaker
    print("\n[Test] TMDB search with circuit breaker protection")
    result = tmdb.search_movie_with_circuit_breaker("RRR")
    if result:
        print(f"✅ Found {len(result.get('results', []))} results")
    else:
        print("⚠️  Search failed or circuit is open")
    
    # Get circuit breaker states
    print("\n[Status] Circuit Breaker States:")
    tmdb_state = tmdb.circuit_breaker.get_state()
    twitter_state = twitter.circuit_breaker.get_state()
    
    print(f"   TMDB API: {tmdb_state['state'].upper()}")
    print(f"   - Failures: {tmdb_state['failure_count']}")
    
    print(f"   Twitter API: {twitter_state['state'].upper()}")
    print(f"   - Failures: {twitter_state['failure_count']}")
    
    print("\n✅ Enhanced collectors with circuit breaker protection ready!")
