"""
In-Memory Caching Service
Provides caching for frequently accessed movie data and recommendations
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import hashlib

logger = logging.getLogger(__name__)


class CacheService:
    """
    In-memory cache service with TTL support for movie data and recommendations.
    Uses a simple dictionary-based cache with automatic expiration.
    """
    
    def __init__(self, default_ttl_seconds: int = 3600):
        """
        Initialize cache service.
        
        Args:
            default_ttl_seconds: Default time-to-live for cache entries (1 hour default)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl_seconds
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
        logger.info(f"Cache Service initialized with TTL: {default_ttl_seconds}s")
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        expiry = entry.get('expiry')
        if not expiry:
            return True
        return datetime.utcnow() > expiry
    
    def _generate_key(self, prefix: str, identifier: Any) -> str:
        """Generate cache key from prefix and identifier"""
        if isinstance(identifier, dict):
            # For complex objects, create hash
            id_str = json.dumps(identifier, sort_keys=True)
            id_hash = hashlib.md5(id_str.encode()).hexdigest()
            return f"{prefix}:{id_hash}"
        return f"{prefix}:{str(identifier)}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        entry = self._cache.get(key)
        
        if not entry:
            self.stats['misses'] += 1
            return None
        
        if self._is_expired(entry):
            # Remove expired entry
            del self._cache[key]
            self.stats['evictions'] += 1
            self.stats['misses'] += 1
            return None
        
        self.stats['hits'] += 1
        return entry.get('value')
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (uses default if not provided)
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        
        self._cache[key] = {
            'value': value,
            'expiry': expiry,
            'created_at': datetime.utcnow()
        }
        self.stats['sets'] += 1
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cache entries")
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        count = len(expired_keys)
        if count > 0:
            self.stats['evictions'] += count
            logger.info(f"Cleaned up {count} expired cache entries")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': len(self._cache)
        }
    
    # Movie-specific cache methods
    
    def get_movie(self, tmdb_id: int) -> Optional[Dict]:
        """Get cached movie data by TMDB ID"""
        key = self._generate_key('movie', tmdb_id)
        return self.get(key)
    
    def set_movie(self, tmdb_id: int, movie_data: Dict, ttl_seconds: Optional[int] = None) -> None:
        """Cache movie data by TMDB ID"""
        key = self._generate_key('movie', tmdb_id)
        self.set(key, movie_data, ttl_seconds)
    
    def get_recommendations(self, search_params: Dict) -> Optional[List[Dict]]:
        """Get cached recommendations for search parameters"""
        key = self._generate_key('recommendations', search_params)
        return self.get(key)
    
    def set_recommendations(self, search_params: Dict, recommendations: List[Dict], 
                          ttl_seconds: Optional[int] = None) -> None:
        """Cache recommendations for search parameters"""
        key = self._generate_key('recommendations', search_params)
        self.set(key, recommendations, ttl_seconds)
    
    def get_movie_reviews(self, tmdb_id: int) -> Optional[List[Dict]]:
        """Get cached movie reviews by TMDB ID"""
        key = self._generate_key('reviews', tmdb_id)
        return self.get(key)
    
    def set_movie_reviews(self, tmdb_id: int, reviews: List[Dict], 
                         ttl_seconds: Optional[int] = None) -> None:
        """Cache movie reviews by TMDB ID"""
        key = self._generate_key('reviews', tmdb_id)
        self.set(key, reviews, ttl_seconds)
    
    def invalidate_movie(self, tmdb_id: int) -> None:
        """Invalidate all cache entries related to a movie"""
        movie_key = self._generate_key('movie', tmdb_id)
        reviews_key = self._generate_key('reviews', tmdb_id)
        
        self.delete(movie_key)
        self.delete(reviews_key)
        
        logger.info(f"Invalidated cache for movie {tmdb_id}")


# Global cache instance
_cache_instance: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get or create global cache service instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService(default_ttl_seconds=3600)  # 1 hour default
    return _cache_instance


def clear_cache():
    """Clear global cache instance"""
    global _cache_instance
    if _cache_instance:
        _cache_instance.clear()
