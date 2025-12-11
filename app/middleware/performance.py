"""
Performance Monitoring Middleware
Tracks request/response times and provides profiling data
"""
import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track API endpoint performance metrics.
    Records request duration and collects statistics.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.endpoint_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'errors': 0
        })
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track performance"""
        start_time = time.time()
        
        # Get endpoint path
        path = request.url.path
        method = request.method
        endpoint_key = f"{method} {path}"
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update statistics
            stats = self.endpoint_stats[endpoint_key]
            stats['count'] += 1
            stats['total_time'] += duration
            stats['min_time'] = min(stats['min_time'], duration)
            stats['max_time'] = max(stats['max_time'], duration)
            
            # Track errors
            if response.status_code >= 400:
                stats['errors'] += 1
            
            # Add performance header
            response.headers['X-Response-Time'] = f"{duration:.4f}s"
            
            # Log slow requests (>2 seconds)
            if duration > 2.0:
                logger.warning(
                    f"Slow request: {endpoint_key} took {duration:.2f}s "
                    f"(status: {response.status_code})"
                )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Update error statistics
            stats = self.endpoint_stats[endpoint_key]
            stats['count'] += 1
            stats['errors'] += 1
            stats['total_time'] += duration
            
            logger.error(f"Error in {endpoint_key}: {e}", exc_info=True)
            raise
    
    def get_stats(self):
        """Get performance statistics for all endpoints"""
        results = {}
        
        for endpoint, stats in self.endpoint_stats.items():
            if stats['count'] > 0:
                avg_time = stats['total_time'] / stats['count']
                error_rate = (stats['errors'] / stats['count']) * 100
                
                results[endpoint] = {
                    'requests': stats['count'],
                    'errors': stats['errors'],
                    'error_rate_percent': round(error_rate, 2),
                    'avg_time_seconds': round(avg_time, 4),
                    'min_time_seconds': round(stats['min_time'], 4),
                    'max_time_seconds': round(stats['max_time'], 4),
                    'total_time_seconds': round(stats['total_time'], 2)
                }
        
        return results
    
    def reset_stats(self):
        """Reset all performance statistics"""
        self.endpoint_stats.clear()
        logger.info("Performance statistics reset")


# Global middleware instance
_performance_middleware = None


def get_performance_middleware() -> PerformanceMonitoringMiddleware:
    """Get performance middleware instance"""
    global _performance_middleware
    return _performance_middleware


def set_performance_middleware(middleware: PerformanceMonitoringMiddleware):
    """Set performance middleware instance"""
    global _performance_middleware
    _performance_middleware = middleware
