"""
System Health and Monitoring Endpoints
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any
import logging
from datetime import datetime
from app.services.cache_service import get_cache_service
from app.database import get_database
from app.middleware.performance import get_performance_middleware

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        System health status
    """
    try:
        db = get_database()
        # Try to ping database
        await db.command('ping')
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": db_status,
            "cache": "healthy"
        }
    }


@router.get("/cache/stats")
async def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Cache performance metrics
    """
    cache = get_cache_service()
    stats = cache.get_stats()
    
    return {
        "cache_statistics": stats,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/cache/clear")
async def clear_cache() -> Dict[str, str]:
    """
    Clear all cache entries (admin operation).
    
    Returns:
        Success message
    """
    cache = get_cache_service()
    cache.clear()
    
    logger.info("Cache cleared via API endpoint")
    
    return {
        "message": "Cache cleared successfully",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/cache/cleanup")
async def cleanup_expired_cache() -> Dict[str, Any]:
    """
    Clean up expired cache entries.
    
    Returns:
        Number of entries removed
    """
    cache = get_cache_service()
    removed_count = cache.cleanup_expired()
    
    return {
        "message": f"Cleaned up {removed_count} expired cache entries",
        "removed_count": removed_count,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/performance/stats")
async def get_performance_stats() -> Dict[str, Any]:
    """
    Get API performance statistics for all endpoints.
    
    Returns:
        Performance metrics including request count, avg/min/max times, error rates
    """
    middleware = get_performance_middleware()
    
    if middleware is None:
        return {
            "error": "Performance monitoring not available",
            "stats": {}
        }
    
    stats = middleware.get_stats()
    
    # Calculate overall statistics
    total_requests = sum(s['requests'] for s in stats.values())
    total_errors = sum(s['errors'] for s in stats.values())
    overall_error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
    
    return {
        "overall": {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate_percent": round(overall_error_rate, 2)
        },
        "endpoints": stats,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/performance/reset")
async def reset_performance_stats() -> Dict[str, str]:
    """
    Reset performance statistics (admin operation).
    
    Returns:
        Success message
    """
    middleware = get_performance_middleware()
    
    if middleware is None:
        return {
            "error": "Performance monitoring not available"
        }
    
    middleware.reset_stats()
    
    logger.info("Performance statistics reset via API endpoint")
    
    return {
        "message": "Performance statistics reset successfully",
        "timestamp": datetime.utcnow().isoformat()
    }
