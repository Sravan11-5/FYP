"""
API Routes for Telugu Movie Recommendation System
"""
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime
import logging

from app.models.api_models import (
    MovieSearchRequest,
    MovieSearchResponse,
    MovieBasicInfo,
    HealthCheckResponse,
    ErrorResponse
)
from app.dependencies import (
    get_database,
    get_tmdb_client,
    get_twitter_client
)
from app.services.tmdb_client import TMDBClient
from app.services.twitter_client import TwitterClient
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check the health status of the system and its services"
)
async def health_check(
    db: AsyncIOMotorDatabase = Depends(get_database),
    tmdb_client: TMDBClient = Depends(get_tmdb_client),
    twitter_client: TwitterClient = Depends(get_twitter_client)
):
    """
    Health check endpoint
    
    Returns:
        HealthCheckResponse: System health status
    """
    services_status = {
        "database": "unknown",
        "tmdb_api": "unknown",
        "twitter_api": "unknown"
    }
    
    # Check database connection
    try:
        await db.command("ping")
        services_status["database"] = "healthy"
        logger.info("Database health check: OK")
    except Exception as e:
        services_status["database"] = f"unhealthy: {str(e)}"
        logger.error(f"Database health check failed: {e}")
    
    # Check TMDB API
    try:
        genres = tmdb_client.get_movie_genres()
        if genres:
            services_status["tmdb_api"] = "healthy"
            logger.info("TMDB API health check: OK")
        else:
            services_status["tmdb_api"] = "unhealthy: no response"
    except Exception as e:
        services_status["tmdb_api"] = f"unhealthy: {str(e)}"
        logger.error(f"TMDB API health check failed: {e}")
    
    # Check Twitter API (just verify client is initialized)
    try:
        if twitter_client.client:
            services_status["twitter_api"] = "healthy"
            logger.info("Twitter API health check: OK")
        else:
            services_status["twitter_api"] = "unhealthy: client not initialized"
    except Exception as e:
        services_status["twitter_api"] = f"unhealthy: {str(e)}"
        logger.error(f"Twitter API health check failed: {e}")
    
    # Determine overall status
    all_healthy = all(status == "healthy" for status in services_status.values())
    overall_status = "healthy" if all_healthy else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version=settings.APP_VERSION,
        services=services_status
    )


@router.post(
    "/search",
    response_model=MovieSearchResponse,
    summary="Search Movies",
    description="Search for Telugu movies by name using TMDB API",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def search_movies(
    request: MovieSearchRequest,
    tmdb_client: TMDBClient = Depends(get_tmdb_client),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Search for movies by name
    
    Args:
        request: Movie search request with movie name
        tmdb_client: TMDB API client (injected)
        db: Database instance (injected)
    
    Returns:
        MovieSearchResponse: Search results with movie information
    
    Raises:
        HTTPException: If search fails
    """
    try:
        logger.info(f"Searching for movie: {request.movie_name}")
        
        # Search movies using TMDB
        search_response = tmdb_client.search_movie(
            query=request.movie_name,
            language=request.language
        )
        
        if not search_response or 'results' not in search_response:
            logger.info(f"No movies found for: {request.movie_name}")
            return MovieSearchResponse(
                success=True,
                message="No movies found matching your search",
                movies=[],
                total_results=0
            )
        
        movies_data = search_response.get('results', [])
        
        # Convert to response model
        movies = []
        for movie in movies_data:
            poster_url = None
            if movie.get('poster_path'):
                poster_url = tmdb_client.get_poster_url(movie['poster_path'])
            
            movies.append(MovieBasicInfo(
                tmdb_id=movie['id'],
                title=movie.get('title', 'Unknown'),
                original_title=movie.get('original_title'),
                release_date=movie.get('release_date'),
                rating=movie.get('vote_average'),
                poster_url=poster_url,
                overview=movie.get('overview')
            ))
        
        logger.info(f"Found {len(movies)} movies for: {request.movie_name}")
        
        # Store search in database for analytics (fire and forget)
        try:
            await db.searches.insert_one({
                "query": request.movie_name,
                "language": request.language,
                "results_count": len(movies),
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            logger.warning(f"Failed to log search: {e}")
        
        return MovieSearchResponse(
            success=True,
            message=f"Found {len(movies)} movie(s)",
            movies=movies,
            total_results=len(movies)
        )
        
    except Exception as e:
        logger.error(f"Search failed for '{request.movie_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
