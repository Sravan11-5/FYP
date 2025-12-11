"""
Search API Routes
Endpoints for searching movies and collecting reviews
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from app.collectors import get_tmdb_collector, get_twitter_collector
from app.services import get_storage_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchRequest(BaseModel):
    """Request model for movie search"""
    movie_name: str = Field(..., description="Name of the movie to search")
    language: str = Field(default="te", description="Language code (default: te for Telugu)")
    collect_reviews: bool = Field(default=True, description="Whether to collect Twitter reviews")
    max_reviews: int = Field(default=10, ge=1, le=20, description="Maximum reviews to collect (1-20)")
    

class SearchResponse(BaseModel):
    """Response model for movie search"""
    success: bool
    movie: Optional[Dict[str, Any]] = None
    reviews_collected: int = 0
    message: str
    

class BatchSearchRequest(BaseModel):
    """Request model for batch movie search"""
    movie_names: List[str] = Field(..., description="List of movie names to search")
    language: str = Field(default="te", description="Language code")
    max_reviews_per_movie: int = Field(default=5, ge=1, le=10, description="Max reviews per movie (1-10)")


class BatchSearchResponse(BaseModel):
    """Response model for batch search"""
    success: bool
    movies_processed: int = 0
    total_reviews_collected: int = 0
    results: List[Dict[str, Any]] = []
    message: str


@router.post("/movie", response_model=SearchResponse)
async def search_and_collect_movie(request: SearchRequest):
    """
    Search for a movie and optionally collect Twitter reviews
    
    Args:
        request: SearchRequest containing movie name and collection options
        
    Returns:
        SearchResponse with movie data and review collection status
        
    Raises:
        HTTPException: If movie not found or collection fails
    """
    try:
        logger.info(f"Searching for movie: {request.movie_name}")
        
        # Initialize collectors and storage
        tmdb_collector = get_tmdb_collector()
        twitter_collector = get_twitter_collector()
        storage_service = get_storage_service()
        
        # Search for movie on TMDB
        movie_results = await tmdb_collector.search_movie(
            movie_name=request.movie_name,
            language=request.language
        )
        
        if not movie_results:
            logger.warning(f"No movie found: {request.movie_name}")
            raise HTTPException(
                status_code=404,
                detail=f"No movie found with name: {request.movie_name}"
            )
        
        # Get the first (most relevant) result
        movie_data = movie_results[0]
        tmdb_id = movie_data.get('id')
        
        # Get detailed movie information
        detailed_movie = await tmdb_collector.get_movie_details(tmdb_id)
        
        if not detailed_movie:
            logger.error(f"Failed to get movie details for TMDB ID: {tmdb_id}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve movie details"
            )
        
        # Parse and store movie data
        parsed_movie = tmdb_collector.parse_movie_data(detailed_movie)
        movie_id = await storage_service.store_movie(parsed_movie)
        
        logger.info(f"Movie stored with ID: {movie_id}")
        
        reviews_collected = 0
        
        # Collect Twitter reviews if requested
        if request.collect_reviews:
            logger.info(f"Collecting reviews for: {request.movie_name}")
            
            reviews = await twitter_collector.search_movie_reviews(
                movie_name=request.movie_name,
                max_results=request.max_reviews,
                language=request.language
            )
            
            if reviews:
                # Parse and store reviews
                parsed_reviews = [
                    twitter_collector.parse_tweet_data(review, str(movie_id))
                    for review in reviews
                ]
                
                await storage_service.store_reviews_batch(parsed_reviews)
                reviews_collected = len(parsed_reviews)
                
                logger.info(f"Collected and stored {reviews_collected} reviews")
            else:
                logger.warning(f"No reviews found for: {request.movie_name}")
        
        # Prepare response
        response_movie = {
            "movie_id": str(movie_id),
            "tmdb_id": parsed_movie.get('tmdb_id'),
            "title": parsed_movie.get('title'),
            "original_title": parsed_movie.get('original_title'),
            "release_date": parsed_movie.get('release_date'),
            "genres": parsed_movie.get('genres', []),
            "rating": parsed_movie.get('rating'),
            "overview": parsed_movie.get('overview')
        }
        
        return SearchResponse(
            success=True,
            movie=response_movie,
            reviews_collected=reviews_collected,
            message=f"Successfully processed movie: {request.movie_name}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_and_collect_movie: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/batch", response_model=BatchSearchResponse)
async def batch_search_and_collect(request: BatchSearchRequest):
    """
    Search for multiple movies and collect reviews (batch operation)
    Uses conservative settings for Twitter free tier
    
    Args:
        request: BatchSearchRequest with movie names and collection options
        
    Returns:
        BatchSearchResponse with processing results
    """
    try:
        logger.info(f"Batch search for {len(request.movie_names)} movies")
        
        # Initialize collectors and storage
        tmdb_collector = get_tmdb_collector()
        twitter_collector = get_twitter_collector()
        storage_service = get_storage_service()
        
        results = []
        movies_processed = 0
        total_reviews = 0
        
        # Process each movie
        for movie_name in request.movie_names:
            try:
                logger.info(f"Processing movie: {movie_name}")
                
                # Search and get movie details from TMDB
                movie_results = await tmdb_collector.search_movie(
                    movie_name=movie_name,
                    language=request.language
                )
                
                if not movie_results:
                    logger.warning(f"No movie found: {movie_name}")
                    results.append({
                        "movie_name": movie_name,
                        "success": False,
                        "error": "Movie not found",
                        "reviews_collected": 0
                    })
                    continue
                
                # Get detailed info
                movie_data = movie_results[0]
                tmdb_id = movie_data.get('id')
                detailed_movie = await tmdb_collector.get_movie_details(tmdb_id)
                
                if not detailed_movie:
                    logger.warning(f"Failed to get details: {movie_name}")
                    results.append({
                        "movie_name": movie_name,
                        "success": False,
                        "error": "Failed to get movie details",
                        "reviews_collected": 0
                    })
                    continue
                
                # Store movie
                parsed_movie = tmdb_collector.parse_movie_data(detailed_movie)
                movie_id = await storage_service.store_movie(parsed_movie)
                
                movies_processed += 1
                reviews_collected = 0
                
                # Collect reviews (conservative settings for free tier)
                reviews = await twitter_collector.search_movie_reviews(
                    movie_name=movie_name,
                    max_results=request.max_reviews_per_movie,
                    language=request.language
                )
                
                if reviews:
                    parsed_reviews = [
                        twitter_collector.parse_tweet_data(review, str(movie_id))
                        for review in reviews
                    ]
                    
                    await storage_service.store_reviews_batch(parsed_reviews)
                    reviews_collected = len(parsed_reviews)
                    total_reviews += reviews_collected
                
                results.append({
                    "movie_name": movie_name,
                    "movie_id": str(movie_id),
                    "tmdb_id": parsed_movie.get('tmdb_id'),
                    "success": True,
                    "reviews_collected": reviews_collected
                })
                
                logger.info(f"Successfully processed {movie_name}: {reviews_collected} reviews")
                
            except Exception as e:
                logger.error(f"Error processing {movie_name}: {str(e)}")
                results.append({
                    "movie_name": movie_name,
                    "success": False,
                    "error": str(e),
                    "reviews_collected": 0
                })
        
        return BatchSearchResponse(
            success=True,
            movies_processed=movies_processed,
            total_reviews_collected=total_reviews,
            results=results,
            message=f"Processed {movies_processed}/{len(request.movie_names)} movies"
        )
        
    except Exception as e:
        logger.error(f"Error in batch_search_and_collect: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing error: {str(e)}"
        )


@router.get("/stats")
async def get_collection_stats():
    """
    Get statistics about collected data
    
    Returns:
        Dict with movies count, reviews count, and average reviews per movie
    """
    try:
        storage_service = get_storage_service()
        stats = await storage_service.get_storage_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stats: {str(e)}"
        )


@router.get("/discover")
async def discover_movies(
    genre: str = Query(..., description="Genre name or ID"),
    min_rating: float = Query(default=6.0, ge=0, le=10, description="Minimum rating"),
    max_results: int = Query(default=20, ge=1, le=50, description="Maximum results")
):
    """
    Discover movies by genre
    
    Args:
        genre: Genre name or ID
        min_rating: Minimum rating threshold
        max_results: Maximum number of results
        
    Returns:
        List of discovered movies
    """
    try:
        tmdb_collector = get_tmdb_collector()
        
        # Genre mapping for Telugu/Indian cinema
        genre_map = {
            "action": 28,
            "drama": 18,
            "thriller": 53,
            "comedy": 35,
            "romance": 10749,
            "family": 10751,
            "crime": 80,
            "horror": 27
        }
        
        # Convert genre name to ID if needed
        genre_ids = []
        if genre.lower() in genre_map:
            genre_ids = [genre_map[genre.lower()]]
        elif genre.isdigit():
            genre_ids = [int(genre)]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid genre: {genre}. Use genre name or ID."
            )
        
        movies = await tmdb_collector.discover_movies_by_genre(
            genre_ids=genre_ids,
            min_vote_average=min_rating,
            max_results=max_results
        )
        
        return {
            "success": True,
            "count": len(movies),
            "movies": movies
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in discover_movies: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Discovery error: {str(e)}"
        )
