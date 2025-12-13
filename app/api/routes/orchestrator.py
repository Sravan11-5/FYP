"""
Orchestrator API Routes
Endpoints for the Agentic AI orchestrator
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import time

from app.agents import get_orchestrator

router = APIRouter(prefix="/api", tags=["orchestrator"])
logger = logging.getLogger(__name__)

# TMDB Image Base URL
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

def get_poster_url(poster_path: Optional[str]) -> str:
    """Convert TMDB poster path to full URL"""
    if not poster_path:
        return "https://via.placeholder.com/500x750?text=No+Poster"
    if poster_path.startswith('http'):
        return poster_path
    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"


class WorkflowRequest(BaseModel):
    """Request to start a new workflow"""
    movie_name: str = Field(..., description="Name of the movie to process")
    collect_new_data: bool = Field(
        default=True,
        description="Whether to collect fresh data from APIs"
    )
    max_reviews: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of reviews to collect (1-50)"
    )


class WorkflowResponse(BaseModel):
    """Response from workflow execution"""
    success: bool
    workflow_id: str
    movie: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    recommendations: Optional[list] = None
    error: Optional[str] = None


@router.post("/orchestrator/execute", response_model=WorkflowResponse)
async def execute_workflow(request: WorkflowRequest):
    """
    Execute the complete end-to-end agentic workflow.
    
    This endpoint orchestrates all three agents:
    1. Data Collector Agent - Fetches movie data and reviews
    2. Analyzer Agent - Processes reviews with ML model
    3. Recommender Agent - Generates personalized recommendations
    
    Args:
        request: WorkflowRequest containing movie name and options
        
    Returns:
        WorkflowResponse with complete workflow results
        
    Example:
        ```json
        POST /api/orchestrator/execute
        {
            "movie_name": "RRR",
            "collect_new_data": true,
            "max_reviews": 10
        }
        ```
    """
    try:
        logger.info(f"Executing workflow for movie: {request.movie_name}")
        
        orchestrator = get_orchestrator()
        
        result = await orchestrator.execute_workflow(
            movie_name=request.movie_name,
            collect_new_data=request.collect_new_data,
            max_reviews=request.max_reviews
        )
        
        if result.get("success"):
            return WorkflowResponse(
                success=True,
                workflow_id=result.get("workflow_id"),
                movie=result.get("movie"),
                analysis=result.get("analysis"),
                recommendations=result.get("recommendations")
            )
        else:
            return WorkflowResponse(
                success=False,
                workflow_id=result.get("workflow_id"),
                error=result.get("error")
            )
            
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orchestrator/status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """
    Get the status of a specific workflow.
    
    Args:
        workflow_id: The unique identifier of the workflow
        
    Returns:
        Workflow state including agents executed and current status
    """
    try:
        orchestrator = get_orchestrator()
        status = await orchestrator.get_workflow_status(workflow_id)
        
        if status.get("error"):
            raise HTTPException(status_code=404, detail=status.get("error"))
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orchestrator/quick-recommend")
async def quick_recommend(movie_name: str):
    """
    Quick recommendation endpoint using cached data only.
    
    This endpoint uses the orchestrator but skips fresh data collection,
    relying on cached data from the database for faster responses.
    
    Args:
        movie_name: Name of the movie
        
    Returns:
        Quick recommendations based on cached data
    """
    try:
        logger.info(f"Quick recommend for: {movie_name}")
        
        orchestrator = get_orchestrator()
        
        result = await orchestrator.execute_workflow(
            movie_name=movie_name,
            collect_new_data=False,  # Use cached data only
            max_reviews=0
        )
        
        if result.get("success"):
            return {
                "success": True,
                "movie": result.get("movie"),
                "recommendations": result.get("recommendations", [])[:5],  # Top 5 only
                "message": "Using cached data for faster response"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=result.get("error", "Movie not found in cache")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick recommend failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orchestrator/background-task")
async def start_background_workflow(
    background_tasks: BackgroundTasks,
    request: WorkflowRequest
):
    """
    Start a workflow in the background and return immediately.
    
    Useful for processing that takes a long time (many reviews to collect).
    Client can poll the status endpoint to check progress.
    
    Args:
        background_tasks: FastAPI background tasks
        request: WorkflowRequest containing movie name and options
        
    Returns:
        Workflow ID for status tracking
    """
    try:
        orchestrator = get_orchestrator()
        
        # Generate workflow ID
        import time
        workflow_id = f"workflow_{int(time.time() * 1000)}"
        
        # Add to background tasks
        async def run_workflow():
            await orchestrator.execute_workflow(
                movie_name=request.movie_name,
                collect_new_data=request.collect_new_data,
                max_reviews=request.max_reviews
            )
        
        background_tasks.add_task(run_workflow)
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": "Workflow started in background",
            "status_endpoint": f"/api/orchestrator/status/{workflow_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start background workflow: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class AutoSearchRequest(BaseModel):
    """Request for automated search"""
    user_input: str = Field(..., description="Movie name to search")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    max_reviews: int = Field(default=10, ge=1, le=50, description="Max reviews to collect")


@router.post("/orchestrator/auto-search")
async def automated_search(request: AutoSearchRequest):
    """
    New workflow: Search or fetch movie from TMDB, translate reviews, recommend by genre.
    
    Steps:
    1. Check if movie exists in database
    2. If not, fetch from TMDB + translate English reviews to Telugu
    3. Store in database
    4. Filter movies by same genre
    5. Use Siamese Network to find similar movies
    """
    try:
        from app.database import get_database
        from app.collectors.tmdb_collector import TMDBDataCollector
        
        movie_name = request.user_input
        logger.info(f"🔍 Auto-search for: {movie_name}")
        
        # Get database connection
        db = get_database()
        
        # STEP 1: Check if movie exists in database (search by title)
        movie = await db.movies.find_one({
            "$or": [
                {"title": {"$regex": movie_name, "$options": "i"}},
                {"original_title": {"$regex": movie_name, "$options": "i"}}
            ]
        })
        
        if movie:
            logger.info(f"✅ Movie '{movie_name}' found in database")
        else:
            # STEP 2: Movie not in database - fetch from TMDB
            logger.info(f"❌ Movie '{movie_name}' not in database. Fetching from TMDB...")
            
            tmdb_collector = TMDBDataCollector()
            
            # Search for movie on TMDB
            search_results = await tmdb_collector.search_movie(movie_name, language="en")
            
            if not search_results:
                return {
                    "status": "failed",
                    "success": False,
                    "error": f"Movie '{movie_name}' not found on TMDB",
                    "message": "Movie not found. Please check spelling and try again.",
                    "recommendations": [],
                    "movie": None,
                    "analysis": None
                }
            
            # Get first result (best match)
            tmdb_movie = search_results[0]
            tmdb_id = tmdb_movie['id']
            
            # Get full movie details
            movie_details = await tmdb_collector.get_movie_details(tmdb_id)
            
            if not movie_details:
                return {
                    "status": "failed",
                    "success": False,
                    "error": "Failed to fetch movie details from TMDB",
                    "message": "Error retrieving movie information",
                    "recommendations": [],
                    "movie": None,
                    "analysis": None
                }
            
            # STEP 3: Fetch and translate reviews from TMDB
            logger.info(f"📝 Fetching TMDB reviews for: {movie_details.get('title')}")
            telugu_reviews = await tmdb_collector.get_reviews_and_translate(tmdb_id, max_reviews=10)
            
            if not telugu_reviews:
                logger.warning(f"No reviews available for {movie_details.get('title')}")
            
            # Get primary genre
            genres = movie_details.get('genres', [])
            primary_genre = genres[0]['name'] if genres else "Unknown"
            
            # STEP 4: Store in database with English title only
            movie_doc = {
                "tmdb_id": tmdb_id,
                "title": movie_details.get('original_title') or movie_details.get('title'),  # Keep English title
                "original_title": movie_details.get('original_title'),
                "genre": primary_genre,
                "rating": movie_details.get('vote_average', 0),
                "overview": movie_details.get('overview', ''),
                "release_date": movie_details.get('release_date', ''),
                "poster_path": movie_details.get('poster_path', ''),
                "reviews": telugu_reviews,  # Only reviews are in Telugu
                "review_count": len(telugu_reviews),
                "has_reviews": len(telugu_reviews) > 0
            }
            
            # Check if movie already exists (might have been added in parallel request)
            existing = await db.movies.find_one({"tmdb_id": tmdb_id})
            if existing:
                logger.info(f"✅ Movie already exists in database (from parallel request)")
                movie = existing
            else:
                result = await db.movies.insert_one(movie_doc)
                movie_doc['_id'] = result.inserted_id
                movie = movie_doc
                logger.info(f"✅ Stored '{movie_details.get('title')}' with {len(telugu_reviews)} Telugu reviews")
        
        # Now continue with existing logic
        reviews = movie.get("reviews", [])
        reviews_count = len(reviews)
        
        # Analyze sentiment if reviews exist
        analysis = None
        if reviews_count > 0:
            try:
                from app.ml.inference import get_model_inference
                
                model = get_model_inference()
                review_texts = [r.get("text", "") for r in reviews]
                
                sentiments = []
                for text in review_texts:
                    try:
                        result = model.predict_sentiment(text)
                        # Handle both dict and float returns
                        if isinstance(result, dict):
                            score = result.get("score", 0.5)
                        else:
                            score = float(result)
                        sentiments.append(score)
                    except Exception as e:
                        logger.warning(f"Failed to predict sentiment: {e}")
                        continue
                
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    
                    # Calculate distribution
                    positive = sum(1 for s in sentiments if s > 0.6)
                    negative = sum(1 for s in sentiments if s < 0.4)
                    neutral = len(sentiments) - positive - negative
                    
                    total = len(sentiments)
                    analysis = {
                        "success": True,
                        "reviews_analyzed": total,
                        "sentiment_distribution": {
                            "positive": positive,
                            "negative": negative,
                            "neutral": neutral,
                            "positive_percentage": round((positive / total * 100), 2) if total > 0 else 0,
                            "negative_percentage": round((negative / total * 100), 2) if total > 0 else 0,
                            "neutral_percentage": round((neutral / total * 100), 2) if total > 0 else 0
                        },
                        "average_sentiment": round(avg_sentiment, 3),
                        "overall_rating": "Positive" if avg_sentiment > 0.6 else "Negative" if avg_sentiment < 0.4 else "Neutral"
                    }
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                analysis = None
        
        # Get recommendations: Filter by genre FIRST, then use Siamese Network
        recommendations = []
        try:
            current_movie_id = str(movie.get("_id"))
            current_genre = movie.get("genre", "").strip()
            
            if not current_genre:
                logger.warning(f"Movie '{movie_name}' has no genre, cannot recommend")
                recommendations = []
            elif reviews_count == 0:
                logger.warning(f"Movie '{movie_name}' has no reviews, cannot recommend")
                recommendations = []
            else:
                # STEP 1: Filter movies by SAME GENRE ONLY
                same_genre_movies = await db.movies.find({
                    "_id": {"$ne": movie.get("_id")},
                    "genre": current_genre,
                    "has_reviews": True
                }).to_list(length=100)
                
                logger.info(f"Found {len(same_genre_movies)} movies with genre '{current_genre}'")
                
                if same_genre_movies:
                    # STEP 2: Load ML model
                    from app.ml.inference import get_model_inference
                    model = get_model_inference()
                    
                    # STEP 3: Calculate similarity using Siamese Network
                    similarities = []
                    current_reviews = [r.get("text", "") for r in reviews[:3]]
                    
                    for other_movie in same_genre_movies:
                        other_reviews = other_movie.get("reviews", [])
                        
                        if len(other_reviews) > 0:
                            try:
                                # Calculate average similarity across review pairs
                                other_review_texts = [r.get("text", "") for r in other_reviews[:3]]
                                
                                similarity_scores = []
                                for i, curr_review in enumerate(current_reviews):
                                    for j, other_review in enumerate(other_review_texts[:2]):
                                        try:
                                            sim = model.compute_similarity(curr_review, other_review)
                                            similarity_scores.append(sim)
                                        except:
                                            continue
                                
                                if similarity_scores:
                                    # Average similarity score
                                    avg_similarity = sum(similarity_scores) / len(similarity_scores)
                                    
                                    similarities.append({
                                        "movie_id": str(other_movie.get("_id")),
                                        "title": other_movie.get("title"),
                                        "tmdb_id": other_movie.get("tmdb_id"),
                                        "genre": other_movie.get("genre", ""),
                                        "similarity": round(avg_similarity, 4),
                                        "vote_average": other_movie.get("vote_average", 0),
                                        "overview": other_movie.get("overview", ""),
                                        "release_date": other_movie.get("release_date", ""),
                                        "poster_path": other_movie.get("poster_path", ""),
                                        "poster_url": get_poster_url(other_movie.get("poster_path"))
                                    })
                            except Exception as e:
                                logger.warning(f"Failed to calculate similarity for {other_movie.get('title')}: {e}")
                                continue
                    
                    # STEP 4: Sort by similarity and return top 5
                    similarities.sort(key=lambda x: x["similarity"], reverse=True)
                    recommendations = similarities[:5]
                    
                    logger.info(f"Recommendations for '{movie_name}' ({current_genre}): {len(recommendations)} movies")
                    if recommendations:
                        logger.info(f"Top match: {recommendations[0]['title']} - {recommendations[0]['similarity']*100:.1f}% similarity")
                else:
                    logger.warning(f"No other movies found with genre '{current_genre}'")
                    recommendations = []
                    
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations = []
        
        # Return results
        return {
            "status": "completed",
            "success": True,
            "task_id": request.session_id or f"task_{int(time.time())}",
            "movie": {
                "id": str(movie.get("_id")),
                "name": movie.get("title", movie_name),
                "tmdb_id": movie.get("tmdb_id"),
                "reviews_analyzed": reviews_count,
                "overview": movie.get("overview", ""),
                "release_date": movie.get("release_date", ""),
                "vote_average": movie.get("vote_average", 0),
                "poster_path": movie.get("poster_path", ""),
                "poster_url": get_poster_url(movie.get("poster_path"))
            },
            "analysis": analysis if analysis else {
                "success": False,
                "message": "No reviews available for analysis"
            },
            "recommendations": recommendations,
            "message": f"Found movie with {reviews_count} reviews and {len(recommendations)} recommendations"
        }
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "success": False,
            "error": str(e),
            "message": "Search failed",
            "recommendations": [],
            "movie": None,
            "analysis": None
        }


@router.get("/orchestrator/automation-stats")
async def get_automation_statistics():
    """
    Get statistics about automated operations.
    
    Returns:
        Statistics including:
        - API call success rates
        - Database operation success rates
        - Error counts
        - Recent trigger history
    """
    try:
        orchestrator = get_orchestrator()
        automation_manager = orchestrator.get_automation_manager()
        
        stats = automation_manager.get_automation_statistics()
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get automation stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orchestrator/decision-stats")
async def get_decision_statistics():
    """
    Get statistics about autonomous decisions made by the system.
    
    Returns:
        Statistics including:
        - Total decisions made
        - Action distribution (fetch_new vs use_cached)
        - Average confidence levels
        - Recent decisions with reasoning
    """
    try:
        orchestrator = get_orchestrator()
        decision_maker = orchestrator.get_decision_maker()
        
        stats = decision_maker.get_decision_statistics()
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get decision stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orchestrator/configure-decisions")
async def configure_decision_maker(
    staleness_hours: Optional[int] = None,
    min_reviews: Optional[int] = None,
    policy: Optional[str] = None
):
    """
    Configure the autonomous decision maker parameters.
    
    Args:
        staleness_hours: Hours before cached data is considered stale
        min_reviews: Minimum reviews to consider cached data useful
        policy: Freshness policy ('always_fresh', 'prefer_cached', 'time_based', 'smart')
        
    Returns:
        Configuration confirmation
    """
    try:
        orchestrator = get_orchestrator()
        decision_maker = orchestrator.get_decision_maker()
        
        # Import policy enum
        from app.agents.decision_maker import DataFreshnessPolicy
        
        policy_enum = None
        if policy:
            policy_map = {
                'always_fresh': DataFreshnessPolicy.ALWAYS_FRESH,
                'prefer_cached': DataFreshnessPolicy.PREFER_CACHED,
                'time_based': DataFreshnessPolicy.TIME_BASED,
                'smart': DataFreshnessPolicy.SMART
            }
            policy_enum = policy_map.get(policy.lower())
        
        decision_maker.configure(
            staleness_hours=staleness_hours,
            min_reviews=min_reviews,
            policy=policy_enum
        )
        
        return {
            "success": True,
            "message": "Decision maker configured",
            "configuration": {
                "staleness_threshold_hours": decision_maker.staleness_threshold_hours,
                "min_reviews_threshold": decision_maker.min_reviews_threshold,
                "freshness_policy": decision_maker.freshness_policy.value
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to configure decision maker: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
