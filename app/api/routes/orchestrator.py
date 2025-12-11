"""
Orchestrator API Routes
Endpoints for the Agentic AI orchestrator
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

from app.agents import get_orchestrator

router = APIRouter()
logger = logging.getLogger(__name__)


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


@router.post("/orchestrator/auto-search")
async def automated_search(movie_name: str, max_reviews: int = 10):
    """
    Automated search endpoint with automatic workflow triggering.
    
    This endpoint demonstrates the automated workflow manager that:
    - Automatically triggers data collection on user search
    - Coordinates API calls with retry logic
    - Manages database operations without manual intervention
    - Handles errors automatically
    
    Args:
        movie_name: Name of the movie to search
        max_reviews: Maximum reviews to collect (default: 10)
        
    Returns:
        Complete workflow results with automation statistics
    """
    try:
        logger.info(f"Automated search triggered for: {movie_name}")
        
        orchestrator = get_orchestrator()
        automation_manager = orchestrator.get_automation_manager()
        
        # Use automated workflow manager
        result = await automation_manager.handle_user_search(
            movie_name=movie_name,
            collect_new_data=True,
            max_reviews=max_reviews
        )
        
        # Get automation statistics
        stats = automation_manager.get_automation_statistics()
        
        return {
            "success": result.get("success"),
            "workflow_result": result,
            "automation_stats": stats,
            "message": "Automated workflow completed"
        }
        
    except Exception as e:
        logger.error(f"Automated search failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
