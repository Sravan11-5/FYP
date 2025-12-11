"""
Automated Workflow Manager
Handles automatic triggers and coordination for end-to-end workflow
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class WorkflowTrigger:
    """Manages automatic workflow triggers based on events"""
    
    def __init__(self):
        self.triggers: Dict[str, List[Callable]] = defaultdict(list)
        self.trigger_history: List[Dict[str, Any]] = []
        
    def register_trigger(self, event_type: str, callback: Callable):
        """
        Register a callback function to be triggered on specific events.
        
        Args:
            event_type: Type of event (e.g., 'user_search', 'data_update')
            callback: Async function to call when event occurs
        """
        self.triggers[event_type].append(callback)
        logger.info(f"Registered trigger for event: {event_type}")
    
    async def trigger_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Trigger all callbacks registered for an event type.
        
        Args:
            event_type: Type of event occurring
            event_data: Data associated with the event
        """
        logger.info(f"Event triggered: {event_type}")
        
        # Record trigger
        self.trigger_history.append({
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": event_data
        })
        
        # Execute all registered callbacks
        callbacks = self.triggers.get(event_type, [])
        results = []
        
        for callback in callbacks:
            try:
                result = await callback(event_data)
                results.append({
                    "callback": callback.__name__,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Trigger callback failed: {callback.__name__}: {str(e)}")
                results.append({
                    "callback": callback.__name__,
                    "success": False,
                    "error": str(e)
                })
        
        return results


class AutomatedCoordinator:
    """
    Coordinates API calls and database operations automatically.
    Implements retry logic, rate limiting, and error handling.
    """
    
    def __init__(self):
        self.api_call_history: List[Dict[str, Any]] = []
        self.db_operation_history: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_api_calls: Dict[str, datetime] = {}
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 2.0  # seconds
        self.exponential_backoff = True
        self.rate_limit_window = 60  # seconds
        self.max_calls_per_window = 15  # Twitter free tier
    
    async def coordinate_api_call(
        self,
        api_name: str,
        api_function: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Coordinate an API call with automatic retries and rate limiting.
        
        Args:
            api_name: Name of the API (e.g., 'tmdb', 'twitter')
            api_function: Async function to call
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Dict with success status and result or error
        """
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            try:
                # Check rate limit
                await self._check_rate_limit(api_name)
                
                # Make API call
                logger.info(f"[{api_name}] API call attempt {attempt + 1}/{self.max_retries}")
                result = await api_function(*args, **kwargs)
                
                # Record successful call
                self._record_api_call(api_name, True)
                
                return {
                    "success": True,
                    "api": api_name,
                    "result": result,
                    "attempts": attempt + 1
                }
                
            except Exception as e:
                last_error = e
                attempt += 1
                self.error_counts[api_name] += 1
                
                logger.warning(
                    f"[{api_name}] API call failed (attempt {attempt}/{self.max_retries}): {str(e)}"
                )
                
                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = self._calculate_retry_delay(attempt)
                    logger.info(f"[{api_name}] Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        self._record_api_call(api_name, False, str(last_error))
        
        return {
            "success": False,
            "api": api_name,
            "error": str(last_error),
            "attempts": self.max_retries
        }
    
    async def coordinate_db_operation(
        self,
        operation_name: str,
        db_function: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Coordinate a database operation with automatic retries.
        
        Args:
            operation_name: Name of the operation (e.g., 'store_movie', 'get_reviews')
            db_function: Async function to call
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Dict with success status and result or error
        """
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            try:
                logger.info(f"[DB:{operation_name}] Operation attempt {attempt + 1}/{self.max_retries}")
                result = await db_function(*args, **kwargs)
                
                # Record successful operation
                self._record_db_operation(operation_name, True)
                
                return {
                    "success": True,
                    "operation": operation_name,
                    "result": result,
                    "attempts": attempt + 1
                }
                
            except Exception as e:
                last_error = e
                attempt += 1
                self.error_counts[f"db_{operation_name}"] += 1
                
                logger.warning(
                    f"[DB:{operation_name}] Operation failed (attempt {attempt}/{self.max_retries}): {str(e)}"
                )
                
                if attempt < self.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.info(f"[DB:{operation_name}] Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        self._record_db_operation(operation_name, False, str(last_error))
        
        return {
            "success": False,
            "operation": operation_name,
            "error": str(last_error),
            "attempts": self.max_retries
        }
    
    async def _check_rate_limit(self, api_name: str):
        """Check if we're within rate limits for the API"""
        if api_name == "twitter":
            now = datetime.now()
            
            # Count recent calls in the window
            recent_calls = [
                call for call in self.api_call_history
                if call.get("api") == api_name
                and datetime.fromisoformat(call.get("timestamp")) > now - timedelta(seconds=self.rate_limit_window)
            ]
            
            if len(recent_calls) >= self.max_calls_per_window:
                wait_time = self.rate_limit_window
                logger.warning(f"[{api_name}] Rate limit reached. Waiting {wait_time} seconds...")
                await asyncio.sleep(wait_time)
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay before retry with exponential backoff"""
        if self.exponential_backoff:
            return self.retry_delay * (2 ** attempt)
        return self.retry_delay
    
    def _record_api_call(self, api_name: str, success: bool, error: Optional[str] = None):
        """Record an API call in history"""
        self.api_call_history.append({
            "api": api_name,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "error": error
        })
        self.last_api_calls[api_name] = datetime.now()
    
    def _record_db_operation(self, operation: str, success: bool, error: Optional[str] = None):
        """Record a database operation in history"""
        self.db_operation_history.append({
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "error": error
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about API calls and database operations"""
        total_api_calls = len(self.api_call_history)
        successful_api_calls = sum(1 for call in self.api_call_history if call.get("success"))
        
        total_db_ops = len(self.db_operation_history)
        successful_db_ops = sum(1 for op in self.db_operation_history if op.get("success"))
        
        return {
            "api_calls": {
                "total": total_api_calls,
                "successful": successful_api_calls,
                "failed": total_api_calls - successful_api_calls,
                "success_rate": (successful_api_calls / total_api_calls * 100) if total_api_calls > 0 else 0
            },
            "database_operations": {
                "total": total_db_ops,
                "successful": successful_db_ops,
                "failed": total_db_ops - successful_db_ops,
                "success_rate": (successful_db_ops / total_db_ops * 100) if total_db_ops > 0 else 0
            },
            "error_counts": dict(self.error_counts)
        }


class AutomatedWorkflowManager:
    """
    Main manager for automated end-to-end workflows.
    Combines triggers and coordination for complete automation.
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.trigger = WorkflowTrigger()
        self.coordinator = AutomatedCoordinator()
        
        # Register default triggers
        self._register_default_triggers()
        
        logger.info("Automated Workflow Manager initialized")
    
    def _register_default_triggers(self):
        """Register default triggers for common events"""
        
        # Trigger workflow on user search
        async def on_user_search(event_data: Dict[str, Any]):
            movie_name = event_data.get("movie_name")
            collect_new_data = event_data.get("collect_new_data", True)
            max_reviews = event_data.get("max_reviews", 10)
            
            logger.info(f"[TRIGGER] User search detected: {movie_name}")
            
            # Automatically execute workflow
            result = await self.orchestrator.execute_workflow(
                movie_name=movie_name,
                collect_new_data=collect_new_data,
                max_reviews=max_reviews
            )
            
            return result
        
        self.trigger.register_trigger("user_search", on_user_search)
    
    async def handle_user_search(
        self,
        movie_name: str,
        collect_new_data: bool = True,
        max_reviews: int = 10
    ) -> Dict[str, Any]:
        """
        Handle user search event with automatic workflow execution.
        
        This is the main entry point for automated workflows.
        
        Args:
            movie_name: Name of the movie to search
            collect_new_data: Whether to collect fresh data
            max_reviews: Maximum reviews to collect
            
        Returns:
            Complete workflow results
        """
        logger.info(f"[AUTOMATION] Handling user search: {movie_name}")
        
        # Trigger the event
        event_data = {
            "movie_name": movie_name,
            "collect_new_data": collect_new_data,
            "max_reviews": max_reviews,
            "timestamp": datetime.now().isoformat()
        }
        
        results = await self.trigger.trigger_event("user_search", event_data)
        
        if results and results[0].get("success"):
            return results[0].get("result")
        else:
            return {
                "success": False,
                "error": "Automated workflow failed",
                "details": results
            }
    
    async def coordinate_data_collection(
        self,
        movie_name: str,
        max_reviews: int
    ) -> Dict[str, Any]:
        """
        Coordinate TMDB and Twitter data collection with automatic retries.
        
        Args:
            movie_name: Name of the movie
            max_reviews: Maximum reviews to collect
            
        Returns:
            Collection results
        """
        logger.info(f"[COORDINATION] Starting data collection for: {movie_name}")
        
        results = {
            "movie_name": movie_name,
            "tmdb_data": None,
            "twitter_data": None,
            "errors": []
        }
        
        # Coordinate TMDB call
        tmdb_result = await self.coordinator.coordinate_api_call(
            "tmdb",
            self.orchestrator.tmdb_collector.search_movie,
            movie_name
        )
        
        if tmdb_result.get("success"):
            results["tmdb_data"] = tmdb_result.get("result")
        else:
            results["errors"].append(f"TMDB: {tmdb_result.get('error')}")
        
        # Coordinate Twitter call
        twitter_result = await self.coordinator.coordinate_api_call(
            "twitter",
            self.orchestrator.twitter_collector.search_movie_reviews,
            movie_name=movie_name,
            max_results=max_reviews,
            language="te"
        )
        
        if twitter_result.get("success"):
            results["twitter_data"] = twitter_result.get("result")
        else:
            results["errors"].append(f"Twitter: {twitter_result.get('error')}")
        
        results["success"] = len(results["errors"]) == 0
        
        return results
    
    async def coordinate_database_operations(
        self,
        movie_data: Dict[str, Any],
        reviews_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Coordinate database storage operations with automatic retries.
        
        Args:
            movie_data: Movie data to store
            reviews_data: Reviews data to store
            
        Returns:
            Storage results
        """
        logger.info(f"[COORDINATION] Starting database operations")
        
        results = {
            "movie_stored": False,
            "reviews_stored": 0,
            "errors": []
        }
        
        # Store movie
        movie_result = await self.coordinator.coordinate_db_operation(
            "store_movie",
            self.orchestrator.storage_service.store_movie,
            movie_data
        )
        
        if movie_result.get("success"):
            results["movie_stored"] = True
            results["movie_id"] = movie_result.get("result")
        else:
            results["errors"].append(f"Store movie: {movie_result.get('error')}")
        
        # Store reviews
        if reviews_data:
            reviews_result = await self.coordinator.coordinate_db_operation(
                "store_reviews_batch",
                self.orchestrator.storage_service.store_reviews_batch,
                reviews_data
            )
            
            if reviews_result.get("success"):
                results["reviews_stored"] = len(reviews_data)
            else:
                results["errors"].append(f"Store reviews: {reviews_result.get('error')}")
        
        results["success"] = len(results["errors"]) == 0
        
        return results
    
    def get_automation_statistics(self) -> Dict[str, Any]:
        """Get statistics about automated operations"""
        return {
            "coordinator_stats": self.coordinator.get_statistics(),
            "trigger_history": len(self.trigger.trigger_history),
            "recent_triggers": self.trigger.trigger_history[-10:] if self.trigger.trigger_history else []
        }


# Singleton instance
_workflow_manager = None


def get_workflow_manager(orchestrator):
    """Get the singleton workflow manager instance"""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = AutomatedWorkflowManager(orchestrator)
    return _workflow_manager
