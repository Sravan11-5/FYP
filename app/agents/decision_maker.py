"""
Autonomous Decision Making Module
Implements intelligent decision-making for the agentic AI system
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1  # User-requested movie
    HIGH = 2      # Similar movies, recommendations
    MEDIUM = 3    # Additional metadata
    LOW = 4       # Background updates


class DataFreshnessPolicy(Enum):
    """Policies for determining data freshness"""
    ALWAYS_FRESH = "always_fresh"      # Always fetch new data
    PREFER_CACHED = "prefer_cached"    # Use cache if available
    TIME_BASED = "time_based"          # Based on staleness threshold
    SMART = "smart"                     # Intelligent decision based on multiple factors


@dataclass
class TaskDefinition:
    """Definition of a task with priority and dependencies"""
    task_id: str
    task_type: str
    priority: TaskPriority
    params: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CachedDataInfo:
    """Information about cached data"""
    data_type: str
    identifier: str
    cached_at: datetime
    size: int
    is_complete: bool


class AutonomousDecisionMaker:
    """
    Makes autonomous decisions about data fetching, task prioritization,
    and failure handling.
    """
    
    def __init__(self, storage_service=None):
        self.storage_service = storage_service
        
        # Configuration
        self.staleness_threshold_hours = 24  # Data older than 24 hours is stale
        self.min_reviews_threshold = 5       # Minimum reviews to consider cached data useful
        self.freshness_policy = DataFreshnessPolicy.SMART
        
        # Task queue
        self.task_queue: List[TaskDefinition] = []
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[Tuple[str, str]] = []  # (task_id, error)
        
        # Decision history
        self.decision_history: List[Dict[str, Any]] = []
        
        logger.info("Autonomous Decision Maker initialized")
    
    async def decide_data_strategy(
        self,
        data_type: str,
        identifier: str,
        user_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Autonomously decide whether to fetch new data or use cached data.
        
        Args:
            data_type: Type of data (e.g., 'movie', 'reviews')
            identifier: Unique identifier (e.g., movie_id, movie_name)
            user_preference: Optional user preference ('fresh', 'cached', None)
            
        Returns:
            Decision with reasoning and action
        """
        logger.info(f"[DECISION] Evaluating data strategy for {data_type}: {identifier}")
        
        decision = {
            "data_type": data_type,
            "identifier": identifier,
            "timestamp": datetime.now().isoformat(),
            "action": None,
            "reasoning": [],
            "confidence": 0.0
        }
        
        # Factor 1: User preference (highest priority)
        if user_preference:
            if user_preference.lower() == 'fresh':
                decision["action"] = "fetch_new"
                decision["reasoning"].append("User explicitly requested fresh data")
                decision["confidence"] = 1.0
                self._record_decision(decision)
                return decision
            elif user_preference.lower() == 'cached':
                decision["action"] = "use_cached"
                decision["reasoning"].append("User explicitly requested cached data")
                decision["confidence"] = 1.0
                self._record_decision(decision)
                return decision
        
        # Factor 2: Check if cached data exists
        cached_info = await self._check_cached_data(data_type, identifier)
        
        if not cached_info:
            decision["action"] = "fetch_new"
            decision["reasoning"].append("No cached data available")
            decision["confidence"] = 1.0
            self._record_decision(decision)
            return decision
        
        # Factor 3: Data staleness
        age_hours = (datetime.now() - cached_info.cached_at).total_seconds() / 3600
        is_stale = age_hours > self.staleness_threshold_hours
        
        if is_stale:
            decision["reasoning"].append(
                f"Cached data is stale ({age_hours:.1f} hours old, threshold: {self.staleness_threshold_hours} hours)"
            )
            decision["confidence"] += 0.3
        else:
            decision["reasoning"].append(
                f"Cached data is fresh ({age_hours:.1f} hours old)"
            )
            decision["confidence"] += 0.4
        
        # Factor 4: Data completeness
        if data_type == 'reviews':
            if cached_info.size < self.min_reviews_threshold:
                decision["reasoning"].append(
                    f"Insufficient cached reviews ({cached_info.size} < {self.min_reviews_threshold})"
                )
                decision["confidence"] += 0.3
            else:
                decision["reasoning"].append(
                    f"Sufficient cached reviews ({cached_info.size})"
                )
                decision["confidence"] -= 0.2
        
        # Factor 5: Policy application
        if self.freshness_policy == DataFreshnessPolicy.ALWAYS_FRESH:
            decision["action"] = "fetch_new"
            decision["reasoning"].append("Policy: Always fetch fresh data")
        elif self.freshness_policy == DataFreshnessPolicy.PREFER_CACHED:
            decision["action"] = "use_cached"
            decision["reasoning"].append("Policy: Prefer cached data")
        elif self.freshness_policy == DataFreshnessPolicy.TIME_BASED:
            decision["action"] = "fetch_new" if is_stale else "use_cached"
            decision["reasoning"].append("Policy: Time-based decision")
        else:  # SMART
            # Smart decision based on all factors
            if is_stale or (data_type == 'reviews' and cached_info.size < self.min_reviews_threshold):
                decision["action"] = "fetch_new"
                decision["reasoning"].append("Smart decision: Fetch new data for better quality")
            else:
                decision["action"] = "use_cached"
                decision["reasoning"].append("Smart decision: Use cached data for efficiency")
        
        # Normalize confidence
        decision["confidence"] = min(1.0, decision["confidence"])
        
        logger.info(f"[DECISION] Action: {decision['action']} (confidence: {decision['confidence']:.2f})")
        self._record_decision(decision)
        
        return decision
    
    async def _check_cached_data(
        self,
        data_type: str,
        identifier: str
    ) -> Optional[CachedDataInfo]:
        """Check if cached data exists and get its info"""
        if not self.storage_service:
            return None
        
        try:
            if data_type == 'movie':
                # Check if movie exists in database
                movie = await self.storage_service.get_movie_by_tmdb_id(identifier)
                if movie:
                    return CachedDataInfo(
                        data_type='movie',
                        identifier=identifier,
                        cached_at=movie.get('created_at', datetime.now()),
                        size=1,
                        is_complete=True
                    )
            
            elif data_type == 'reviews':
                # Check if reviews exist for movie
                reviews = await self.storage_service.get_reviews_for_movie(identifier)
                if reviews:
                    # Get oldest review date as cache date
                    oldest_date = min(
                        review.get('created_at', datetime.now())
                        for review in reviews
                    )
                    return CachedDataInfo(
                        data_type='reviews',
                        identifier=identifier,
                        cached_at=oldest_date,
                        size=len(reviews),
                        is_complete=len(reviews) >= self.min_reviews_threshold
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking cached data: {str(e)}")
            return None
    
    def prioritize_tasks(
        self,
        user_movie: str,
        similar_movies: List[str],
        additional_movies: Optional[List[str]] = None
    ) -> List[TaskDefinition]:
        """
        Autonomously prioritize tasks based on importance.
        
        Args:
            user_movie: The movie the user searched for (highest priority)
            similar_movies: Similar movies to fetch (high priority)
            additional_movies: Additional movies for context (medium priority)
            
        Returns:
            Ordered list of tasks by priority
        """
        logger.info("[PRIORITIZATION] Organizing tasks by priority")
        
        tasks = []
        task_counter = 0
        
        # Priority 1: User-requested movie (CRITICAL)
        tasks.append(TaskDefinition(
            task_id=f"task_{task_counter}",
            task_type="fetch_user_movie",
            priority=TaskPriority.CRITICAL,
            params={"movie_name": user_movie}
        ))
        task_counter += 1
        logger.info(f"[PRIORITY] CRITICAL: Fetch user movie '{user_movie}'")
        
        # Priority 2: User movie reviews (CRITICAL)
        tasks.append(TaskDefinition(
            task_id=f"task_{task_counter}",
            task_type="fetch_user_reviews",
            priority=TaskPriority.CRITICAL,
            params={"movie_name": user_movie},
            dependencies=[f"task_{task_counter-1}"]
        ))
        task_counter += 1
        logger.info(f"[PRIORITY] CRITICAL: Fetch reviews for '{user_movie}'")
        
        # Priority 3: Similar movies (HIGH)
        for movie in similar_movies[:5]:  # Limit to top 5
            tasks.append(TaskDefinition(
                task_id=f"task_{task_counter}",
                task_type="fetch_similar_movie",
                priority=TaskPriority.HIGH,
                params={"movie_name": movie},
                dependencies=[f"task_0"]  # Depends on user movie
            ))
            task_counter += 1
        logger.info(f"[PRIORITY] HIGH: Fetch {min(5, len(similar_movies))} similar movies")
        
        # Priority 4: Analysis task (HIGH)
        tasks.append(TaskDefinition(
            task_id=f"task_{task_counter}",
            task_type="analyze_sentiment",
            priority=TaskPriority.HIGH,
            params={"movie_name": user_movie},
            dependencies=[f"task_1"]  # Depends on reviews
        ))
        task_counter += 1
        logger.info(f"[PRIORITY] HIGH: Analyze sentiment")
        
        # Priority 5: Generate recommendations (HIGH)
        tasks.append(TaskDefinition(
            task_id=f"task_{task_counter}",
            task_type="generate_recommendations",
            priority=TaskPriority.HIGH,
            params={"movie_name": user_movie},
            dependencies=[f"task_{task_counter-1}"]  # Depends on analysis
        ))
        task_counter += 1
        logger.info(f"[PRIORITY] HIGH: Generate recommendations")
        
        # Priority 6: Additional movies (MEDIUM)
        if additional_movies:
            for movie in additional_movies[:3]:  # Limit to 3
                tasks.append(TaskDefinition(
                    task_id=f"task_{task_counter}",
                    task_type="fetch_additional_movie",
                    priority=TaskPriority.MEDIUM,
                    params={"movie_name": movie}
                ))
                task_counter += 1
            logger.info(f"[PRIORITY] MEDIUM: Fetch {min(3, len(additional_movies))} additional movies")
        
        # Sort by priority
        tasks.sort(key=lambda t: (t.priority.value, t.created_at))
        
        self.task_queue = tasks
        logger.info(f"[PRIORITIZATION] Created {len(tasks)} tasks in priority order")
        
        return tasks
    
    async def execute_tasks_with_priority(
        self,
        tasks: List[TaskDefinition],
        executor_func
    ) -> Dict[str, Any]:
        """
        Execute tasks in priority order with dependency management.
        
        Args:
            tasks: List of tasks to execute
            executor_func: Async function to execute each task
            
        Returns:
            Execution results
        """
        logger.info("[EXECUTION] Starting prioritized task execution")
        
        results = {
            "total_tasks": len(tasks),
            "completed": [],
            "failed": [],
            "skipped": [],
            "execution_order": []
        }
        
        pending_tasks = tasks.copy()
        
        while pending_tasks:
            # Find tasks that can be executed (dependencies met)
            executable_tasks = [
                task for task in pending_tasks
                if all(dep in self.completed_tasks for dep in task.dependencies)
            ]
            
            if not executable_tasks:
                logger.warning("[EXECUTION] No executable tasks, possible circular dependency")
                results["skipped"].extend([task.task_id for task in pending_tasks])
                break
            
            # Execute tasks in priority order
            for task in executable_tasks:
                try:
                    logger.info(f"[EXECUTION] Executing {task.task_type} (priority: {task.priority.name})")
                    
                    result = await executor_func(task)
                    
                    results["completed"].append({
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "priority": task.priority.name,
                        "result": result
                    })
                    results["execution_order"].append(task.task_id)
                    
                    self.completed_tasks.append(task.task_id)
                    pending_tasks.remove(task)
                    
                    logger.info(f"[EXECUTION] ✓ Completed {task.task_type}")
                    
                except Exception as e:
                    logger.error(f"[EXECUTION] ✗ Failed {task.task_type}: {str(e)}")
                    
                    # Handle failure with retry
                    should_retry = await self.handle_failure(task, str(e))
                    
                    if should_retry and task.retry_count < task.max_retries:
                        task.retry_count += 1
                        logger.info(f"[EXECUTION] Retrying {task.task_type} (attempt {task.retry_count + 1})")
                        # Keep in pending for retry
                    else:
                        results["failed"].append({
                            "task_id": task.task_id,
                            "task_type": task.task_type,
                            "error": str(e),
                            "retry_count": task.retry_count
                        })
                        self.failed_tasks.append((task.task_id, str(e)))
                        pending_tasks.remove(task)
        
        logger.info(f"[EXECUTION] Completed: {len(results['completed'])}, Failed: {len(results['failed'])}")
        
        return results
    
    async def handle_failure(
        self,
        task: TaskDefinition,
        error: str
    ) -> bool:
        """
        Autonomously handle task failures and decide whether to retry.
        
        Args:
            task: The failed task
            error: Error message
            
        Returns:
            True if should retry, False otherwise
        """
        logger.info(f"[FAILURE HANDLING] Analyzing failure for {task.task_type}")
        
        # Categorize error
        error_lower = error.lower()
        
        # Transient errors - should retry
        transient_errors = [
            'timeout',
            'connection',
            'network',
            'unavailable',
            'rate limit',
            'temporary'
        ]
        
        is_transient = any(err in error_lower for err in transient_errors)
        
        # Permanent errors - should not retry
        permanent_errors = [
            'not found',
            'invalid',
            'forbidden',
            'unauthorized',
            'bad request'
        ]
        
        is_permanent = any(err in error_lower for err in permanent_errors)
        
        # Decision logic
        if is_permanent:
            logger.info(f"[FAILURE HANDLING] Permanent error detected, will not retry")
            return False
        
        if is_transient:
            logger.info(f"[FAILURE HANDLING] Transient error detected, will retry")
            
            # Calculate backoff delay
            delay = self._calculate_backoff_delay(task.retry_count)
            logger.info(f"[FAILURE HANDLING] Waiting {delay:.1f}s before retry")
            await asyncio.sleep(delay)
            
            return True
        
        # Check retry count
        if task.retry_count < task.max_retries:
            logger.info(f"[FAILURE HANDLING] Retry attempt {task.retry_count + 1}/{task.max_retries}")
            
            delay = self._calculate_backoff_delay(task.retry_count)
            await asyncio.sleep(delay)
            
            return True
        
        logger.info(f"[FAILURE HANDLING] Max retries reached, giving up")
        return False
    
    def _calculate_backoff_delay(self, retry_count: int) -> float:
        """Calculate exponential backoff delay"""
        base_delay = 2.0  # seconds
        max_delay = 60.0  # seconds
        
        delay = base_delay * (2 ** retry_count)
        return min(delay, max_delay)
    
    def _record_decision(self, decision: Dict[str, Any]):
        """Record decision in history"""
        self.decision_history.append(decision)
        
        # Keep only last 100 decisions
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about decisions made"""
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "actions": {},
                "confidence_avg": 0.0
            }
        
        actions = {}
        total_confidence = 0.0
        
        for decision in self.decision_history:
            action = decision.get("action", "unknown")
            actions[action] = actions.get(action, 0) + 1
            total_confidence += decision.get("confidence", 0.0)
        
        return {
            "total_decisions": len(self.decision_history),
            "actions": actions,
            "confidence_avg": total_confidence / len(self.decision_history),
            "recent_decisions": self.decision_history[-5:]
        }
    
    def configure(
        self,
        staleness_hours: Optional[int] = None,
        min_reviews: Optional[int] = None,
        policy: Optional[DataFreshnessPolicy] = None
    ):
        """Configure decision maker parameters"""
        if staleness_hours is not None:
            self.staleness_threshold_hours = staleness_hours
            logger.info(f"[CONFIG] Staleness threshold: {staleness_hours} hours")
        
        if min_reviews is not None:
            self.min_reviews_threshold = min_reviews
            logger.info(f"[CONFIG] Min reviews threshold: {min_reviews}")
        
        if policy is not None:
            self.freshness_policy = policy
            logger.info(f"[CONFIG] Freshness policy: {policy.value}")


# Singleton instance
_decision_maker = None


def get_decision_maker(storage_service=None):
    """Get the singleton decision maker instance"""
    global _decision_maker
    if _decision_maker is None:
        _decision_maker = AutonomousDecisionMaker(storage_service)
    return _decision_maker
