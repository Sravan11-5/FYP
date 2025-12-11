"""
Error Handling Utilities
Centralized error handling, retry mechanisms, and circuit breaker pattern
"""
import time
import logging
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"      # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascading failures
    
    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Too many failures, requests blocked immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
        name: str = "default"
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            success_threshold: Consecutive successes needed to close circuit
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_attempt_time = None
        
        logger.info(f"Circuit breaker '{name}' initialized: "
                   f"failure_threshold={failure_threshold}, "
                   f"recovery_timeout={recovery_timeout}s")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                self.state = CircuitState.HALF_OPEN
            else:
                time_since_failure = time.time() - self.last_failure_time if self.last_failure_time else 0
                wait_time = self.recovery_timeout - time_since_failure
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Wait {wait_time:.1f}s before retry."
                )
        
        try:
            self.last_attempt_time = time.time()
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(f"Circuit breaker '{self.name}' success in HALF_OPEN: "
                        f"{self.success_count}/{self.success_threshold}")
            
            if self.success_count >= self.success_threshold:
                self._close_circuit()
        else:
            # Reset failure count on success in CLOSED state
            self.failure_count = 0
    
    def _on_failure(self, exception: Exception):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        logger.warning(f"Circuit breaker '{self.name}' failure: {exception}. "
                      f"Count: {self.failure_count}/{self.failure_threshold}")
        
        if self.state == CircuitState.HALF_OPEN:
            # Any failure in HALF_OPEN reopens circuit
            self._open_circuit()
        elif self.failure_count >= self.failure_threshold:
            self._open_circuit()
    
    def _open_circuit(self):
        """Open circuit breaker"""
        self.state = CircuitState.OPEN
        self.success_count = 0
        logger.error(f"Circuit breaker '{self.name}' OPENED after "
                    f"{self.failure_count} failures")
    
    def _close_circuit(self):
        """Close circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' CLOSED - service recovered")
    
    def reset(self):
        """Manually reset circuit breaker"""
        logger.info(f"Circuit breaker '{self.name}' manually reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'last_attempt_time': self.last_attempt_time
        }


class RetryStrategy:
    """Configurable retry strategy with exponential backoff"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize retry strategy
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff calculation
            jitter: Add random jitter to prevent thundering herd
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff: delay = initial_delay * (base ^ attempt)
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd problem
        if self.jitter:
            import random
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator for automatic retry with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry
        
    Example:
        @with_retry(max_retries=3, initial_delay=1.0)
        def fetch_data():
            # Code that might fail
            pass
    """
    strategy = RetryStrategy(max_retries, initial_delay, max_delay, exponential_base)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = strategy.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}. "
                            f"Waiting {delay:.2f}s..."
                        )
                        
                        if on_retry:
                            on_retry(attempt, e)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}"
                        )
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator


class ErrorTracker:
    """Track and aggregate errors for monitoring"""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize error tracker
        
        Args:
            max_history: Maximum number of errors to keep in history
        """
        self.max_history = max_history
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
    
    def record_error(
        self,
        error_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record an error occurrence
        
        Args:
            error_type: Type/category of error
            message: Error message
            context: Additional context information
        """
        error_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': error_type,
            'message': message,
            'context': context or {}
        }
        
        self.errors.append(error_entry)
        
        # Maintain max history size
        if len(self.errors) > self.max_history:
            self.errors = self.errors[-self.max_history:]
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        logger.error(f"Error recorded: {error_type} - {message}", extra=context or {})
    
    def get_error_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get summary of recent errors
        
        Args:
            time_window_minutes: Time window for summary in minutes
            
        Returns:
            Dictionary with error statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        cutoff_str = cutoff_time.isoformat()
        
        recent_errors = [
            e for e in self.errors 
            if e['timestamp'] >= cutoff_str
        ]
        
        # Count by type
        recent_counts = {}
        for error in recent_errors:
            error_type = error['error_type']
            recent_counts[error_type] = recent_counts.get(error_type, 0) + 1
        
        return {
            'time_window_minutes': time_window_minutes,
            'total_errors': len(recent_errors),
            'error_counts_by_type': recent_counts,
            'most_recent_errors': recent_errors[-5:] if recent_errors else [],
            'all_time_counts': dict(self.error_counts)
        }
    
    def clear_history(self):
        """Clear error history"""
        self.errors.clear()
        logger.info("Error history cleared")


# Global instances
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_error_tracker = ErrorTracker()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    success_threshold: int = 2
) -> CircuitBreaker:
    """
    Get or create a circuit breaker instance
    
    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds to wait before attempting recovery
        success_threshold: Consecutive successes needed to close
        
    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            name=name
        )
    
    return _circuit_breakers[name]


def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance"""
    return _error_tracker


def log_error(
    error_type: str,
    message: str,
    exception: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None
):
    """
    Log error with tracking
    
    Args:
        error_type: Type/category of error
        message: Error message
        exception: Optional exception object
        context: Additional context information
    """
    full_context = context or {}
    
    if exception:
        full_context['exception_type'] = type(exception).__name__
        full_context['exception_message'] = str(exception)
    
    _error_tracker.record_error(error_type, message, full_context)
    
    if exception:
        logger.exception(f"{error_type}: {message}")
    else:
        logger.error(f"{error_type}: {message}", extra=full_context)
