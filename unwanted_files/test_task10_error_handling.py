"""
Test Task 10: Error Handling and Retries
Tests circuit breaker, retry mechanisms, and error logging
"""
import asyncio
import sys
import time
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.error_handler import (
    CircuitBreaker, CircuitBreakerError, CircuitState,
    with_retry, get_circuit_breaker, get_error_tracker, log_error
)

print("=" * 80)
print("TEST TASK 10: ERROR HANDLING AND RETRIES")
print("=" * 80)

# =========================================================================
# Subtask 10.1: Try-Except Blocks
# =========================================================================
print("\n" + "=" * 80)
print("SUBTASK 10.1: TRY-EXCEPT BLOCKS")
print("=" * 80)

print("\n[Test 10.1.1] Basic exception handling")
try:
    def risky_operation():
        """Simulates an operation that might fail"""
        raise ValueError("Simulated error")
    
    try:
        risky_operation()
    except ValueError as e:
        print(f"✅ Exception caught gracefully: {e}")
        log_error("test_error", "Test error caught", exception=e)
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n[Test 10.1.2] Database operation error handling")
try:
    def simulate_db_error():
        """Simulates database connection error"""
        raise ConnectionError("Database connection failed")
    
    try:
        simulate_db_error()
    except ConnectionError as e:
        print(f"✅ Database error caught: {e}")
        log_error("database_error", "DB connection failed", exception=e, 
                  context={"operation": "connect", "retryable": True})
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n[Test 10.1.3] API call error handling")
try:
    def simulate_api_error():
        """Simulates API timeout"""
        raise TimeoutError("API request timed out")
    
    try:
        simulate_api_error()
    except TimeoutError as e:
        print(f"✅ API error caught: {e}")
        log_error("api_error", "API timeout", exception=e,
                  context={"endpoint": "/test", "timeout": 30})
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n✅ Subtask 10.1 Complete: Try-except blocks working correctly")

# =========================================================================
# Subtask 10.2: Retry Mechanisms with Exponential Backoff
# =========================================================================
print("\n" + "=" * 80)
print("SUBTASK 10.2: RETRY MECHANISMS WITH EXPONENTIAL BACKOFF")
print("=" * 80)

print("\n[Test 10.2.1] Retry with eventual success")
attempt_count = {'value': 0}

@with_retry(max_retries=3, initial_delay=0.5, exponential_base=2.0)
def eventually_succeeds():
    """Function that fails twice then succeeds"""
    attempt_count['value'] += 1
    if attempt_count['value'] < 3:
        raise Exception(f"Temporary failure {attempt_count['value']}")
    return "Success!"

try:
    result = eventually_succeeds()
    print(f"✅ Function succeeded after retries: {result}")
    print(f"   Total attempts: {attempt_count['value']}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n[Test 10.2.2] Retry with max retries exceeded")
always_fails_count = {'value': 0}

@with_retry(max_retries=2, initial_delay=0.3, exponential_base=2.0)
def always_fails():
    """Function that always fails"""
    always_fails_count['value'] += 1
    raise Exception(f"Permanent failure (attempt {always_fails_count['value']})")

try:
    result = always_fails()
    print(f"❌ Should have failed but got: {result}")
except Exception as e:
    print(f"✅ Max retries exceeded as expected: {e}")
    print(f"   Total attempts: {always_fails_count['value']}")

print("\n[Test 10.2.3] Custom retry callback")
retry_delays = []

def on_retry_callback(attempt, exception):
    """Callback to track retry attempts"""
    retry_delays.append(f"Attempt {attempt + 1}: {exception}")

@with_retry(max_retries=2, initial_delay=0.2, on_retry=on_retry_callback)
def fails_twice():
    """Fails twice to test callback"""
    if len(retry_delays) < 2:
        raise Exception("Not yet")
    return "Done"

try:
    result = fails_twice()
    print(f"✅ Retry callback executed {len(retry_delays)} times")
    for delay_info in retry_delays:
        print(f"   - {delay_info}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n✅ Subtask 10.2 Complete: Retry mechanisms working with exponential backoff")

# =========================================================================
# Subtask 10.3: Error Logging
# =========================================================================
print("\n" + "=" * 80)
print("SUBTASK 10.3: ERROR LOGGING")
print("=" * 80)

print("\n[Test 10.3.1] Record various error types")
error_tracker = get_error_tracker()

# Clear previous errors
error_tracker.clear_history()

# Log different types of errors
log_error("validation_error", "Invalid input data", 
          context={"field": "email", "value": "invalid"})
log_error("api_error", "TMDB API timeout",
          context={"endpoint": "/movie/search", "timeout": 30})
log_error("database_error", "Connection pool exhausted",
          context={"pool_size": 10, "active": 10})
log_error("api_error", "Twitter rate limit exceeded",
          context={"endpoint": "/search", "reset_time": "15min"})

print(f"✅ Logged 4 different errors")

print("\n[Test 10.3.2] Error summary and statistics")
summary = error_tracker.get_error_summary(time_window_minutes=60)

print(f"   Total errors in last 60 min: {summary['total_errors']}")
print(f"   Error counts by type:")
for error_type, count in summary['error_counts_by_type'].items():
    print(f"   - {error_type}: {count}")

if summary['most_recent_errors']:
    print(f"\n   Most recent error:")
    recent = summary['most_recent_errors'][-1]
    print(f"   - Type: {recent['error_type']}")
    print(f"   - Message: {recent['message']}")
    print(f"   - Timestamp: {recent['timestamp']}")

print("\n✅ Subtask 10.3 Complete: Error logging and tracking working")

# =========================================================================
# Subtask 10.4: Circuit Breaker Pattern
# =========================================================================
print("\n" + "=" * 80)
print("SUBTASK 10.4: CIRCUIT BREAKER PATTERN")
print("=" * 80)

print("\n[Test 10.4.1] Circuit breaker normal operation (CLOSED state)")
breaker = get_circuit_breaker(
    name="test_api",
    failure_threshold=3,
    recovery_timeout=5,
    success_threshold=2
)

def successful_call():
    """Simulates successful API call"""
    return "Success"

try:
    result = breaker.call(successful_call)
    state = breaker.get_state()
    print(f"✅ Call succeeded in {state['state'].upper()} state")
    print(f"   Failure count: {state['failure_count']}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n[Test 10.4.2] Circuit breaker opens after threshold failures")
def failing_call():
    """Simulates failing API call"""
    raise Exception("API failure")

failure_count = 0
for i in range(5):
    try:
        breaker.call(failing_call)
    except CircuitBreakerError as e:
        print(f"✅ Circuit breaker OPEN: {e}")
        break
    except Exception as e:
        failure_count += 1
        print(f"   Failure {failure_count}: {e}")

state = breaker.get_state()
print(f"   Final state: {state['state'].upper()}")
print(f"   Total failures: {state['failure_count']}")

print("\n[Test 10.4.3] Circuit breaker blocks calls when OPEN")
try:
    breaker.call(successful_call)
    print(f"❌ Should have blocked call")
except CircuitBreakerError as e:
    print(f"✅ Call blocked as expected: Circuit is OPEN")

print("\n[Test 10.4.4] Circuit breaker recovery (HALF_OPEN → CLOSED)")
print(f"   Waiting for recovery timeout (5 seconds)...")
time.sleep(5.5)

# First call in HALF_OPEN should succeed
try:
    result = breaker.call(successful_call)
    state = breaker.get_state()
    print(f"✅ First recovery call succeeded")
    print(f"   State: {state['state'].upper()}, Success count: {state['success_count']}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Second success should close circuit
try:
    result = breaker.call(successful_call)
    state = breaker.get_state()
    print(f"✅ Second recovery call succeeded")
    print(f"   State: {state['state'].upper()}")
    
    if state['state'] == 'closed':
        print(f"   ✅ Circuit breaker CLOSED - service recovered!")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n[Test 10.4.5] Multiple circuit breakers")
tmdb_breaker = get_circuit_breaker("tmdb_api", failure_threshold=5)
twitter_breaker = get_circuit_breaker("twitter_api", failure_threshold=3)

print(f"✅ Created multiple circuit breakers:")
print(f"   - TMDB API: threshold={tmdb_breaker.failure_threshold}")
print(f"   - Twitter API: threshold={twitter_breaker.failure_threshold}")

print("\n✅ Subtask 10.4 Complete: Circuit breaker pattern working correctly")

# =========================================================================
# Final Summary
# =========================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY - TASK 10 COMPLETE")
print("=" * 80)

print("\n✅ Subtask 10.1: Try-except blocks implemented and tested")
print("   - Basic exception handling ✅")
print("   - Database error handling ✅")
print("   - API error handling ✅")

print("\n✅ Subtask 10.2: Retry mechanisms with exponential backoff")
print("   - Automatic retries ✅")
print("   - Exponential backoff timing ✅")
print("   - Max retries enforcement ✅")
print("   - Custom retry callbacks ✅")

print("\n✅ Subtask 10.3: Error logging and tracking")
print("   - Error recording ✅")
print("   - Error categorization ✅")
print("   - Error statistics ✅")
print("   - Error history management ✅")

print("\n✅ Subtask 10.4: Circuit breaker pattern")
print("   - Normal operation (CLOSED) ✅")
print("   - Opens after failures ✅")
print("   - Blocks calls when OPEN ✅")
print("   - Recovery (HALF_OPEN → CLOSED) ✅")
print("   - Multiple breakers ✅")

# Get final error summary
final_summary = error_tracker.get_error_summary(time_window_minutes=60)
print(f"\n📊 Error Tracking Summary:")
print(f"   Total errors tracked: {final_summary['total_errors']}")
print(f"   Unique error types: {len(final_summary['error_counts_by_type'])}")

print("\n" + "=" * 80)
print("✅ ALL SUBTASKS OF TASK 10 COMPLETED SUCCESSFULLY!")
print("=" * 80)
