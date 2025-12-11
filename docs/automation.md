# Automated End-to-End Workflow

## Overview

Task 20 implements complete automation of the movie recommendation workflow. The system now automatically handles:
- Data collection triggering on user search
- Coordinated API calls with retry logic
- Database operations without manual intervention
- Comprehensive error handling and recovery

## Architecture

### Components

1. **WorkflowTrigger**
   - Manages event-based workflow triggering
   - Supports multiple callbacks per event
   - Tracks trigger history

2. **AutomatedCoordinator**
   - Coordinates API calls with automatic retries
   - Implements exponential backoff
   - Handles rate limiting
   - Manages database operations
   - Tracks success/failure statistics

3. **AutomatedWorkflowManager**
   - Main automation interface
   - Combines triggers and coordination
   - Provides high-level automation methods

## Automatic Workflow Triggering

### User Search Event

When a user searches for a movie, the workflow automatically:

1. **Triggers Data Collection**
   - Detects user search event
   - Automatically starts TMDB movie lookup
   - Initiates Twitter review collection
   - No manual intervention required

2. **Coordinates API Calls**
   - TMDB API: Search movie, get detailsbt
   - Twitter API: Collect reviews with rate limiting
   - Automatic retries on failures
   - Exponential backoff strategy

3. **Manages Database**
   - Stores movie data automatically
   - Stores reviews with duplicate prevention
   - Handles database errors gracefully
   - Retries failed operations

4. **Returns Results**
   - Complete workflow results
   - Automation statistics
   - Error information if any

## API Coordination

### Retry Logic

The coordinator implements intelligent retry logic:

```python
max_retries = 3
retry_delay = 2.0  # seconds
exponential_backoff = True

# Retry delays: 2s, 4s, 8s
```

**Features:**
- Configurable retry attempts
- Exponential backoff to avoid overwhelming services
- Different strategies for different APIs
- Detailed error logging

### Rate Limiting

Automatic rate limit handling for Twitter API:

```python
rate_limit_window = 60  # seconds
max_calls_per_window = 15  # Twitter free tier

# Automatically waits when limit reached
```

**Features:**
- Tracks API call history
- Calculates remaining quota
- Automatic waiting when limit reached
- Prevents rate limit errors

## Database Operations

### Automatic Management

Database operations are coordinated with:

1. **Retry on Failure**
   - Automatic retry up to 3 times
   - Exponential backoff between retries
   - Detailed error logging

2. **Transaction Safety**
   - Each operation tracked independently
   - Partial success handling
   - Rollback on critical failures

3. **Duplicate Prevention**
   - Checks existing data before insertion
   - Uses unique identifiers (tmdb_id, tweet_id)
   - Prevents data duplication

## Error Handling

### Comprehensive Error Recovery

The system handles errors at multiple levels:

#### 1. API Level Errors
- Network timeouts
- Rate limit exceeded
- Invalid responses
- Service unavailable

**Recovery:**
- Automatic retry with backoff
- Fallback to cached data
- Graceful degradation

#### 2. Database Errors
- Connection failures
- Query timeouts
- Constraint violations
- Storage full

**Recovery:**
- Automatic reconnection
- Retry with backoff
- Alternative storage strategies

#### 3. Data Errors
- Missing required fields
- Invalid data format
- Parsing failures
- Encoding issues

**Recovery:**
- Skip invalid records
- Use default values
- Log for manual review

## API Endpoints

### Automated Search

```http
POST /api/orchestrator/auto-search?movie_name=RRR&max_reviews=10
```

**Features:**
- Automatic workflow triggering
- Coordinated API calls
- Managed database operations
- Error handling
- Statistics tracking

**Response:**
```json
{
  "success": true,
  "workflow_result": {
    "workflow_id": "workflow_1234567890",
    "movie": { "id": "...", "name": "RRR" },
    "analysis": { "average_sentiment": 0.85 },
    "recommendations": [...]
  },
  "automation_stats": {
    "coordinator_stats": {
      "api_calls": {
        "total": 5,
        "successful": 5,
        "failed": 0,
        "success_rate": 100.0
      },
      "database_operations": {
        "total": 3,
        "successful": 3,
        "failed": 0,
        "success_rate": 100.0
      }
    },
    "trigger_history": 1
  }
}
```

### Automation Statistics

```http
GET /api/orchestrator/automation-stats
```

**Returns:**
- API call success rates
- Database operation success rates
- Error counts by type
- Recent trigger history
- Performance metrics

## Usage Examples

### Example 1: Automatic Workflow

```python
from app.agents import get_orchestrator

# Get orchestrator and automation manager
orchestrator = get_orchestrator()
automation_manager = orchestrator.get_automation_manager()

# Automatic workflow on user search
result = await automation_manager.handle_user_search(
    movie_name="Pushpa",
    collect_new_data=True,
    max_reviews=10
)

print(f"Success: {result['success']}")
print(f"Movie: {result['movie']['name']}")
print(f"Recommendations: {len(result['recommendations'])}")
```

### Example 2: Coordinated Data Collection

```python
# Coordinate TMDB and Twitter data collection
result = await automation_manager.coordinate_data_collection(
    movie_name="RRR",
    max_reviews=10
)

print(f"TMDB Results: {len(result['tmdb_data'])}")
print(f"Twitter Reviews: {len(result['twitter_data'])}")
print(f"Errors: {result['errors']}")
```

### Example 3: Database Coordination

```python
# Coordinate database storage with retries
result = await automation_manager.coordinate_database_operations(
    movie_data=movie_dict,
    reviews_data=reviews_list
)

print(f"Movie Stored: {result['movie_stored']}")
print(f"Reviews Stored: {result['reviews_stored']}")
```

### Example 4: Get Statistics

```python
# Get automation statistics
stats = automation_manager.get_automation_statistics()

api_stats = stats['coordinator_stats']['api_calls']
print(f"API Success Rate: {api_stats['success_rate']:.1f}%")

db_stats = stats['coordinator_stats']['database_operations']
print(f"DB Success Rate: {db_stats['success_rate']:.1f}%")
```

## Event System

### Registering Custom Triggers

You can register custom triggers for events:

```python
automation_manager = orchestrator.get_automation_manager()

# Define custom callback
async def on_recommendation_generated(event_data):
    movie_name = event_data['movie_name']
    recommendations = event_data['recommendations']
    # Custom logic here
    print(f"Generated {len(recommendations)} for {movie_name}")

# Register trigger
automation_manager.trigger.register_trigger(
    "recommendation_generated",
    on_recommendation_generated
)

# Trigger the event
await automation_manager.trigger.trigger_event(
    "recommendation_generated",
    {
        "movie_name": "RRR",
        "recommendations": [...]
    }
)
```

## Configuration

### Retry Configuration

Customize retry behavior:

```python
coordinator = automation_manager.coordinator

# Configure retries
coordinator.max_retries = 5  # More attempts
coordinator.retry_delay = 1.0  # Faster retries
coordinator.exponential_backoff = True  # Use backoff
```

### Rate Limit Configuration

Customize rate limiting:

```python
# Adjust for different Twitter tiers
coordinator.rate_limit_window = 60  # seconds
coordinator.max_calls_per_window = 15  # free tier

# For elevated tier:
# coordinator.max_calls_per_window = 50
```

## Testing

Run the automation test suite:

```bash
python tests/test_automation.py
```

**Tests Include:**
1. Automatic workflow triggering
2. API coordination with retries
3. Database coordination
4. Error handling
5. Statistics tracking

## Performance Metrics

The automation system tracks:

### API Calls
- Total calls made
- Success/failure counts
- Success rate percentage
- Error types and counts

### Database Operations
- Total operations
- Success/failure counts
- Success rate percentage
- Operation types

### Triggers
- Total triggers fired
- Recent trigger history
- Trigger outcomes

## Error Recovery Examples

### Example 1: API Failure Recovery

```
[API CALL] Attempt 1: FAILED (timeout)
[RETRY] Waiting 2 seconds...
[API CALL] Attempt 2: FAILED (timeout)
[RETRY] Waiting 4 seconds...
[API CALL] Attempt 3: SUCCESS
```

### Example 2: Rate Limit Handling

```
[API CALL] Twitter API: 15 calls in window
[RATE LIMIT] Limit reached, waiting 60 seconds...
[API CALL] Resuming after rate limit wait
[API CALL] Twitter API: SUCCESS
```

### Example 3: Database Retry

```
[DB] Store movie: FAILED (connection error)
[RETRY] Reconnecting... waiting 2 seconds
[DB] Store movie: FAILED (connection error)
[RETRY] Reconnecting... waiting 4 seconds
[DB] Store movie: SUCCESS
```

## Integration with Frontend

### Automatic Search Endpoint

```javascript
// Frontend code
async function searchMovie(movieName) {
  const response = await fetch(
    `/api/orchestrator/auto-search?movie_name=${movieName}&max_reviews=10`,
    { method: 'POST' }
  );
  
  const data = await response.json();
  
  // Display results
  console.log('Success:', data.success);
  console.log('Movie:', data.workflow_result.movie);
  console.log('Recommendations:', data.workflow_result.recommendations);
  
  // Show statistics
  console.log('API Calls:', data.automation_stats.coordinator_stats.api_calls);
}
```

## Benefits of Automation

1. **No Manual Intervention**
   - Workflows execute automatically
   - No need to manage individual steps
   - Reduces human error

2. **Intelligent Error Recovery**
   - Automatic retries on failures
   - Exponential backoff prevents overwhelming
   - Graceful degradation

3. **Performance Tracking**
   - Detailed statistics
   - Success rate monitoring
   - Error analysis

4. **Scalability**
   - Handles multiple concurrent workflows
   - Rate limit management
   - Resource optimization

5. **Maintainability**
   - Centralized error handling
   - Consistent retry logic
   - Easy to extend

## Future Enhancements

Potential improvements:

1. **Smart Scheduling**
   - Optimal times for data collection
   - Batch processing during off-peak
   - Priority queuing

2. **Adaptive Retry**
   - Learn from failure patterns
   - Adjust retry strategy dynamically
   - Predict success probability

3. **Cost Optimization**
   - Minimize API calls
   - Smart caching strategies
   - Bandwidth optimization

4. **Advanced Monitoring**
   - Real-time dashboards
   - Alert system for failures
   - Performance analytics

## Summary

Task 20 provides complete workflow automation:

✅ **Automatic Triggering**: Workflows start on user search
✅ **API Coordination**: Intelligent retry logic with exponential backoff
✅ **Database Management**: Automatic operations with error recovery
✅ **Error Handling**: Comprehensive handling at all levels
✅ **Statistics Tracking**: Detailed performance metrics
✅ **Rate Limiting**: Automatic rate limit management
✅ **Event System**: Flexible trigger-based architecture
✅ **Testing**: Complete test coverage
