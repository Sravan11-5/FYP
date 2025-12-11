# Agentic AI Orchestrator

## Overview

The Agentic AI Orchestrator is the intelligent automation layer that coordinates the entire movie recommendation workflow. It implements a multi-agent architecture where specialized agents work together autonomously to deliver personalized recommendations.

## Architecture

### Three Agent Roles

1. **Data Collector Agent** (`AgentRole.DATA_COLLECTOR`)
   - Fetches movie data from TMDB
   - Collects Telugu language reviews from Twitter
   - Makes autonomous decisions about:
     - Whether to use fresh or cached data
     - How to handle API rate limits
     - Which data sources to prioritize

2. **Analyzer Agent** (`AgentRole.ANALYZER`)
   - Processes reviews with the Siamese Network ML model
   - Performs batch sentiment analysis
   - Autonomously handles:
     - Missing or corrupted data
     - Batch processing strategies
     - Sentiment score aggregation

3. **Recommender Agent** (`AgentRole.RECOMMENDER`)
   - Generates personalized movie recommendations
   - Creates human-readable explanations
   - Makes decisions about:
     - Recommendation strategy (sentiment-based vs genre-based)
     - Number of recommendations to return
     - How to rank and score recommendations

### Tool Integrations

Each agent has access to the following tools:

- **TMDB Tool**: Movie search, details, similar movies
- **Twitter Tool**: Review collection with rate limiting
- **Database Tool**: Storage and retrieval operations
- **ML Model Tool**: Sentiment analysis with Siamese Network

## Workflow Execution

### Complete Workflow

```python
from app.agents import get_orchestrator

orchestrator = get_orchestrator()

result = await orchestrator.execute_workflow(
    movie_name="RRR",
    collect_new_data=True,
    max_reviews=10
)
```

**Workflow Steps:**

1. **Data Collection Phase**
   - Agent searches TMDB for movie
   - Gets detailed movie information
   - Stores movie in database
   - Collects Telugu reviews from Twitter
   - Stores reviews with duplicate prevention

2. **Analysis Phase**
   - Agent retrieves reviews from database
   - Processes reviews in batches
   - Analyzes sentiment with ML model
   - Calculates aggregate metrics
   - Stores sentiment distribution

3. **Recommendation Phase**
   - Agent determines recommendation strategy
   - Generates recommendations using the engine
   - Creates explanations for each recommendation
   - Returns ranked recommendations

### API Endpoints

#### Execute Complete Workflow
```http
POST /api/orchestrator/execute
Content-Type: application/json

{
  "movie_name": "RRR",
  "collect_new_data": true,
  "max_reviews": 10
}
```

**Response:**
```json
{
  "success": true,
  "workflow_id": "workflow_1234567890",
  "movie": {
    "id": "507f1f77bcf86cd799439011",
    "name": "RRR",
    "reviews_analyzed": 10
  },
  "analysis": {
    "reviews_analyzed": 10,
    "sentiment_distribution": {
      "positive": 8,
      "negative": 1,
      "neutral": 1,
      "positive_percentage": 80.0,
      "negative_percentage": 10.0,
      "neutral_percentage": 10.0
    },
    "average_sentiment": 0.82
  },
  "recommendations": [
    {
      "title": "Baahubali 2",
      "rating": 8.5,
      "recommendation_score": 92.0,
      "explanation": "Same genre: Action, Drama | Higher rating: 8.5/10 | Positive reviews in Telugu"
    }
  ]
}
```

#### Quick Recommend (Cached Data)
```http
POST /api/orchestrator/quick-recommend?movie_name=RRR
```

**Use Case:** Faster responses using only cached data, no fresh API calls.

#### Background Workflow
```http
POST /api/orchestrator/background-task
Content-Type: application/json

{
  "movie_name": "RRR",
  "collect_new_data": true,
  "max_reviews": 50
}
```

**Response:**
```json
{
  "success": true,
  "workflow_id": "workflow_1234567890",
  "message": "Workflow started in background",
  "status_endpoint": "/api/orchestrator/status/workflow_1234567890"
}
```

**Use Case:** For long-running workflows (many reviews), start in background and poll status.

#### Get Workflow Status
```http
GET /api/orchestrator/status/{workflow_id}
```

**Response:**
```json
{
  "workflow_id": "workflow_1234567890",
  "status": "completed",
  "started_at": "2024-01-01T12:00:00",
  "completed_at": "2024-01-01T12:01:30",
  "agents_executed": [
    {
      "agent": "data_collector",
      "status": "completed",
      "result": {...}
    },
    {
      "agent": "analyzer",
      "status": "completed",
      "result": {...}
    },
    {
      "agent": "recommender",
      "status": "completed",
      "result": {...}
    }
  ]
}
```

## Autonomous Decision Making

### Data Collection Decisions

The Data Collector Agent makes these autonomous decisions:

1. **Fresh vs Cached Data**
   - If `collect_new_data=True`: Fetch fresh reviews from Twitter
   - If `collect_new_data=False`: Use existing database reviews
   - Automatically falls back to cached if API fails

2. **Rate Limit Handling**
   - Respects Twitter free tier limits (15 requests per 15 minutes)
   - Automatically waits when limit reached
   - Continues with cached data if rate limited

3. **Data Quality**
   - Validates movie data from TMDB
   - Parses and cleans review text
   - Prevents duplicate storage

### Analysis Decisions

The Analyzer Agent autonomously handles:

1. **Batch Processing**
   - Processes all reviews efficiently
   - Handles missing/empty review text
   - Continues on individual review failures

2. **Sentiment Aggregation**
   - Calculates overall sentiment distribution
   - Computes average sentiment scores
   - Generates percentage breakdowns

### Recommendation Decisions

The Recommender Agent decides:

1. **Strategy Selection**
   - Sentiment-based: When reviews are available
   - Genre-based: When no reviews found
   - Hybrid: Combines both when possible

2. **Explanation Generation**
   - Chooses most relevant factors
   - Prioritizes genre matches
   - Includes rating comparisons
   - Mentions sentiment when applicable

3. **Ranking**
   - Combines multiple signals
   - Weights based on data quality
   - Returns top N recommendations

## Error Handling

The orchestrator includes comprehensive error handling:

- **Agent-level errors**: Each agent can fail independently
- **Tool errors**: API failures, rate limits, timeouts
- **Data errors**: Missing data, validation failures
- **Workflow recovery**: Partial results on agent failure

## Testing

Run the test suite:

```bash
python tests/test_orchestrator.py
```

Tests include:
- Complete workflow with fresh data
- Quick recommendations with cached data
- Different movies
- Workflow status tracking
- Error scenarios

## Performance Considerations

1. **Rate Limits**: Twitter free tier = 15 requests per 15 minutes
2. **Batch Processing**: Reviews processed in batches for efficiency
3. **Caching**: Database caching reduces API calls
4. **Background Tasks**: Long workflows can run in background

## Configuration

The orchestrator uses existing services:
- MongoDB for data storage
- Trained Siamese Network (checkpoints/best_model.pt)
- TMDB API for movie data
- Twitter API for reviews

## Usage Examples

### Example 1: New Movie Analysis
```python
orchestrator = get_orchestrator()

result = await orchestrator.execute_workflow(
    movie_name="Pushpa",
    collect_new_data=True,
    max_reviews=10
)

print(f"Analyzed {result['movie']['reviews_analyzed']} reviews")
print(f"Sentiment: {result['analysis']['average_sentiment']:.2f}")
print(f"Top recommendation: {result['recommendations'][0]['title']}")
```

### Example 2: Quick Cached Lookup
```python
result = await orchestrator.execute_workflow(
    movie_name="RRR",
    collect_new_data=False,
    max_reviews=0
)

# Fast response using only database
```

### Example 3: High-Volume Collection
```python
# Start in background
workflow_id = "workflow_123"

# ... start background task ...

# Poll for status
status = await orchestrator.get_workflow_status(workflow_id)
print(f"Status: {status['status']}")
```

## Integration with Frontend

The orchestrator endpoints are designed for easy frontend integration:

```javascript
// Execute workflow
const response = await fetch('/api/orchestrator/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    movie_name: 'RRR',
    collect_new_data: true,
    max_reviews: 10
  })
});

const result = await response.json();
console.log('Recommendations:', result.recommendations);
```

## Future Enhancements

Potential improvements for the orchestrator:

1. **Multi-language support**: Extend beyond Telugu
2. **User preferences**: Personalized agent behavior
3. **Learning from feedback**: Improve recommendations over time
4. **Advanced scheduling**: Optimal times for data collection
5. **Cost optimization**: Minimize API usage while maximizing quality

## Summary

The Agentic AI Orchestrator provides:
- ✅ Autonomous end-to-end workflow automation
- ✅ Three specialized agent roles
- ✅ Intelligent decision making at each step
- ✅ Comprehensive error handling
- ✅ Flexible API endpoints for integration
- ✅ Background task support
- ✅ Workflow status tracking
