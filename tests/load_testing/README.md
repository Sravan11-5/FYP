# Load Testing Guide

This directory contains load testing scripts for the Telugu Movie Recommendation System.

## Prerequisites

Install Locust:
```bash
pip install locust
```

## Running Load Tests

### Quick Start

1. **Start the application** (in a separate terminal):
```bash
python -m uvicorn main:app --reload
```

2. **Run basic load test** (100 users for 5 minutes):
```bash
cd tests/load_testing
python run_load_test.py
```

### Custom Load Test

Customize the test parameters:
```bash
python run_load_test.py --users 200 --spawn-rate 20 --duration 10m
```

Parameters:
- `--users`: Number of concurrent users (default: 100)
- `--spawn-rate`: Users to spawn per second (default: 10)
- `--duration`: Test duration, e.g., `5m` (5 minutes), `30s` (30 seconds)
- `--host`: Target URL (default: http://localhost:8000)

### Stress Test

Run automated stress test with increasing load:
```bash
python run_load_test.py --stress
```

This will run 5 consecutive tests with increasing user counts:
1. Warm-up: 10 users (2 min)
2. Normal Load: 50 users (3 min)
3. High Load: 100 users (5 min)
4. Stress Load: 200 users (3 min)
5. Peak Load: 300 users (2 min)

### Interactive Mode

Run Locust with web UI:
```bash
locust -f locustfile.py --host http://localhost:8000
```

Then open http://localhost:8089 in your browser.

## Test Scenarios

The load tests simulate realistic user behavior:

### Regular Users (MovieRecommendationUser)
- **Search Movie** (50% of actions): Search for Telugu movies
- **Get Recommendations** (30%): Get recommendations for specific movies
- **Auto Search** (20%): Use orchestrator for automated search
- **Submit Feedback** (10%): Rate movies
- **Health Checks** (20%): Check system status
- **Cache Stats** (10%): View cache performance

### Admin Users (AdminUser)
- View performance statistics
- View feedback analytics
- Lower frequency monitoring

## Interpreting Results

### Key Metrics

1. **Requests per Second (RPS)**: Higher is better
2. **Response Time**:
   - Median: 50th percentile response time
   - Average: Mean response time
   - 95th percentile: 95% of requests complete within this time
3. **Failure Rate**: Should be < 1%
4. **Concurrent Users**: Number of active users

### Performance Targets

Based on Task 27 requirements:
- **Cached results**: 5-10 seconds
- **New searches**: 30-60 seconds
- **System load**: Handle 100+ concurrent users
- **Error rate**: < 1%

### Report Files

After each test, find results in `load_test_results/`:
- `load_test_TIMESTAMP.html`: Interactive HTML report
- `load_test_TIMESTAMP_stats.csv`: Endpoint statistics
- `load_test_TIMESTAMP_stats_history.csv`: Time-series data
- `load_test_TIMESTAMP_failures.csv`: Failed requests log

## Monitoring During Tests

While tests are running, monitor:

1. **System Health**:
```bash
curl http://localhost:8000/api/system/health
```

2. **Performance Stats**:
```bash
curl http://localhost:8000/api/system/performance/stats
```

3. **Cache Performance**:
```bash
curl http://localhost:8000/api/system/cache/stats
```

4. **System Resources**:
   - CPU usage
   - Memory usage
   - Database connections
   - Network I/O

## Troubleshooting

### High Failure Rate

If failure rate > 5%:
1. Check application logs: `app.log`
2. Verify database connection
3. Check MongoDB performance
4. Review error messages in failures CSV

### Slow Response Times

If response times exceed targets:
1. Check cache hit rate: `/api/system/cache/stats`
2. Review database indexes
3. Monitor MongoDB slow queries
4. Check network latency

### Connection Errors

If getting connection refused:
1. Ensure application is running
2. Verify port (default: 8000)
3. Check firewall settings

## Best Practices

1. **Warm-up**: Start with low user count to warm up caches
2. **Gradual Increase**: Increase load gradually to avoid overwhelming the system
3. **Realistic Patterns**: Test scenarios match actual user behavior
4. **Monitor Resources**: Watch CPU, memory, and database during tests
5. **Test Isolation**: Run tests on dedicated test environment when possible
6. **Baseline Metrics**: Establish baseline before making changes

## Example Workflow

```bash
# 1. Start the application
python -m uvicorn main:app

# 2. Run quick test (separate terminal)
cd tests/load_testing
python run_load_test.py --users 50 --duration 3m

# 3. Review results
# Open generated HTML report

# 4. Run full stress test
python run_load_test.py --stress

# 5. Analyze and optimize
# Review bottlenecks from reports
# Make optimizations
# Rerun tests to verify improvements
```

## Advanced Configuration

### Custom User Behavior

Edit `locustfile.py` to:
- Add new endpoints
- Change task weights
- Modify wait times
- Add custom test data

### Distributed Load Testing

Run tests from multiple machines:

**Master node**:
```bash
locust -f locustfile.py --master --host http://target-host:8000
```

**Worker nodes**:
```bash
locust -f locustfile.py --worker --master-host <master-ip>
```

## Task 29 Verification

To complete Task 29:

1. ✅ Run load test with 100+ users
2. ✅ Monitor performance metrics
3. ✅ Identify bottlenecks from reports
4. ✅ Verify system handles load
5. ✅ Response times within targets
6. ✅ Error rate < 1%
