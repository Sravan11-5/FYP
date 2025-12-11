"""
Load Testing with Locust
Simulates concurrent users accessing the Telugu Movie Recommendation System
"""
from locust import HttpUser, task, between, events
import random
import json
import time

# Test data - sample movie queries and IDs
SAMPLE_QUERIES = [
    "Baahubali",
    "RRR",
    "Pushpa",
    "Eega",
    "Arjun Reddy",
    "Ala Vaikunthapurramuloo",
    "Sarileru Neekevvaru",
    "Sye Raa Narasimha Reddy",
    "Maharshi",
    "Rangasthalam"
]

SAMPLE_MOVIE_IDS = [
    100809,  # RRR
    575982,  # Pushpa
    140300,  # Eega
    297762,  # Wonder Woman
    335984,  # Blade Runner 2049
]


class MovieRecommendationUser(HttpUser):
    """
    Simulates a user interacting with the movie recommendation system.
    Performs realistic user actions with variable wait times.
    """
    
    # Wait between 1-5 seconds between tasks (realistic user behavior)
    wait_time = between(1, 5)
    
    def on_start(self):
        """Initialize user session"""
        self.user_id = f"user_{random.randint(1000, 9999)}"
        self.session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"
        self.client.headers['User-Agent'] = 'LoadTestUser/1.0'
    
    @task(5)
    def search_movie(self):
        """
        Search for a movie (most common action - weight: 5)
        """
        query = random.choice(SAMPLE_QUERIES)
        
        with self.client.post(
            "/api/search/movie",
            json={"query": query},
            catch_response=True,
            name="/api/search/movie"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    response.success()
                else:
                    response.failure(f"Search failed: {data.get('message')}")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(3)
    def get_recommendations_manual(self):
        """
        Get recommendations manually (weight: 3)
        """
        movie_id = random.choice(SAMPLE_MOVIE_IDS)
        
        with self.client.post(
            "/api/recommendations/get",
            json={"tmdb_id": movie_id},
            catch_response=True,
            name="/api/recommendations/get"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    response.success()
                else:
                    response.failure(f"Recommendations failed: {data.get('message')}")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(2)
    def auto_search_recommendations(self):
        """
        Auto search with orchestrator (weight: 2)
        """
        query = random.choice(SAMPLE_QUERIES)
        
        with self.client.post(
            "/api/orchestrator/auto-search",
            json={
                "query": query,
                "user_id": self.user_id,
                "session_id": self.session_id
            },
            catch_response=True,
            name="/api/orchestrator/auto-search"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'processing':
                    task_id = data.get('task_id')
                    # Poll for results
                    self.poll_task_status(task_id)
                    response.success()
                else:
                    response.failure(f"Auto-search failed: {data.get('message')}")
            else:
                response.failure(f"Status: {response.status_code}")
    
    def poll_task_status(self, task_id: str, max_attempts: int = 5):
        """Poll for task completion"""
        for attempt in range(max_attempts):
            time.sleep(2)  # Wait 2 seconds between polls
            
            with self.client.get(
                f"/api/orchestrator/task/{task_id}",
                catch_response=True,
                name="/api/orchestrator/task/status"
            ) as response:
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status')
                    
                    if status == 'completed':
                        response.success()
                        return
                    elif status == 'failed':
                        response.failure("Task failed")
                        return
                else:
                    response.failure(f"Status check failed: {response.status_code}")
                    return
    
    @task(1)
    def submit_feedback(self):
        """
        Submit feedback on a movie (least common - weight: 1)
        """
        movie_id = random.choice(SAMPLE_MOVIE_IDS)
        rating = random.randint(1, 5)
        
        with self.client.post(
            "/api/feedback/submit",
            json={
                "user_id": self.user_id,
                "session_id": self.session_id,
                "movie_id": movie_id,
                "rating": rating,
                "comment": f"Load test feedback {rating} stars"
            },
            catch_response=True,
            name="/api/feedback/submit"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(2)
    def check_health(self):
        """
        Check system health (weight: 2)
        """
        with self.client.get(
            "/api/system/health",
            catch_response=True,
            name="/api/system/health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(1)
    def get_cache_stats(self):
        """
        Get cache statistics (weight: 1)
        """
        with self.client.get(
            "/api/system/cache/stats",
            catch_response=True,
            name="/api/system/cache/stats"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


class AdminUser(HttpUser):
    """
    Simulates admin users monitoring the system.
    Lower frequency, different endpoints.
    """
    
    wait_time = between(5, 15)
    
    @task
    def get_performance_stats(self):
        """Get performance statistics"""
        with self.client.get(
            "/api/system/performance/stats",
            catch_response=True,
            name="/api/system/performance/stats [Admin]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task
    def get_feedback_stats(self):
        """Get overall feedback statistics"""
        with self.client.get(
            "/api/feedback/stats/overall",
            catch_response=True,
            name="/api/feedback/stats/overall [Admin]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


# Event handlers for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the load test starts"""
    print("Load test starting...")
    print(f"Target host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the load test stops"""
    print("\nLoad test completed!")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"Max response time: {environment.stats.total.max_response_time:.2f}ms")
