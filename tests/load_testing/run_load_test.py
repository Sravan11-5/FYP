"""
Load Testing Runner
Automated script to run load tests and generate reports
"""
import subprocess
import sys
import time
from datetime import datetime
import json
import os

def run_load_test(
    host: str = "http://localhost:8000",
    users: int = 100,
    spawn_rate: int = 10,
    duration: str = "5m",
    output_dir: str = "load_test_results"
):
    """
    Run load test with Locust.
    
    Args:
        host: Target host URL
        users: Number of concurrent users to simulate
        spawn_rate: Users to spawn per second
        duration: Test duration (e.g., '5m' for 5 minutes)
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_prefix = f"{output_dir}/load_test_{timestamp}"
    
    print(f"🚀 Starting load test...")
    print(f"   Target: {host}")
    print(f"   Users: {users}")
    print(f"   Spawn rate: {spawn_rate}/s")
    print(f"   Duration: {duration}")
    print(f"   Report prefix: {report_prefix}\n")
    
    # Build Locust command
    cmd = [
        "locust",
        "-f", "locustfile.py",
        "--host", host,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", duration,
        "--headless",
        "--html", f"{report_prefix}.html",
        "--csv", report_prefix,
        "--loglevel", "INFO"
    ]
    
    try:
        # Run Locust
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration_seconds = time.time() - start_time
        
        print("\n✅ Load test completed successfully!")
        print(f"   Duration: {duration_seconds:.2f} seconds")
        print(f"\n📊 Reports generated:")
        print(f"   HTML Report: {report_prefix}.html")
        print(f"   CSV Stats: {report_prefix}_stats.csv")
        print(f"   CSV History: {report_prefix}_stats_history.csv")
        print(f"   CSV Failures: {report_prefix}_failures.csv")
        
        # Parse and display summary from CSV
        display_summary(report_prefix)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Load test failed!")
        print(f"   Error: {e}")
        print(f"   Output: {e.output}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Locust not found!")
        print("   Please install: pip install locust")
        sys.exit(1)


def display_summary(report_prefix: str):
    """Display summary from CSV stats"""
    try:
        stats_file = f"{report_prefix}_stats.csv"
        
        if not os.path.exists(stats_file):
            return
        
        print("\n📈 Performance Summary:")
        print("-" * 80)
        
        with open(stats_file, 'r') as f:
            lines = f.readlines()
            
            # Skip header and aggregated row
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 10 and parts[1] != "Aggregated":
                    method = parts[0]
                    name = parts[1]
                    requests = parts[2]
                    failures = parts[3]
                    median = parts[4]
                    avg = parts[5]
                    min_time = parts[6]
                    max_time = parts[7]
                    
                    print(f"\n{method} {name}")
                    print(f"  Requests: {requests}")
                    print(f"  Failures: {failures}")
                    print(f"  Response times: min={min_time}ms, avg={avg}ms, max={max_time}ms")
        
        print("-" * 80)
        
    except Exception as e:
        print(f"   Could not parse summary: {e}")


def run_stress_test():
    """Run stress test with increasing load"""
    print("\n🔥 STRESS TEST MODE")
    print("=" * 80)
    
    # Gradually increase load
    test_configs = [
        {"users": 10, "duration": "2m", "name": "Warm-up"},
        {"users": 50, "duration": "3m", "name": "Normal Load"},
        {"users": 100, "duration": "5m", "name": "High Load"},
        {"users": 200, "duration": "3m", "name": "Stress Load"},
        {"users": 300, "duration": "2m", "name": "Peak Load"}
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🎯 Test {i}/{len(test_configs)}: {config['name']}")
        print(f"   Running with {config['users']} users for {config['duration']}...")
        
        run_load_test(
            users=config['users'],
            spawn_rate=10,
            duration=config['duration'],
            output_dir=f"load_test_results/stress_test_{config['name'].replace(' ', '_').lower()}"
        )
        
        # Cool down between tests
        if i < len(test_configs):
            print("\n   ⏸️  Cooling down for 30 seconds...")
            time.sleep(30)
    
    print("\n✅ Stress test completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run load tests on the Movie Recommendation System")
    parser.add_argument("--host", default="http://localhost:8000", help="Target host URL")
    parser.add_argument("--users", type=int, default=100, help="Number of concurrent users")
    parser.add_argument("--spawn-rate", type=int, default=10, help="Users to spawn per second")
    parser.add_argument("--duration", default="5m", help="Test duration (e.g., 5m, 10s)")
    parser.add_argument("--stress", action="store_true", help="Run stress test with increasing load")
    
    args = parser.parse_args()
    
    if args.stress:
        run_stress_test()
    else:
        run_load_test(
            host=args.host,
            users=args.users,
            spawn_rate=args.spawn_rate,
            duration=args.duration
        )
