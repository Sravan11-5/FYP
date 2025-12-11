"""
Test API Endpoints - Task 5 Verification
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test GET /api/health endpoint"""
    print("\n" + "="*60)
    print("TESTING: GET /api/health")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check endpoint working!")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Timestamp: {data.get('timestamp')}")
            print("\n   Services Status:")
            for service, status in data.get('services', {}).items():
                icon = "✅" if status == "healthy" else "⚠️"
                print(f"      {icon} {service}: {status}")
            return True
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_search_endpoint():
    """Test POST /api/search endpoint"""
    print("\n" + "="*60)
    print("TESTING: POST /api/search")
    print("="*60)
    
    test_cases = [
        {"movie_name": "Baahubali", "language": "te"},
        {"movie_name": "RRR", "language": "te"},
        {"movie_name": "Pushpa", "language": "te"}
    ]
    
    success_count = 0
    
    for test_case in test_cases:
        print(f"\n📝 Testing search: {test_case['movie_name']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/search",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Search successful!")
                print(f"      Success: {data.get('success')}")
                print(f"      Message: {data.get('message')}")
                print(f"      Total Results: {data.get('total_results')}")
                
                if data.get('movies'):
                    print(f"\n      First Result:")
                    movie = data['movies'][0]
                    print(f"         Title: {movie.get('title')}")
                    print(f"         TMDB ID: {movie.get('tmdb_id')}")
                    print(f"         Release: {movie.get('release_date')}")
                    print(f"         Rating: {movie.get('rating')}")
                
                success_count += 1
            else:
                print(f"   ❌ Search failed with status: {response.status_code}")
                print(f"      Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    
    return success_count == len(test_cases)


def test_input_validation():
    """Test input validation for search endpoint"""
    print("\n" + "="*60)
    print("TESTING: Input Validation")
    print("="*60)
    
    invalid_cases = [
        {"movie_name": ""},  # Empty string
        {"movie_name": "   "},  # Only whitespace
        {},  # Missing movie_name
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(invalid_cases, 1):
        print(f"\n📝 Testing invalid case {i}: {test_case}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/search",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"   Status Code: {response.status_code}")
            
            # Expecting 422 (Validation Error) or 400 (Bad Request)
            if response.status_code in [400, 422]:
                print(f"   ✅ Validation working correctly - rejected invalid input")
                success_count += 1
            else:
                print(f"   ⚠️  Expected validation error but got: {response.status_code}")
                print(f"      Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
    
    return success_count == len(invalid_cases)


def test_root_endpoint():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("TESTING: GET /")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/")
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Root endpoint working!")
            print(f"   Message: {data.get('message')}")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            return True
        else:
            print(f"❌ Root endpoint failed")
            return False
            
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🎬 TASK 5: API ENDPOINTS TESTING")
    print("="*60)
    print("\nMake sure the server is running: python main.py")
    print("Testing against:", BASE_URL)
    
    # Check if server is running
    try:
        response = requests.get(BASE_URL, timeout=2)
    except requests.exceptions.RequestException:
        print("\n❌ ERROR: Server is not running!")
        print("Please start the server first: python main.py")
        return
    
    results = {}
    
    # Run tests
    results['root'] = test_root_endpoint()
    results['health'] = test_health_endpoint()
    results['search'] = test_search_endpoint()
    results['validation'] = test_input_validation()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name.upper()}: {'PASSED' if passed else 'FAILED'}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nTask 5 Subtasks Completed:")
        print("   ✅ 5.1 - POST /api/search endpoint defined")
        print("   ✅ 5.2 - GET /api/health endpoint implemented")
        print("   ✅ 5.3 - FastAPI dependency injection working")
        print("   ✅ 5.4 - Input validation implemented")
    else:
        print("\n⚠️  SOME TESTS FAILED")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
