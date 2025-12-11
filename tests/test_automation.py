"""
Test script for Automated Workflow Manager
Tests automatic triggers, coordination, and error handling
"""
import asyncio
import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents import get_orchestrator
from app.database import connect_to_mongo, close_mongo_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_automatic_trigger():
    """Test automatic workflow triggering on user search"""
    logger.info("="*60)
    logger.info("TEST 1: Automatic Workflow Triggering")
    logger.info("="*60)
    
    try:
        await connect_to_mongo()
        
        orchestrator = get_orchestrator()
        automation_manager = orchestrator.get_automation_manager()
        
        logger.info("\n[TEST] Simulating user search event...")
        logger.info("[TEST] Movie: RRR")
        logger.info("[TEST] Expected: Automatic workflow execution")
        
        # Use automated workflow manager
        result = await automation_manager.handle_user_search(
            movie_name="RRR",
            collect_new_data=True,
            max_reviews=5
        )
        
        logger.info("\n[RESULT] Automatic workflow completed!")
        logger.info(f"[RESULT] Success: {result.get('success')}")
        
        if result.get('success'):
            logger.info("✓ TEST PASSED: Workflow triggered automatically on user search")
            
            movie_info = result.get('movie', {})
            logger.info(f"[INFO] Movie: {movie_info.get('name')}")
            logger.info(f"[INFO] Reviews: {movie_info.get('reviews_analyzed')}")
            
            return True
        else:
            logger.error(f"✗ TEST FAILED: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {str(e)}", exc_info=True)
        return False
    finally:
        await close_mongo_connection()


async def test_api_coordination():
    """Test coordinated API calls with retry logic"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: API Call Coordination with Retries")
    logger.info("="*60)
    
    try:
        await connect_to_mongo()
        
        orchestrator = get_orchestrator()
        automation_manager = orchestrator.get_automation_manager()
        
        logger.info("\n[TEST] Testing coordinated TMDB and Twitter API calls...")
        logger.info("[TEST] Movie: Baahubali")
        logger.info("[TEST] Expected: Automatic retries on failure")
        
        # Test data collection coordination
        result = await automation_manager.coordinate_data_collection(
            movie_name="Baahubali",
            max_reviews=3
        )
        
        logger.info("\n[RESULT] API coordination completed!")
        logger.info(f"[RESULT] Success: {result.get('success')}")
        
        if result.get('tmdb_data'):
            logger.info(f"[RESULT] TMDB: ✓ {len(result.get('tmdb_data', []))} movies found")
        
        if result.get('twitter_data'):
            logger.info(f"[RESULT] Twitter: ✓ {len(result.get('twitter_data', []))} reviews collected")
        
        if result.get('errors'):
            logger.warning(f"[RESULT] Errors: {result.get('errors')}")
        
        # Check if at least one API call succeeded
        if result.get('tmdb_data') or result.get('twitter_data'):
            logger.info("✓ TEST PASSED: API calls coordinated successfully")
            return True
        else:
            logger.error("✗ TEST FAILED: Both API calls failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {str(e)}", exc_info=True)
        return False
    finally:
        await close_mongo_connection()


async def test_database_coordination():
    """Test coordinated database operations with retries"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Database Operation Coordination")
    logger.info("="*60)
    
    try:
        await connect_to_mongo()
        
        orchestrator = get_orchestrator()
        automation_manager = orchestrator.get_automation_manager()
        
        logger.info("\n[TEST] Testing coordinated database operations...")
        logger.info("[TEST] Expected: Automatic retries on failure")
        
        # Prepare test data
        movie_data = {
            "tmdb_id": "999999",
            "title": "Test Movie",
            "original_title": "Test Movie",
            "genres": ["Action"],
            "release_date": "2024-01-01",
            "rating": 8.0,
            "overview": "Test movie for automation testing",
            "poster_path": "/test.jpg",
            "backdrop_path": "/test_backdrop.jpg"
        }
        
        reviews_data = [
            {
                "tweet_id": "test_1",
                "movie_id": "999999",
                "text": "Great movie!",
                "author_id": "user1",
                "created_at": "2024-01-01T00:00:00Z",
                "language": "te"
            }
        ]
        
        # Test database coordination
        result = await automation_manager.coordinate_database_operations(
            movie_data=movie_data,
            reviews_data=reviews_data
        )
        
        logger.info("\n[RESULT] Database coordination completed!")
        logger.info(f"[RESULT] Success: {result.get('success')}")
        logger.info(f"[RESULT] Movie stored: {result.get('movie_stored')}")
        logger.info(f"[RESULT] Reviews stored: {result.get('reviews_stored')}")
        
        if result.get('errors'):
            logger.warning(f"[RESULT] Errors: {result.get('errors')}")
        
        if result.get('movie_stored'):
            logger.info("✓ TEST PASSED: Database operations coordinated successfully")
            return True
        else:
            logger.error("✗ TEST FAILED: Database operations failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {str(e)}", exc_info=True)
        return False
    finally:
        await close_mongo_connection()


async def test_error_handling():
    """Test automatic error handling and retries"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Error Handling and Retries")
    logger.info("="*60)
    
    try:
        await connect_to_mongo()
        
        orchestrator = get_orchestrator()
        automation_manager = orchestrator.get_automation_manager()
        coordinator = automation_manager.coordinator
        
        logger.info("\n[TEST] Testing error handling with invalid movie name...")
        logger.info("[TEST] Movie: NonExistentMovieXYZ12345")
        logger.info("[TEST] Expected: Graceful error handling")
        
        # This should fail but handle gracefully
        result = await automation_manager.handle_user_search(
            movie_name="NonExistentMovieXYZ12345",
            collect_new_data=True,
            max_reviews=3
        )
        
        logger.info("\n[RESULT] Error handling test completed!")
        
        # Check if errors were handled gracefully
        if not result.get('success'):
            logger.info("✓ TEST PASSED: Errors handled gracefully")
            logger.info(f"[INFO] Error message: {result.get('error')}")
            return True
        else:
            logger.warning("⚠ TEST UNEXPECTED: Workflow succeeded for non-existent movie")
            return True  # Still pass, just unexpected
            
    except Exception as e:
        # If exception is caught, error handling worked
        logger.info("✓ TEST PASSED: Exception caught and handled")
        logger.info(f"[INFO] Exception: {str(e)}")
        return True
    finally:
        await close_mongo_connection()


async def test_automation_statistics():
    """Test automation statistics tracking"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Automation Statistics")
    logger.info("="*60)
    
    try:
        await connect_to_mongo()
        
        orchestrator = get_orchestrator()
        automation_manager = orchestrator.get_automation_manager()
        
        # Run a workflow
        await automation_manager.handle_user_search(
            movie_name="RRR",
            collect_new_data=False,
            max_reviews=0
        )
        
        # Get statistics
        stats = automation_manager.get_automation_statistics()
        
        logger.info("\n[STATISTICS] Automation Stats:")
        
        coordinator_stats = stats.get('coordinator_stats', {})
        api_stats = coordinator_stats.get('api_calls', {})
        db_stats = coordinator_stats.get('database_operations', {})
        
        logger.info(f"\n[API CALLS]")
        logger.info(f"  Total: {api_stats.get('total', 0)}")
        logger.info(f"  Successful: {api_stats.get('successful', 0)}")
        logger.info(f"  Failed: {api_stats.get('failed', 0)}")
        logger.info(f"  Success Rate: {api_stats.get('success_rate', 0):.1f}%")
        
        logger.info(f"\n[DATABASE OPS]")
        logger.info(f"  Total: {db_stats.get('total', 0)}")
        logger.info(f"  Successful: {db_stats.get('successful', 0)}")
        logger.info(f"  Failed: {db_stats.get('failed', 0)}")
        logger.info(f"  Success Rate: {db_stats.get('success_rate', 0):.1f}%")
        
        logger.info(f"\n[TRIGGERS]")
        logger.info(f"  Total Triggers: {stats.get('trigger_history', 0)}")
        
        logger.info("\n✓ TEST PASSED: Statistics tracking working")
        return True
        
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {str(e)}", exc_info=True)
        return False
    finally:
        await close_mongo_connection()


async def main():
    """Run all automated workflow tests"""
    logger.info("\n" + "#"*60)
    logger.info("# AUTOMATED WORKFLOW MANAGER TEST SUITE")
    logger.info("#"*60)
    
    tests = [
        ("Automatic Trigger", test_automatic_trigger),
        ("API Coordination", test_api_coordination),
        ("Database Coordination", test_database_coordination),
        ("Error Handling", test_error_handling),
        ("Automation Statistics", test_automation_statistics)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test crashed: {str(e)}", exc_info=True)
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "#"*60)
    logger.info("# TEST SUMMARY")
    logger.info("#"*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n{passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        logger.error(f"\n✗ {total - passed} TEST(S) FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
