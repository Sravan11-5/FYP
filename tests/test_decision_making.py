"""
Test script for Autonomous Decision Making
Tests intelligent decision-making, task prioritization, and failure handling
"""
import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents import get_orchestrator, get_decision_maker, TaskPriority, DataFreshnessPolicy
from app.agents.decision_maker import TaskDefinition
from app.database import connect_to_mongo, close_mongo_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_data_fetching_decisions():
    """Test autonomous decisions about fetching new vs cached data"""
    logger.info("="*60)
    logger.info("TEST 1: Data Fetching Decisions")
    logger.info("="*60)
    
    try:
        await connect_to_mongo()
        
        orchestrator = get_orchestrator()
        decision_maker = orchestrator.get_decision_maker()
        
        # Test 1a: No cached data
        logger.info("\n[TEST 1a] Decision when no cached data exists")
        decision = await decision_maker.decide_data_strategy(
            data_type="movie",
            identifier="NonExistentMovie123",
            user_preference=None
        )
        
        logger.info(f"[RESULT] Action: {decision['action']}")
        logger.info(f"[RESULT] Reasoning: {'; '.join(decision['reasoning'])}")
        logger.info(f"[RESULT] Confidence: {decision['confidence']:.2f}")
        
        assert decision['action'] == 'fetch_new', "Should fetch new when no cache"
        logger.info("✓ TEST 1a PASSED")
        
        # Test 1b: User explicitly requests fresh data
        logger.info("\n[TEST 1b] Decision when user requests fresh data")
        decision = await decision_maker.decide_data_strategy(
            data_type="movie",
            identifier="RRR",
            user_preference="fresh"
        )
        
        logger.info(f"[RESULT] Action: {decision['action']}")
        logger.info(f"[RESULT] Confidence: {decision['confidence']:.2f}")
        
        assert decision['action'] == 'fetch_new', "Should respect user preference"
        assert decision['confidence'] == 1.0, "User preference should have max confidence"
        logger.info("✓ TEST 1b PASSED")
        
        # Test 1c: User explicitly requests cached data
        logger.info("\n[TEST 1c] Decision when user requests cached data")
        decision = await decision_maker.decide_data_strategy(
            data_type="movie",
            identifier="RRR",
            user_preference="cached"
        )
        
        logger.info(f"[RESULT] Action: {decision['action']}")
        logger.info(f"[RESULT] Confidence: {decision['confidence']:.2f}")
        
        assert decision['action'] == 'use_cached', "Should respect user preference"
        logger.info("✓ TEST 1c PASSED")
        
        # Test 1d: Different freshness policies
        logger.info("\n[TEST 1d] Testing different freshness policies")
        
        policies = [
            (DataFreshnessPolicy.ALWAYS_FRESH, "fetch_new"),
            (DataFreshnessPolicy.PREFER_CACHED, "use_cached"),
            (DataFreshnessPolicy.SMART, None)  # Depends on data state
        ]
        
        for policy, expected_action in policies:
            decision_maker.configure(policy=policy)
            decision = await decision_maker.decide_data_strategy(
                data_type="movie",
                identifier="TestMovie",
                user_preference=None
            )
            
            logger.info(f"[RESULT] Policy: {policy.value}, Action: {decision['action']}")
            
            if expected_action:
                assert decision['action'] == expected_action, f"Policy {policy.value} should result in {expected_action}"
        
        logger.info("✓ TEST 1d PASSED")
        
        logger.info("\n✓ ALL DATA FETCHING DECISION TESTS PASSED")
        return True
        
    except AssertionError as e:
        logger.error(f"✗ TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {str(e)}", exc_info=True)
        return False
    finally:
        await close_mongo_connection()


async def test_task_prioritization():
    """Test autonomous task prioritization"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Task Prioritization")
    logger.info("="*60)
    
    try:
        await connect_to_mongo()
        
        orchestrator = get_orchestrator()
        decision_maker = orchestrator.get_decision_maker()
        
        logger.info("\n[TEST] Creating prioritized task list")
        
        # Create tasks
        tasks = decision_maker.prioritize_tasks(
            user_movie="RRR",
            similar_movies=["Baahubali", "KGF", "Pushpa"],
            additional_movies=["Ala Vaikunthapurramuloo", "Sye Raa"]
        )
        
        logger.info(f"\n[RESULT] Created {len(tasks)} tasks")
        
        # Verify prioritization
        logger.info("\n[VERIFICATION] Task order:")
        for i, task in enumerate(tasks[:10], 1):  # Show first 10
            logger.info(f"  {i}. {task.task_type} (Priority: {task.priority.name})")
        
        # Verify critical tasks are first
        critical_tasks = [t for t in tasks if t.priority == TaskPriority.CRITICAL]
        high_tasks = [t for t in tasks if t.priority == TaskPriority.HIGH]
        
        logger.info(f"\n[STATS]")
        logger.info(f"  Critical: {len(critical_tasks)}")
        logger.info(f"  High: {len(high_tasks)}")
        logger.info(f"  Total: {len(tasks)}")
        
        # Verify first task is fetching user movie
        assert tasks[0].task_type == "fetch_user_movie", "First task should be user movie"
        assert tasks[0].priority == TaskPriority.CRITICAL, "User movie should be critical"
        
        # Verify second task is fetching user reviews
        assert tasks[1].task_type == "fetch_user_reviews", "Second task should be user reviews"
        assert tasks[1].priority == TaskPriority.CRITICAL, "User reviews should be critical"
        
        # Verify critical tasks come before high tasks
        last_critical_idx = max(i for i, t in enumerate(tasks) if t.priority == TaskPriority.CRITICAL)
        first_medium_idx = next((i for i, t in enumerate(tasks) if t.priority == TaskPriority.MEDIUM), len(tasks))
        
        if first_medium_idx < len(tasks):
            assert last_critical_idx < first_medium_idx, "Critical tasks should come before medium tasks"
        
        logger.info("\n✓ TASK PRIORITIZATION TEST PASSED")
        return True
        
    except AssertionError as e:
        logger.error(f"✗ TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {str(e)}", exc_info=True)
        return False
    finally:
        await close_mongo_connection()


async def test_failure_handling():
    """Test autonomous failure handling and retry logic"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Failure Handling and Retries")
    logger.info("="*60)
    
    try:
        await connect_to_mongo()
        
        orchestrator = get_orchestrator()
        decision_maker = orchestrator.get_decision_maker()
        
        # Test 3a: Transient error - should retry
        logger.info("\n[TEST 3a] Transient error handling")
        task = TaskDefinition(
            task_id="test_task_1",
            task_type="test_api_call",
            priority=TaskPriority.HIGH,
            params={"test": True}
        )
        
        should_retry = await decision_maker.handle_failure(
            task,
            "Connection timeout - network unavailable"
        )
        
        logger.info(f"[RESULT] Should retry: {should_retry}")
        assert should_retry == True, "Should retry on transient error"
        logger.info("✓ TEST 3a PASSED")
        
        # Test 3b: Permanent error - should not retry
        logger.info("\n[TEST 3b] Permanent error handling")
        task = TaskDefinition(
            task_id="test_task_2",
            task_type="test_api_call",
            priority=TaskPriority.HIGH,
            params={"test": True}
        )
        
        should_retry = await decision_maker.handle_failure(
            task,
            "404 Not Found - resource does not exist"
        )
        
        logger.info(f"[RESULT] Should retry: {should_retry}")
        assert should_retry == False, "Should not retry on permanent error"
        logger.info("✓ TEST 3b PASSED")
        
        # Test 3c: Max retries reached
        logger.info("\n[TEST 3c] Max retries handling")
        task = TaskDefinition(
            task_id="test_task_3",
            task_type="test_api_call",
            priority=TaskPriority.HIGH,
            params={"test": True},
            max_retries=3
        )
        task.retry_count = 3  # Already at max
        
        should_retry = await decision_maker.handle_failure(
            task,
            "Connection timeout"
        )
        
        logger.info(f"[RESULT] Should retry: {should_retry}")
        assert should_retry == False, "Should not retry when max reached"
        logger.info("✓ TEST 3c PASSED")
        
        # Test 3d: Rate limit error - should retry with wait
        logger.info("\n[TEST 3d] Rate limit error handling")
        task = TaskDefinition(
            task_id="test_task_4",
            task_type="test_api_call",
            priority=TaskPriority.HIGH,
            params={"test": True}
        )
        
        import time
        start_time = time.time()
        
        should_retry = await decision_maker.handle_failure(
            task,
            "Rate limit exceeded - too many requests"
        )
        
        elapsed = time.time() - start_time
        
        logger.info(f"[RESULT] Should retry: {should_retry}")
        logger.info(f"[RESULT] Wait time: {elapsed:.1f}s")
        assert should_retry == True, "Should retry on rate limit"
        assert elapsed >= 2.0, "Should wait before retry"
        logger.info("✓ TEST 3d PASSED")
        
        logger.info("\n✓ ALL FAILURE HANDLING TESTS PASSED")
        return True
        
    except AssertionError as e:
        logger.error(f"✗ TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {str(e)}", exc_info=True)
        return False
    finally:
        await close_mongo_connection()


async def test_decision_statistics():
    """Test decision statistics tracking"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Decision Statistics")
    logger.info("="*60)
    
    try:
        await connect_to_mongo()
        
        orchestrator = get_orchestrator()
        decision_maker = orchestrator.get_decision_maker()
        
        # Make several decisions
        logger.info("\n[TEST] Making multiple decisions")
        
        for i in range(5):
            await decision_maker.decide_data_strategy(
                data_type="movie",
                identifier=f"TestMovie{i}",
                user_preference=None
            )
        
        # Get statistics
        stats = decision_maker.get_decision_statistics()
        
        logger.info("\n[STATISTICS]")
        logger.info(f"  Total decisions: {stats['total_decisions']}")
        logger.info(f"  Actions: {stats['actions']}")
        logger.info(f"  Average confidence: {stats['confidence_avg']:.2f}")
        
        logger.info("\n[RECENT DECISIONS]")
        for decision in stats['recent_decisions'][:3]:
            logger.info(f"  - {decision['action']} for {decision['identifier']}")
        
        assert stats['total_decisions'] >= 5, "Should track all decisions"
        assert 'actions' in stats, "Should track action distribution"
        assert 'confidence_avg' in stats, "Should track confidence"
        
        logger.info("\n✓ DECISION STATISTICS TEST PASSED")
        return True
        
    except AssertionError as e:
        logger.error(f"✗ TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {str(e)}", exc_info=True)
        return False
    finally:
        await close_mongo_connection()


async def test_integrated_workflow():
    """Test integrated workflow with autonomous decisions"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Integrated Workflow with Autonomous Decisions")
    logger.info("="*60)
    
    try:
        await connect_to_mongo()
        
        orchestrator = get_orchestrator()
        
        logger.info("\n[TEST] Running workflow with autonomous decision-making")
        logger.info("[TEST] Movie: RRR")
        
        result = await orchestrator.execute_workflow(
            movie_name="RRR",
            collect_new_data=True,
            max_reviews=3
        )
        
        logger.info(f"\n[RESULT] Success: {result.get('success')}")
        
        if result.get('success'):
            logger.info("[RESULT] Autonomous decisions were made during workflow")
            
            # Check decision statistics
            decision_maker = orchestrator.get_decision_maker()
            stats = decision_maker.get_decision_statistics()
            
            logger.info(f"\n[DECISIONS MADE]")
            logger.info(f"  Total: {stats['total_decisions']}")
            logger.info(f"  Actions: {stats['actions']}")
            
            assert stats['total_decisions'] > 0, "Workflow should make decisions"
            
            logger.info("\n✓ INTEGRATED WORKFLOW TEST PASSED")
            return True
        else:
            logger.error(f"[RESULT] Workflow failed: {result.get('error')}")
            return False
        
    except AssertionError as e:
        logger.error(f"✗ TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {str(e)}", exc_info=True)
        return False
    finally:
        await close_mongo_connection()


async def main():
    """Run all autonomous decision-making tests"""
    logger.info("\n" + "#"*60)
    logger.info("# AUTONOMOUS DECISION MAKING TEST SUITE")
    logger.info("#"*60)
    
    tests = [
        ("Data Fetching Decisions", test_data_fetching_decisions),
        ("Task Prioritization", test_task_prioritization),
        ("Failure Handling", test_failure_handling),
        ("Decision Statistics", test_decision_statistics),
        ("Integrated Workflow", test_integrated_workflow)
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
