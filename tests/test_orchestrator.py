"""
Test script for the Agentic AI Orchestrator
Tests the complete end-to-end workflow
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


async def test_orchestrator_workflow():
    """Test the complete agentic orchestrator workflow"""
    logger.info("="*60)
    logger.info("Testing Agentic AI Orchestrator")
    logger.info("="*60)
    
    try:
        # Connect to database
        logger.info("\n[SETUP] Connecting to database...")
        await connect_to_mongo()
        logger.info("[SETUP] ✓ Database connected")
        
        # Get orchestrator
        logger.info("\n[SETUP] Initializing orchestrator...")
        orchestrator = get_orchestrator()
        logger.info("[SETUP] ✓ Orchestrator ready")
        
        # Test 1: Execute workflow with minimal data collection
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Complete Workflow with Fresh Data Collection")
        logger.info("="*60)
        
        movie_name = "RRR"
        
        logger.info(f"\n[TEST] Movie: {movie_name}")
        logger.info(f"[TEST] Strategy: Collect fresh data (max 5 reviews)")
        
        result = await orchestrator.execute_workflow(
            movie_name=movie_name,
            collect_new_data=True,
            max_reviews=5  # Limited to avoid rate limits
        )
        
        logger.info("\n[RESULT] Workflow completed!")
        logger.info(f"[RESULT] Success: {result.get('success')}")
        logger.info(f"[RESULT] Workflow ID: {result.get('workflow_id')}")
        
        if result.get('success'):
            movie_info = result.get('movie', {})
            logger.info(f"\n[MOVIE] ID: {movie_info.get('id')}")
            logger.info(f"[MOVIE] Name: {movie_info.get('name')}")
            logger.info(f"[MOVIE] Reviews Analyzed: {movie_info.get('reviews_analyzed')}")
            
            analysis = result.get('analysis', {})
            logger.info(f"\n[ANALYSIS] Reviews Processed: {analysis.get('reviews_analyzed')}")
            logger.info(f"[ANALYSIS] Average Sentiment: {analysis.get('average_sentiment', 0):.2f}")
            
            distribution = analysis.get('sentiment_distribution', {})
            logger.info(f"[ANALYSIS] Positive: {distribution.get('positive_percentage', 0):.1f}%")
            logger.info(f"[ANALYSIS] Negative: {distribution.get('negative_percentage', 0):.1f}%")
            logger.info(f"[ANALYSIS] Neutral: {distribution.get('neutral_percentage', 0):.1f}%")
            
            recommendations = result.get('recommendations', [])
            logger.info(f"\n[RECOMMENDATIONS] Count: {len(recommendations)}")
            
            if recommendations:
                logger.info("\n[RECOMMENDATIONS] Top 3:")
                for i, rec in enumerate(recommendations[:3], 1):
                    logger.info(f"  {i}. {rec.get('title', 'Unknown')}")
                    logger.info(f"     Score: {rec.get('recommendation_score', 0):.1f}")
                    logger.info(f"     Explanation: {rec.get('explanation', 'N/A')}")
            
            # Show workflow state
            workflow_state = result.get('workflow_state', {})
            logger.info(f"\n[WORKFLOW] Agents Executed: {len(workflow_state.get('agents_executed', []))}")
            for agent_exec in workflow_state.get('agents_executed', []):
                logger.info(f"  - {agent_exec.get('agent')}: {agent_exec.get('status')}")
        else:
            logger.error(f"\n[ERROR] Workflow failed: {result.get('error')}")
        
        # Test 2: Quick recommendation with cached data
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Quick Recommendation (Cached Data)")
        logger.info("="*60)
        
        logger.info(f"\n[TEST] Movie: {movie_name}")
        logger.info(f"[TEST] Strategy: Use cached data only")
        
        cached_result = await orchestrator.execute_workflow(
            movie_name=movie_name,
            collect_new_data=False,
            max_reviews=0
        )
        
        logger.info("\n[RESULT] Cached workflow completed!")
        logger.info(f"[RESULT] Success: {cached_result.get('success')}")
        
        if cached_result.get('success'):
            cached_recs = cached_result.get('recommendations', [])
            logger.info(f"[RESULT] Recommendations from cache: {len(cached_recs)}")
        
        # Test 3: Different movie
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Different Movie")
        logger.info("="*60)
        
        different_movie = "Baahubali"
        
        logger.info(f"\n[TEST] Movie: {different_movie}")
        logger.info(f"[TEST] Strategy: Fresh data (max 3 reviews)")
        
        different_result = await orchestrator.execute_workflow(
            movie_name=different_movie,
            collect_new_data=True,
            max_reviews=3
        )
        
        logger.info("\n[RESULT] Workflow completed!")
        logger.info(f"[RESULT] Success: {different_result.get('success')}")
        
        if different_result.get('success'):
            diff_movie_info = different_result.get('movie', {})
            logger.info(f"[RESULT] Movie: {diff_movie_info.get('name')}")
            logger.info(f"[RESULT] Reviews: {diff_movie_info.get('reviews_analyzed')}")
            diff_recs = different_result.get('recommendations', [])
            logger.info(f"[RESULT] Recommendations: {len(diff_recs)}")
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"\n[ERROR] Test failed: {str(e)}", exc_info=True)
        return False
    
    finally:
        # Cleanup
        logger.info("\n[CLEANUP] Closing database connection...")
        await close_mongo_connection()
        logger.info("[CLEANUP] ✓ Done")


async def test_workflow_status():
    """Test workflow status tracking"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Workflow Status Tracking")
    logger.info("="*60)
    
    try:
        await connect_to_mongo()
        orchestrator = get_orchestrator()
        
        # Execute a workflow
        result = await orchestrator.execute_workflow(
            movie_name="RRR",
            collect_new_data=False,
            max_reviews=0
        )
        
        workflow_id = result.get('workflow_id')
        logger.info(f"\n[TEST] Workflow ID: {workflow_id}")
        
        # Get status
        status = await orchestrator.get_workflow_status(workflow_id)
        logger.info(f"[STATUS] Current Status: {status.get('status')}")
        logger.info(f"[STATUS] Agents: {len(status.get('agents_executed', []))}")
        
        # Try to get non-existent workflow
        fake_status = await orchestrator.get_workflow_status("fake_id")
        logger.info(f"\n[TEST] Non-existent workflow: {fake_status.get('error')}")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Status test failed: {str(e)}", exc_info=True)
        return False
    
    finally:
        await close_mongo_connection()


async def main():
    """Run all tests"""
    logger.info("\n" + "#"*60)
    logger.info("# AGENTIC AI ORCHESTRATOR TEST SUITE")
    logger.info("#"*60)
    
    # Test 1: Complete workflow
    test1_passed = await test_orchestrator_workflow()
    
    # Test 2: Status tracking
    test2_passed = await test_workflow_status()
    
    # Summary
    logger.info("\n" + "#"*60)
    logger.info("# TEST SUMMARY")
    logger.info("#"*60)
    logger.info(f"Orchestrator Workflow: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    logger.info(f"Status Tracking: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        logger.error("\n✗ SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
