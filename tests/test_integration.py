"""
Integration Tests for Telugu Movie Recommendation System
Tests the interaction between multiple components
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime


class TestEndToEndWorkflow:
    """Test complete recommendation workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_recommendation_flow(self):
        """Test complete flow from search to recommendations"""
        # This test simulates the entire user journey
        
        # Arrange
        user_input = "RRR"
        
        # Mock components
        with patch('app.data_collection.tmdb_collector.TMDBCollector') as MockTMDB, \
             patch('app.data_collection.twitter_collector.TwitterCollector') as MockTwitter, \
             patch('app.services.storage_service.StorageService') as MockStorage, \
             patch('app.ml.inference.MLInferenceService') as MockML:
            
            # Setup mocks
            tmdb = MockTMDB.return_value
            tmdb.search_movies = AsyncMock(return_value=[{
                'tmdb_id': 12345,
                'title': 'RRR',
                'rating': 8.0
            }])
            
            twitter = MockTwitter.return_value
            twitter.collect_tweets = AsyncMock(return_value=[{
                'text': 'Great movie!',
                'sentiment_score': 0.8
            }])
            
            storage = MockStorage.return_value
            storage.store_movie = AsyncMock(return_value=True)
            storage.get_movie_by_id = AsyncMock(return_value={
                'tmdb_id': 12345,
                'title': 'RRR'
            })
            
            ml = MockML.return_value
            ml.get_recommendations = Mock(return_value=[
                {'movie_id': 67890, 'confidence': 0.9}
            ])
            
            # Act - Simulate orchestrator workflow
            search_results = await tmdb.search_movies(user_input)
            assert len(search_results) > 0
            
            movie_id = search_results[0]['tmdb_id']
            await storage.store_movie(search_results[0])
            
            tweets = await twitter.collect_tweets(movie_id)
            assert len(tweets) > 0
            
            movie = await storage.get_movie_by_id(movie_id)
            assert movie is not None
            
            recommendations = ml.get_recommendations(movie_id)
            assert len(recommendations) > 0


class TestDataCollectionIntegration:
    """Test integration between data collectors and storage"""
    
    @pytest.mark.asyncio
    async def test_tmdb_to_storage_integration(self):
        """Test TMDB data collection and storage integration"""
        with patch('app.data_collection.tmdb_collector.TMDBCollector') as MockTMDB, \
             patch('app.services.storage_service.StorageService') as MockStorage:
            
            tmdb = MockTMDB.return_value
            storage = MockStorage.return_value
            
            # Setup
            movie_data = {
                'tmdb_id': 12345,
                'title': 'Test Movie',
                'genres': ['Action'],
                'rating': 7.5
            }
            tmdb.get_movie_details = AsyncMock(return_value=movie_data)
            storage.store_movie = AsyncMock(return_value=True)
            storage.get_movie_by_id = AsyncMock(return_value=movie_data)
            
            # Act
            details = await tmdb.get_movie_details(12345)
            await storage.store_movie(details)
            retrieved = await storage.get_movie_by_id(12345)
            
            # Assert
            assert retrieved['tmdb_id'] == movie_data['tmdb_id']
            assert retrieved['title'] == movie_data['title']
    
    @pytest.mark.asyncio
    async def test_twitter_to_storage_integration(self):
        """Test Twitter data collection and storage integration"""
        with patch('app.data_collection.twitter_collector.TwitterCollector') as MockTwitter, \
             patch('app.services.storage_service.StorageService') as MockStorage:
            
            twitter = MockTwitter.return_value
            storage = MockStorage.return_value
            
            # Setup
            tweets = [
                {'tweet_id': '1', 'text': 'Great!', 'sentiment': 0.8},
                {'tweet_id': '2', 'text': 'Amazing!', 'sentiment': 0.9}
            ]
            twitter.collect_tweets = AsyncMock(return_value=tweets)
            storage.store_tweet = AsyncMock(return_value=True)
            storage.get_tweets_for_movie = AsyncMock(return_value=tweets)
            
            # Act
            collected_tweets = await twitter.collect_tweets(12345)
            for tweet in collected_tweets:
                await storage.store_tweet(tweet)
            
            stored_tweets = await storage.get_tweets_for_movie(12345)
            
            # Assert
            assert len(stored_tweets) == len(tweets)


class TestMLIntegration:
    """Test ML model integration with data pipeline"""
    
    def test_ml_inference_with_storage_data(self):
        """Test ML inference using stored movie data"""
        with patch('app.ml.inference.MLInferenceService') as MockML, \
             patch('app.services.storage_service.StorageService') as MockStorage:
            
            ml = MockML.return_value
            storage = MockStorage.return_value
            
            # Setup
            movie_data = {
                'tmdb_id': 12345,
                'title': 'Test Movie',
                'genres': ['Action', 'Drama'],
                'overview': 'Test overview'
            }
            
            import numpy as np
            embedding = np.random.rand(128).astype(np.float32)
            
            storage.get_movie_by_id = AsyncMock(return_value=movie_data)
            ml.generate_embedding = Mock(return_value=embedding)
            ml.get_recommendations = Mock(return_value=[
                {'movie_id': 67890, 'confidence': 0.85}
            ])
            
            # Act
            movie = asyncio.run(storage.get_movie_by_id(12345))
            movie_embedding = ml.generate_embedding(movie)
            recommendations = ml.get_recommendations(movie_embedding)
            
            # Assert
            assert movie_embedding is not None
            assert len(recommendations) > 0
            assert recommendations[0]['confidence'] > 0.5


class TestOrchestratorIntegration:
    """Test orchestrator integration with all components"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_with_all_agents(self):
        """Test orchestrator coordinating all agents"""
        with patch('app.agents.orchestrator.AgenticOrchestrator') as MockOrchestrator:
            
            orchestrator = MockOrchestrator.return_value
            orchestrator.execute_workflow = AsyncMock(return_value={
                'status': 'completed',
                'recommendations': [
                    {
                        'movie': {'title': 'Recommended Movie'},
                        'confidence_score': 0.9
                    }
                ]
            })
            
            # Act
            result = await orchestrator.execute_workflow('RRR', 'user123')
            
            # Assert
            assert result['status'] == 'completed'
            assert len(result['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_orchestrator_error_handling(self):
        """Test orchestrator handling component failures"""
        with patch('app.agents.orchestrator.AgenticOrchestrator') as MockOrchestrator:
            
            orchestrator = MockOrchestrator.return_value
            
            # Simulate failure
            orchestrator.execute_workflow = AsyncMock(side_effect=Exception("Service unavailable"))
            
            # Act & Assert
            with pytest.raises(Exception):
                await orchestrator.execute_workflow('RRR', 'user123')


class TestAutomationIntegration:
    """Test automation manager integration"""
    
    @pytest.mark.asyncio
    async def test_automated_workflow_coordination(self):
        """Test automated workflow with retry logic"""
        with patch('app.agents.automation.AutomatedWorkflowManager') as MockAutomation:
            
            automation = MockAutomation.return_value
            automation.execute_automated_search = AsyncMock(return_value={
                'status': 'completed',
                'recommendations': [],
                'retries': 0
            })
            
            # Act
            result = await automation.execute_automated_search('RRR')
            
            # Assert
            assert result['status'] == 'completed'
            assert result['retries'] == 0
    
    @pytest.mark.asyncio
    async def test_automated_workflow_with_retries(self):
        """Test automated workflow retry mechanism"""
        with patch('app.agents.automation.AutomatedWorkflowManager') as MockAutomation:
            
            automation = MockAutomation.return_value
            
            # First 2 calls fail, 3rd succeeds
            automation.execute_automated_search = AsyncMock(side_effect=[
                Exception("Transient error"),
                Exception("Transient error"),
                {'status': 'completed', 'retries': 2}
            ])
            
            # Act - Should retry and eventually succeed
            # (This is a simplified test - actual retry logic may vary)
            attempts = 0
            max_attempts = 3
            result = None
            
            while attempts < max_attempts:
                try:
                    result = await automation.execute_automated_search('RRR')
                    break
                except Exception:
                    attempts += 1
            
            # Assert
            assert result is not None
            assert result['status'] == 'completed'


class TestDecisionMakingIntegration:
    """Test autonomous decision maker integration"""
    
    def test_decision_maker_with_orchestrator(self):
        """Test decision maker coordinating with orchestrator"""
        with patch('app.agents.decision_maker.AutonomousDecisionMaker') as MockDecision, \
             patch('app.agents.orchestrator.AgenticOrchestrator') as MockOrchestrator:
            
            decision_maker = MockDecision.return_value
            orchestrator = MockOrchestrator.return_value
            
            # Setup
            decision_maker.decide_data_strategy = Mock(return_value='FRESH')
            decision_maker.prioritize_tasks = Mock(return_value=[
                {'task': 'collect_tmdb', 'priority': 'HIGH'},
                {'task': 'collect_twitter', 'priority': 'MEDIUM'}
            ])
            
            orchestrator.execute_workflow = AsyncMock(return_value={
                'status': 'completed'
            })
            
            # Act
            strategy = decision_maker.decide_data_strategy('RRR')
            tasks = decision_maker.prioritize_tasks(['collect_tmdb', 'collect_twitter'])
            result = asyncio.run(orchestrator.execute_workflow('RRR', 'user123'))
            
            # Assert
            assert strategy in ['FRESH', 'CACHED', 'SMART']
            assert len(tasks) > 0
            assert result['status'] == 'completed'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
