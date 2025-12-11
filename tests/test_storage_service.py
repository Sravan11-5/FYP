"""
Unit Tests for Storage Service
Tests MongoDB database operations
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from app.services.storage_service import StorageService


class TestStorageService:
    """Test suite for Storage Service"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock MongoDB database"""
        db = Mock()
        db.movies = Mock()
        db.tweets = Mock()
        db.recommendations = Mock()
        return db
    
    @pytest.fixture
    async def storage_service(self, mock_db):
        """Create Storage Service with mocked database"""
        with patch('motor.motor_asyncio.AsyncIOMotorClient') as mock_client:
            mock_client.return_value.__getitem__.return_value = mock_db
            service = StorageService()
            service.db = mock_db
            return service
    
    @pytest.mark.asyncio
    async def test_store_movie_success(self, storage_service, mock_db):
        """Test successful movie storage"""
        # Arrange
        movie_data = {
            'tmdb_id': 12345,
            'title': 'RRR',
            'original_title': 'రౌద్రం రణం రుధిరం',
            'genres': ['Action', 'Drama'],
            'rating': 8.0
        }
        mock_db.movies.update_one = AsyncMock(return_value=Mock(modified_count=1))
        
        # Act
        result = await storage_service.store_movie(movie_data)
        
        # Assert
        assert result is not None
        mock_db.movies.update_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_movie_duplicate(self, storage_service, mock_db):
        """Test storing duplicate movie"""
        # Arrange
        movie_data = {'tmdb_id': 12345, 'title': 'RRR'}
        mock_db.movies.update_one = AsyncMock(return_value=Mock(modified_count=0))
        
        # Act
        result = await storage_service.store_movie(movie_data)
        
        # Assert
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_get_movie_by_id_found(self, storage_service, mock_db):
        """Test retrieving existing movie"""
        # Arrange
        expected_movie = {
            '_id': '123',
            'tmdb_id': 12345,
            'title': 'RRR'
        }
        mock_db.movies.find_one = AsyncMock(return_value=expected_movie)
        
        # Act
        movie = await storage_service.get_movie_by_id(12345)
        
        # Assert
        assert movie is not None
        assert movie['tmdb_id'] == 12345
        assert movie['title'] == 'RRR'
    
    @pytest.mark.asyncio
    async def test_get_movie_by_id_not_found(self, storage_service, mock_db):
        """Test retrieving non-existent movie"""
        # Arrange
        mock_db.movies.find_one = AsyncMock(return_value=None)
        
        # Act
        movie = await storage_service.get_movie_by_id(99999)
        
        # Assert
        assert movie is None
    
    @pytest.mark.asyncio
    async def test_search_movies_by_title(self, storage_service, mock_db):
        """Test movie search by title"""
        # Arrange
        expected_movies = [
            {'tmdb_id': 1, 'title': 'RRR'},
            {'tmdb_id': 2, 'title': 'RRR 2'}
        ]
        mock_cursor = Mock()
        mock_cursor.to_list = AsyncMock(return_value=expected_movies)
        mock_db.movies.find = Mock(return_value=mock_cursor)
        
        # Act
        movies = await storage_service.search_movies('RRR')
        
        # Assert
        assert movies is not None
        assert len(movies) == 2
    
    @pytest.mark.asyncio
    async def test_store_tweet_success(self, storage_service, mock_db):
        """Test successful tweet storage"""
        # Arrange
        tweet_data = {
            'tweet_id': '123456',
            'text': 'Great movie!',
            'movie_id': 12345,
            'sentiment_score': 0.8
        }
        mock_db.tweets.insert_one = AsyncMock(return_value=Mock(inserted_id='abc'))
        
        # Act
        result = await storage_service.store_tweet(tweet_data)
        
        # Assert
        assert result is not None
        mock_db.tweets.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_tweets_for_movie(self, storage_service, mock_db):
        """Test retrieving tweets for a movie"""
        # Arrange
        expected_tweets = [
            {'tweet_id': '1', 'text': 'Great!'},
            {'tweet_id': '2', 'text': 'Awesome!'}
        ]
        mock_cursor = Mock()
        mock_cursor.to_list = AsyncMock(return_value=expected_tweets)
        mock_db.tweets.find = Mock(return_value=mock_cursor)
        
        # Act
        tweets = await storage_service.get_tweets_for_movie(12345)
        
        # Assert
        assert tweets is not None
        assert len(tweets) == 2
    
    @pytest.mark.asyncio
    async def test_store_recommendation_success(self, storage_service, mock_db):
        """Test successful recommendation storage"""
        # Arrange
        recommendation_data = {
            'user_id': 'user123',
            'query_movie_id': 12345,
            'recommended_movies': [67890, 11111],
            'confidence_scores': [0.9, 0.8],
            'timestamp': datetime.utcnow()
        }
        mock_db.recommendations.insert_one = AsyncMock(
            return_value=Mock(inserted_id='rec123')
        )
        
        # Act
        result = await storage_service.store_recommendation(recommendation_data)
        
        # Assert
        assert result is not None
        mock_db.recommendations.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_recommendations(self, storage_service, mock_db):
        """Test retrieving user recommendations"""
        # Arrange
        expected_recs = [
            {'user_id': 'user123', 'recommended_movies': [1, 2, 3]},
            {'user_id': 'user123', 'recommended_movies': [4, 5, 6]}
        ]
        mock_cursor = Mock()
        mock_cursor.to_list = AsyncMock(return_value=expected_recs)
        mock_db.recommendations.find = Mock(return_value=mock_cursor)
        
        # Act
        recommendations = await storage_service.get_user_recommendations('user123')
        
        # Assert
        assert recommendations is not None
        assert len(recommendations) == 2
    
    @pytest.mark.asyncio
    async def test_delete_movie_success(self, storage_service, mock_db):
        """Test successful movie deletion"""
        # Arrange
        mock_db.movies.delete_one = AsyncMock(
            return_value=Mock(deleted_count=1)
        )
        
        # Act
        result = await storage_service.delete_movie(12345)
        
        # Assert
        assert result is True
        mock_db.movies.delete_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_movie_not_found(self, storage_service, mock_db):
        """Test deleting non-existent movie"""
        # Arrange
        mock_db.movies.delete_one = AsyncMock(
            return_value=Mock(deleted_count=0)
        )
        
        # Act
        result = await storage_service.delete_movie(99999)
        
        # Assert
        assert result is False


class TestStorageServiceEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test handling of database connection errors"""
        with patch('motor.motor_asyncio.AsyncIOMotorClient') as mock_client:
            mock_client.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception):
                service = StorageService()
    
    @pytest.mark.asyncio
    async def test_invalid_movie_data(self):
        """Test storing invalid movie data"""
        service = StorageService()
        
        with pytest.raises((ValueError, TypeError, KeyError)):
            await service.store_movie({})  # Missing required fields
    
    @pytest.mark.asyncio
    async def test_large_batch_insert(self):
        """Test inserting large batch of movies"""
        service = StorageService()
        movies = [
            {'tmdb_id': i, 'title': f'Movie {i}'}
            for i in range(1000)
        ]
        
        # Should handle large batches gracefully
        try:
            for movie in movies:
                await service.store_movie(movie)
        except Exception as e:
            pytest.fail(f"Failed to handle large batch: {e}")
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent database access"""
        import asyncio
        service = StorageService()
        
        async def store_movie(movie_id):
            return await service.store_movie({
                'tmdb_id': movie_id,
                'title': f'Movie {movie_id}'
            })
        
        # Run concurrent operations
        tasks = [store_movie(i) for i in range(10)]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Should complete without deadlocks
            assert len(results) == 10
        except Exception as e:
            pytest.fail(f"Concurrent access failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
