"""
Unit Tests for ML Inference Service
Tests the machine learning model inference functionality
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.ml.inference import MLInferenceService


class TestMLInferenceService:
    """Test suite for ML Inference Service"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock ML model"""
        model = Mock()
        model.eval = Mock()
        model.return_value = Mock()
        return model
    
    @pytest.fixture
    def inference_service(self, mock_model):
        """Create ML Inference Service with mocked model"""
        with patch('app.ml.inference.SiameseNetwork', return_value=mock_model):
            service = MLInferenceService()
            service.model = mock_model
            return service
    
    def test_generate_embedding_success(self, inference_service):
        """Test successful embedding generation"""
        # Arrange
        movie_data = {
            'title': 'RRR',
            'genres': ['Action', 'Drama'],
            'overview': 'Epic movie'
        }
        expected_embedding = np.random.rand(128).astype(np.float32)
        
        with patch.object(inference_service, '_preprocess_movie', return_value=Mock()):
            with patch('numpy.array', return_value=expected_embedding):
                # Act
                embedding = inference_service.generate_embedding(movie_data)
                
                # Assert
                assert embedding is not None
                assert isinstance(embedding, np.ndarray)
                assert len(embedding) > 0
    
    def test_generate_embedding_invalid_data(self, inference_service):
        """Test embedding generation with invalid data"""
        # Arrange
        invalid_data = None
        
        # Act & Assert
        with pytest.raises((ValueError, TypeError, AttributeError)):
            inference_service.generate_embedding(invalid_data)
    
    def test_calculate_similarity_success(self, inference_service):
        """Test successful similarity calculation"""
        # Arrange
        embedding1 = np.random.rand(128).astype(np.float32)
        embedding2 = np.random.rand(128).astype(np.float32)
        
        # Act
        similarity = inference_service.calculate_similarity(embedding1, embedding2)
        
        # Assert
        assert similarity is not None
        assert isinstance(similarity, (float, np.floating))
        assert -1.0 <= similarity <= 1.0
    
    def test_calculate_similarity_identical_embeddings(self, inference_service):
        """Test similarity of identical embeddings"""
        # Arrange
        embedding = np.random.rand(128).astype(np.float32)
        
        # Act
        similarity = inference_service.calculate_similarity(embedding, embedding)
        
        # Assert
        assert similarity == pytest.approx(1.0, abs=0.01)
    
    def test_calculate_similarity_different_shapes(self, inference_service):
        """Test similarity calculation with mismatched shapes"""
        # Arrange
        embedding1 = np.random.rand(128).astype(np.float32)
        embedding2 = np.random.rand(64).astype(np.float32)
        
        # Act & Assert
        with pytest.raises((ValueError, AssertionError)):
            inference_service.calculate_similarity(embedding1, embedding2)
    
    def test_get_recommendations_success(self, inference_service):
        """Test successful recommendation generation"""
        # Arrange
        query_embedding = np.random.rand(128).astype(np.float32)
        candidate_embeddings = [
            np.random.rand(128).astype(np.float32) for _ in range(10)
        ]
        top_k = 5
        
        # Act
        recommendations = inference_service.get_recommendations(
            query_embedding,
            candidate_embeddings,
            top_k
        )
        
        # Assert
        assert recommendations is not None
        assert isinstance(recommendations, list)
        assert len(recommendations) <= top_k
    
    def test_get_recommendations_empty_candidates(self, inference_service):
        """Test recommendations with empty candidate list"""
        # Arrange
        query_embedding = np.random.rand(128).astype(np.float32)
        candidate_embeddings = []
        
        # Act
        recommendations = inference_service.get_recommendations(
            query_embedding,
            candidate_embeddings,
            top_k=5
        )
        
        # Assert
        assert recommendations is not None
        assert len(recommendations) == 0
    
    def test_model_loading(self):
        """Test ML model loading"""
        # Act
        with patch('torch.load') as mock_load:
            mock_load.return_value = {'state_dict': {}}
            service = MLInferenceService()
            
            # Assert
            assert service.model is not None
    
    def test_batch_embedding_generation(self, inference_service):
        """Test batch embedding generation"""
        # Arrange
        movies = [
            {'title': f'Movie{i}', 'genres': ['Action'], 'overview': f'Movie {i}'}
            for i in range(5)
        ]
        
        # Act
        embeddings = []
        for movie in movies:
            try:
                embedding = inference_service.generate_embedding(movie)
                embeddings.append(embedding)
            except Exception:
                pass
        
        # Assert
        assert len(embeddings) > 0


class TestMLInferenceEdgeCases:
    """Test edge cases and error handling"""
    
    def test_none_embedding(self):
        """Test handling of None embedding"""
        service = MLInferenceService()
        
        with pytest.raises((ValueError, TypeError, AttributeError)):
            service.calculate_similarity(None, np.random.rand(128))
    
    def test_zero_vector_embedding(self):
        """Test handling of zero vector"""
        service = MLInferenceService()
        zero_embedding = np.zeros(128).astype(np.float32)
        normal_embedding = np.random.rand(128).astype(np.float32)
        
        # Should handle gracefully (may return 0 or raise exception)
        try:
            similarity = service.calculate_similarity(zero_embedding, normal_embedding)
            assert isinstance(similarity, (float, np.floating))
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable to raise error for zero vectors
    
    def test_very_large_batch(self):
        """Test performance with large batch"""
        service = MLInferenceService()
        query_embedding = np.random.rand(128).astype(np.float32)
        large_batch = [np.random.rand(128).astype(np.float32) for _ in range(1000)]
        
        # Act
        try:
            recommendations = service.get_recommendations(
                query_embedding,
                large_batch,
                top_k=10
            )
            
            # Assert
            assert len(recommendations) <= 10
        except MemoryError:
            pytest.skip("Insufficient memory for large batch test")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
