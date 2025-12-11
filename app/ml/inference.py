"""
ML Model Inference Module
==========================
Handles loading and inference for the trained Siamese Network
"""

import torch
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
from functools import lru_cache

from app.ml.models.siamese_network import SiameseNetwork, create_siamese_model

# Configure logging
logger = logging.getLogger(__name__)


class TeluguTokenizer:
    """Telugu text tokenizer for preprocessing reviews."""
    
    def __init__(self):
        self.vocab_size = 0
        self.word2idx = {}
        self.idx2word = {}
    
    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer from JSON file."""
        logger.info(f"Loading tokenizer from {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        
        logger.info(f"Tokenizer loaded: {len(tokenizer.word2idx)} tokens")
        return tokenizer
    
    def encode(self, text: str, max_length: int = 50) -> Tuple[List[int], int]:
        """
        Encode text to token indices.
        
        Args:
            text: Input Telugu text
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (token_ids, actual_length)
        """
        words = text.split()
        tokens = [
            self.word2idx.get(word, self.word2idx.get("<UNK>", 1)) 
            for word in words
        ]
        length = min(len(tokens), max_length)
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens = tokens + [self.word2idx.get("<PAD>", 0)] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        
        return tokens, length
    
    def encode_batch(self, texts: List[str], max_length: int = 50) -> Tuple[List[List[int]], List[int]]:
        """
        Encode multiple texts to token indices.
        
        Args:
            texts: List of Telugu texts
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (list of token_ids, list of lengths)
        """
        all_tokens = []
        all_lengths = []
        
        for text in texts:
            tokens, length = self.encode(text, max_length)
            all_tokens.append(tokens)
            all_lengths.append(length)
        
        return all_tokens, all_lengths


class ModelInference:
    """Handles model loading and inference operations."""
    
    # Singleton instance
    _instance: Optional['ModelInference'] = None
    
    # Sentiment label mapping (matches training)
    SENTIMENT_MAP = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }
    
    def __init__(
        self,
        model_path: str = "checkpoints/best_model.pt",
        tokenizer_path: str = "checkpoints/tokenizer.json",
        device: Optional[str] = None
    ):
        """
        Initialize model inference.
        
        Args:
            model_path: Path to trained model checkpoint
            tokenizer_path: Path to tokenizer JSON
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing model inference on device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = TeluguTokenizer.load(tokenizer_path)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Model metadata
        self.vocab_size = len(self.tokenizer.word2idx)
        self.max_length = 50
        
        logger.info("Model inference initialized successfully")
    
    def _load_model(self, model_path: str) -> SiameseNetwork:
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {model_path}")
        
        # Use actual tokenizer vocabulary size
        vocab_size = len(self.tokenizer.word2idx)
        logger.info(f"Creating model with vocab_size={vocab_size}")
        
        # Create model architecture
        model = create_siamese_model(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            device=self.device
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        
        logger.info(f"Model loaded successfully (epoch {checkpoint['epoch']}, "
                   f"val_acc: {checkpoint['best_val_acc']:.2f}%)")
        
        return model
    
    @classmethod
    def get_instance(
        cls,
        model_path: str = "checkpoints/best_model.pt",
        tokenizer_path: str = "checkpoints/tokenizer.json",
        device: Optional[str] = None
    ) -> 'ModelInference':
        """
        Get singleton instance of ModelInference.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer
            device: Device to use
            
        Returns:
            ModelInference instance
        """
        if cls._instance is None:
            cls._instance = cls(model_path, tokenizer_path, device)
        return cls._instance
    
    def predict_sentiment(
        self,
        text: str,
        return_confidence: bool = True
    ) -> Dict[str, any]:
        """
        Predict sentiment for a single review.
        
        Args:
            text: Telugu review text
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with sentiment prediction and optional confidence
        """
        # Encode text
        tokens, length = self.tokenizer.encode(text, self.max_length)
        
        # Convert to tensors
        review_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        length_tensor = torch.tensor([length], dtype=torch.long).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model.predict_sentiment(review_tensor, length_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Build result
        result = {
            'sentiment': self.SENTIMENT_MAP[predicted_class],
            'sentiment_code': predicted_class
        }
        
        if return_confidence:
            result['confidence'] = round(confidence, 4)
            result['probabilities'] = {
                'negative': round(probabilities[0, 0].item(), 4),
                'neutral': round(probabilities[0, 1].item(), 4),
                'positive': round(probabilities[0, 2].item(), 4)
            }
        
        return result
    
    def predict_sentiment_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_confidence: bool = True
    ) -> List[Dict[str, any]]:
        """
        Predict sentiment for multiple reviews (batch processing).
        
        Args:
            texts: List of Telugu review texts
            batch_size: Batch size for processing
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Encode batch
            tokens_batch, lengths_batch = self.tokenizer.encode_batch(
                batch_texts, 
                self.max_length
            )
            
            # Convert to tensors
            reviews_tensor = torch.tensor(tokens_batch, dtype=torch.long).to(self.device)
            lengths_tensor = torch.tensor(lengths_batch, dtype=torch.long).to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model.predict_sentiment(reviews_tensor, lengths_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
                confidences = probabilities.max(dim=1).values.cpu().numpy()
                probs_array = probabilities.cpu().numpy()
            
            # Build results
            for j, pred_class in enumerate(predicted_classes):
                result = {
                    'sentiment': self.SENTIMENT_MAP[pred_class],
                    'sentiment_code': int(pred_class)
                }
                
                if return_confidence:
                    result['confidence'] = round(float(confidences[j]), 4)
                    result['probabilities'] = {
                        'negative': round(float(probs_array[j, 0]), 4),
                        'neutral': round(float(probs_array[j, 1]), 4),
                        'positive': round(float(probs_array[j, 2]), 4)
                    }
                
                results.append(result)
        
        return results
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a review.
        
        Args:
            text: Telugu review text
            
        Returns:
            128-dimensional embedding vector
        """
        # Encode text
        tokens, length = self.tokenizer.encode(text, self.max_length)
        
        # Convert to tensors
        review_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        length_tensor = torch.tensor([length], dtype=torch.long).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model.get_embedding(review_tensor, length_tensor)
            embedding_list = embedding[0].cpu().numpy().tolist()
        
        return embedding_list
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two reviews.
        
        Args:
            text1: First review text
            text2: Second review text
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Encode both texts
        tokens1, length1 = self.tokenizer.encode(text1, self.max_length)
        tokens2, length2 = self.tokenizer.encode(text2, self.max_length)
        
        # Convert to tensors
        review1_tensor = torch.tensor([tokens1], dtype=torch.long).to(self.device)
        review2_tensor = torch.tensor([tokens2], dtype=torch.long).to(self.device)
        length1_tensor = torch.tensor([length1], dtype=torch.long).to(self.device)
        length2_tensor = torch.tensor([length2], dtype=torch.long).to(self.device)
        
        # Get embeddings first
        with torch.no_grad():
            encoding1 = self.model.get_embedding(review1_tensor, length1_tensor)
            encoding2 = self.model.get_embedding(review2_tensor, length2_tensor)
            
            # Now compute similarity between the encodings
            similarity = self.model.compute_similarity(encoding1, encoding2)
            similarity_score = similarity[0].item()
        
        return round(similarity_score, 4)


# Cached model instance getter
@lru_cache(maxsize=1)
def get_model_inference() -> ModelInference:
    """
    Get cached model inference instance.
    
    Returns:
        ModelInference singleton instance
    """
    return ModelInference.get_instance()
