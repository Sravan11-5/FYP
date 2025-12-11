"""
Siamese Network Architecture for Telugu Movie Review Sentiment Analysis
========================================================================

This module implements a Siamese Neural Network for learning semantic similarity
between Telugu movie reviews. The architecture uses twin networks with shared weights
to encode reviews into embeddings, then computes similarity for sentiment analysis.

Architecture Overview:
1. Input Layer: Tokenized Telugu text sequences
2. Embedding Layer: Word embeddings (trainable or pre-trained)
3. Twin Networks: Identical LSTM/GRU networks with shared weights
4. Similarity Layer: Cosine similarity or Euclidean distance
5. Output: Similarity score for sentiment classification

References:
- Bromley et al. (1993) - Siamese Networks for one-shot learning
- Koch et al. (2015) - Siamese Networks for image verification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class TeluguEmbedding(nn.Module):
    """
    Embedding layer specifically designed for Telugu text.
    
    Handles:
    - Telugu Unicode characters (0x0C00-0x0C7F)
    - Out-of-vocabulary (OOV) tokens
    - Padding for variable-length sequences
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embedding vectors
        padding_idx: Index used for padding
        pretrained_embeddings: Optional pre-trained embeddings
        freeze_embeddings: Whether to freeze embedding weights
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        padding_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False
    ):
        super(TeluguEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Create embedding layer
        if pretrained_embeddings is not None:
            # Use pre-trained embeddings
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embeddings,
                padding_idx=padding_idx
            )
        else:
            # Initialize random embeddings
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=padding_idx
            )
            # Xavier uniform initialization
            nn.init.xavier_uniform_(self.embedding.weight)
            # Zero out padding embedding
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through embedding layer.
        
        Args:
            x: Input tensor of token indices [batch_size, seq_length]
        
        Returns:
            Embedded tensor [batch_size, seq_length, embedding_dim]
        """
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        return embedded


class TwinNetwork(nn.Module):
    """
    Twin network for processing review text.
    
    Uses bidirectional LSTM to capture context from both directions
    in Telugu text, which is important for sentiment understanding.
    
    Args:
        embedding_dim: Dimension of input embeddings
        hidden_dim: Dimension of LSTM hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(TwinNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        
        # Attention mechanism (optional, for better focus on important words)
        self.attention = nn.Linear(hidden_dim * self.num_directions, 1)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def attention_forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism to LSTM outputs.
        
        Args:
            lstm_output: LSTM output [batch_size, seq_length, hidden_dim*2]
        
        Returns:
            Attended output [batch_size, hidden_dim*2]
        """
        # Calculate attention weights [batch_size, seq_length, 1]
        attention_weights = torch.softmax(
            self.attention(lstm_output),
            dim=1
        )
        
        # Apply attention [batch_size, hidden_dim*2]
        attended = torch.sum(attention_weights * lstm_output, dim=1)
        
        return attended
    
    def forward(
        self,
        embedded: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through twin network.
        
        Args:
            embedded: Embedded input [batch_size, seq_length, embedding_dim]
            lengths: Actual lengths of sequences (for packing)
        
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        batch_size = embedded.size(0)
        
        # Pack sequence if lengths provided (for efficiency)
        if lengths is not None:
            # Sort by length for packing
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            embedded_sorted = embedded[sort_idx]
            
            # Pack
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded_sorted,
                lengths_sorted.cpu(),
                batch_first=True,
                enforce_sorted=True
            )
            
            # LSTM forward
            packed_output, (hidden, cell) = self.lstm(packed)
            
            # Unpack
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True
            )
            
            # Restore original order
            _, unsort_idx = sort_idx.sort()
            lstm_output = lstm_output[unsort_idx]
        else:
            # Regular LSTM forward
            lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Apply attention
        attended = self.attention_forward(lstm_output)
        
        # Layer normalization
        normalized = self.layer_norm(attended)
        
        # Project to output dimension
        output = self.output_projection(normalized)
        
        # L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=1)
        
        return output


class SiameseNetwork(nn.Module):
    """
    Complete Siamese Network for Telugu Review Similarity.
    
    Architecture:
    1. Embedding layer (shared)
    2. Twin LSTM networks (shared weights)
    3. Similarity computation
    4. Classification layer
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden states
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        similarity_metric: 'cosine' or 'euclidean'
        pretrained_embeddings: Optional pre-trained embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        similarity_metric: str = 'cosine',
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        super(SiameseNetwork, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.similarity_metric = similarity_metric
        
        # Shared embedding layer
        self.embedding = TeluguEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pretrained_embeddings=pretrained_embeddings
        )
        
        # Shared twin network (weights are shared between both inputs)
        self.twin_network = TwinNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Classification head
        output_dim = hidden_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # 3 classes: positive, negative, neutral
        )
        
        # For contrastive loss (alternative to classification)
        self.similarity_layer = nn.Linear(1, 1)
    
    def forward_once(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for one input through shared network.
        
        Args:
            x: Input token indices [batch_size, seq_length]
            lengths: Actual sequence lengths
        
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        # Embed
        embedded = self.embedding(x)
        
        # Encode through twin network
        encoded = self.twin_network(embedded, lengths)
        
        return encoded
    
    def compute_similarity(
        self,
        encoding1: torch.Tensor,
        encoding2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between two encodings.
        
        Args:
            encoding1: First encoding [batch_size, dim]
            encoding2: Second encoding [batch_size, dim]
        
        Returns:
            Similarity scores [batch_size]
        """
        if self.similarity_metric == 'cosine':
            # Cosine similarity (already normalized in twin network)
            similarity = torch.sum(encoding1 * encoding2, dim=1)
        elif self.similarity_metric == 'euclidean':
            # Euclidean distance (negated and scaled)
            distance = torch.sqrt(torch.sum((encoding1 - encoding2) ** 2, dim=1))
            similarity = 1 / (1 + distance)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarity
    
    def forward(
        self,
        review1: torch.Tensor,
        review2: torch.Tensor,
        lengths1: Optional[torch.Tensor] = None,
        lengths2: Optional[torch.Tensor] = None,
        return_similarity: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through Siamese Network.
        
        Args:
            review1: First review token indices [batch_size, seq_length]
            review2: Second review token indices [batch_size, seq_length]
            lengths1: Lengths of first reviews
            lengths2: Lengths of second reviews
            return_similarity: Whether to return similarity scores
        
        Returns:
            If return_similarity=False:
                - Classification logits [batch_size, 3]
            If return_similarity=True:
                - Classification logits [batch_size, 3]
                - Similarity scores [batch_size]
        """
        # Encode both reviews through shared network
        encoding1 = self.forward_once(review1, lengths1)
        encoding2 = self.forward_once(review2, lengths2)
        
        # Compute similarity
        similarity = self.compute_similarity(encoding1, encoding2)
        
        # Concatenate encodings for classification
        combined = torch.cat([encoding1, encoding2], dim=1)
        
        # Classify sentiment
        logits = self.classifier(combined)
        
        if return_similarity:
            return logits, similarity
        else:
            return logits
    
    def predict_sentiment(
        self,
        review: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        reference_review: Optional[torch.Tensor] = None,
        reference_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict sentiment for a single review.
        
        Args:
            review: Review token indices [batch_size, seq_length]
            lengths: Sequence lengths
            reference_review: Optional reference review for comparison
            reference_lengths: Reference review lengths
        
        Returns:
            Sentiment predictions [batch_size, 3] (logits or probabilities)
        """
        self.eval()
        with torch.no_grad():
            if reference_review is None:
                # Use the same review as reference (self-similarity baseline)
                reference_review = review
                reference_lengths = lengths
            
            logits = self.forward(
                review,
                reference_review,
                lengths,
                reference_lengths,
                return_similarity=False
            )
            
            # Return probabilities
            probabilities = F.softmax(logits, dim=1)
            
            return probabilities
    
    def get_embedding(
        self,
        review: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get review embedding (useful for clustering, visualization).
        
        Args:
            review: Review token indices [batch_size, seq_length]
            lengths: Sequence lengths
        
        Returns:
            Review embeddings [batch_size, output_dim]
        """
        self.eval()
        with torch.no_grad():
            encoding = self.forward_once(review, lengths)
            return encoding
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "architecture": "Siamese Network",
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "similarity_metric": self.similarity_metric,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 ** 2)  # Assuming float32
        }


def create_siamese_model(
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    similarity_metric: str = 'cosine',
    device: str = 'cpu'
) -> SiameseNetwork:
    """
    Factory function to create and initialize Siamese Network.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        similarity_metric: 'cosine' or 'euclidean'
        device: Device to place model on
    
    Returns:
        Initialized SiameseNetwork model
    """
    model = SiameseNetwork(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        similarity_metric=similarity_metric
    )
    
    model = model.to(device)
    
    # Print model info
    info = model.get_model_info()
    print("=" * 70)
    print("SIAMESE NETWORK MODEL CREATED")
    print("=" * 70)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    # Test the architecture
    print("\nTesting Siamese Network Architecture...\n")
    
    # Create model
    vocab_size = 10000
    batch_size = 8
    seq_length = 50
    
    model = create_siamese_model(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        device='cpu'
    )
    
    # Create dummy input
    review1 = torch.randint(0, vocab_size, (batch_size, seq_length))
    review2 = torch.randint(0, vocab_size, (batch_size, seq_length))
    lengths1 = torch.randint(10, seq_length, (batch_size,))
    lengths2 = torch.randint(10, seq_length, (batch_size,))
    
    # Forward pass
    print("\nForward pass test:")
    logits, similarity = model(review1, review2, lengths1, lengths2, return_similarity=True)
    print(f"  Logits shape: {logits.shape}")  # [batch_size, 3]
    print(f"  Similarity shape: {similarity.shape}")  # [batch_size]
    
    # Prediction test
    print("\nPrediction test:")
    predictions = model.predict_sentiment(review1, lengths1)
    print(f"  Predictions shape: {predictions.shape}")  # [batch_size, 3]
    print(f"  Sample prediction: {predictions[0]}")
    
    # Embedding test
    print("\nEmbedding test:")
    embeddings = model.get_embedding(review1, lengths1)
    print(f"  Embeddings shape: {embeddings.shape}")  # [batch_size, output_dim]
    
    print("\n✅ All tests passed!")
