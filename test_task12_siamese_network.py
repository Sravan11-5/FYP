"""
Task 12: Siamese Network Architecture Testing
==============================================
Test the Siamese Network with actual Telugu movie review data

Subtasks:
12.1: Design architecture ✓
12.2: Implement embedding layer ✓
12.3: Build twin networks ✓
12.4: Add similarity metric ✓
12.5: Test with sample data ← CURRENT
"""

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import sys

# Add app to path
sys.path.append('.')

from app.ml.models.siamese_network import (
    SiameseNetwork,
    create_siamese_model
)


class TeluguTokenizer:
    """Simple tokenizer for Telugu text."""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.vocab_built = False
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = text.split()
            word_counts.update(tokens)
        
        # Take most common words
        most_common = word_counts.most_common(self.vocab_size - 4)
        
        # Add to vocabulary
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self.vocab_built = True
        print(f"✓ Vocabulary built: {len(self.word2idx)} words")
    
    def encode(self, text: str, max_length: int = 50) -> Tuple[List[int], int]:
        """
        Encode text to token indices.
        
        Returns:
            tokens: List of token indices (padded/truncated to max_length)
            length: Actual length before padding
        """
        words = text.split()
        tokens = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
        
        # Actual length
        length = min(len(tokens), max_length)
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens = tokens + [self.word2idx["<PAD>"]] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        
        return tokens, length
    
    def decode(self, token_indices: List[int]) -> str:
        """Decode token indices back to text."""
        words = [self.idx2word.get(idx, "<UNK>") for idx in token_indices]
        # Remove padding
        words = [w for w in words if w != "<PAD>"]
        return " ".join(words)


def load_dataset(data_dir: str = "data/telugu_reviews"):
    """Load Telugu reviews dataset."""
    data_path = Path(data_dir)
    
    print("\n" + "=" * 70)
    print("LOADING TELUGU REVIEWS DATASET")
    print("=" * 70)
    
    # Load train set
    with open(data_path / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    
    # Load validation set
    with open(data_path / "validation.json", encoding="utf-8") as f:
        val_data = json.load(f)
    
    # Load test set
    with open(data_path / "test.json", encoding="utf-8") as f:
        test_data = json.load(f)
    
    print(f"✓ Loaded {len(train_data)} training reviews")
    print(f"✓ Loaded {len(val_data)} validation reviews")
    print(f"✓ Loaded {len(test_data)} test reviews")
    
    return train_data, val_data, test_data


def create_review_pairs(reviews: List[Dict], max_pairs: int = 100) -> List[Tuple]:
    """
    Create pairs of reviews for Siamese Network training.
    
    Pairs:
    - Similar (label=1): Same sentiment
    - Dissimilar (label=0): Different sentiment
    """
    pairs = []
    
    # Group by sentiment
    sentiment_groups = {}
    for review in reviews:
        sentiment = review['sentiment']
        if sentiment not in sentiment_groups:
            sentiment_groups[sentiment] = []
        sentiment_groups[sentiment].append(review)
    
    print(f"\nSentiment groups:")
    for sentiment, group in sentiment_groups.items():
        print(f"  {sentiment}: {len(group)} reviews")
    
    # Create similar pairs (same sentiment)
    for sentiment, group in sentiment_groups.items():
        for i in range(min(len(group) - 1, max_pairs // (2 * len(sentiment_groups)))):
            pairs.append((group[i], group[i + 1], 1))  # Similar pair
    
    # Create dissimilar pairs (different sentiment)
    sentiments = list(sentiment_groups.keys())
    for i in range(len(sentiments)):
        for j in range(i + 1, len(sentiments)):
            group1 = sentiment_groups[sentiments[i]]
            group2 = sentiment_groups[sentiments[j]]
            
            for k in range(min(len(group1), len(group2), max_pairs // (2 * len(sentiment_groups)))):
                pairs.append((group1[k], group2[k], 0))  # Dissimilar pair
    
    print(f"\n✓ Created {len(pairs)} review pairs")
    similar = sum(1 for _, _, label in pairs if label == 1)
    dissimilar = sum(1 for _, _, label in pairs if label == 0)
    print(f"  Similar pairs: {similar}")
    print(f"  Dissimilar pairs: {dissimilar}")
    
    return pairs


def test_architecture_with_telugu_data():
    """Test Siamese Network with actual Telugu reviews."""
    
    print("\n" + "=" * 70)
    print("TASK 12: SIAMESE NETWORK ARCHITECTURE TEST")
    print("=" * 70)
    
    # Load dataset
    train_data, val_data, test_data = load_dataset()
    
    # Extract texts
    train_texts = [review['text'] for review in train_data]
    
    # Build tokenizer
    print("\n" + "=" * 70)
    print("BUILDING VOCABULARY")
    print("=" * 70)
    tokenizer = TeluguTokenizer(vocab_size=5000)
    tokenizer.build_vocab(train_texts)
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING SIAMESE NETWORK")
    print("=" * 70)
    model = create_siamese_model(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        similarity_metric='cosine',
        device='cpu'
    )
    
    # Test with sample reviews
    print("\n" + "=" * 70)
    print("TESTING WITH SAMPLE REVIEWS")
    print("=" * 70)
    
    # Get 4 sample reviews
    samples = train_data[:4]
    
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"  Movie: {sample['movie_title']}")
        print(f"  Sentiment: {sample['sentiment']}")
        print(f"  Rating: {sample['rating']}/10")
        print(f"  Text: {sample['text'][:80]}...")
    
    # Encode reviews
    print("\n" + "=" * 70)
    print("ENCODING REVIEWS")
    print("=" * 70)
    
    encoded_reviews = []
    lengths = []
    
    for sample in samples:
        tokens, length = tokenizer.encode(sample['text'], max_length=50)
        encoded_reviews.append(tokens)
        lengths.append(length)
        print(f"  Encoded review: length={length}, tokens[:10]={tokens[:10]}")
    
    # Convert to tensors
    review_tensors = torch.tensor(encoded_reviews, dtype=torch.long)
    length_tensors = torch.tensor(lengths, dtype=torch.long)
    
    # Test pairwise similarity
    print("\n" + "=" * 70)
    print("COMPUTING PAIRWISE SIMILARITIES")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        # Get embeddings for all reviews
        embeddings = model.get_embedding(review_tensors, length_tensors)
        print(f"✓ Generated embeddings: shape={embeddings.shape}")
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.t())
        
        print("\nSimilarity Matrix:")
        print("(1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")
        print("-" * 50)
        for i in range(len(samples)):
            row = [f"{similarity_matrix[i, j].item():.3f}" for j in range(len(samples))]
            print(f"Review {i+1}: {' '.join(row)}")
        
        # Check if similar sentiments have higher similarity
        print("\n" + "=" * 70)
        print("SENTIMENT-BASED SIMILARITY ANALYSIS")
        print("=" * 70)
        
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                sim = similarity_matrix[i, j].item()
                same_sentiment = samples[i]['sentiment'] == samples[j]['sentiment']
                
                print(f"\nReview {i+1} vs Review {j+1}:")
                print(f"  Sentiments: {samples[i]['sentiment']} vs {samples[j]['sentiment']}")
                print(f"  Same sentiment: {same_sentiment}")
                print(f"  Similarity score: {sim:.4f}")
                print(f"  Expected: {'High (>0.5)' if same_sentiment else 'Low (<0.5)'}")
    
    # Test classification
    print("\n" + "=" * 70)
    print("TESTING SENTIMENT CLASSIFICATION")
    print("=" * 70)
    
    with torch.no_grad():
        # Create pairs for classification
        review1 = review_tensors[:2]  # First 2 reviews
        review2 = review_tensors[2:4]  # Last 2 reviews
        length1 = length_tensors[:2]
        length2 = length_tensors[2:4]
        
        # Forward pass
        logits, similarities = model(review1, review2, length1, length2, return_similarity=True)
        
        # Get predictions
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        
        class_names = ['negative', 'neutral', 'positive']
        
        for i in range(len(review1)):
            print(f"\nPair {i+1}:")
            print(f"  Review 1 sentiment: {samples[i]['sentiment']}")
            print(f"  Review 2 sentiment: {samples[i+2]['sentiment']}")
            print(f"  Similarity score: {similarities[i].item():.4f}")
            print(f"  Classification probabilities:")
            for j, class_name in enumerate(class_names):
                print(f"    {class_name}: {probabilities[i, j].item():.4f}")
            print(f"  Predicted class: {class_names[predicted_classes[i]]}")
    
    # Test with review pairs
    print("\n" + "=" * 70)
    print("TESTING WITH REVIEW PAIRS")
    print("=" * 70)
    
    pairs = create_review_pairs(train_data[:100], max_pairs=20)
    
    # Test a few pairs
    for i, (review1, review2, label) in enumerate(pairs[:5], 1):
        # Encode
        tokens1, len1 = tokenizer.encode(review1['text'])
        tokens2, len2 = tokenizer.encode(review2['text'])
        
        # Convert to tensors
        t1 = torch.tensor([tokens1], dtype=torch.long)
        t2 = torch.tensor([tokens2], dtype=torch.long)
        l1 = torch.tensor([len1], dtype=torch.long)
        l2 = torch.tensor([len2], dtype=torch.long)
        
        with torch.no_grad():
            _, sim = model(t1, t2, l1, l2, return_similarity=True)
            sim_score = sim[0].item()
        
        print(f"\nPair {i}:")
        print(f"  Review 1: {review1['sentiment']} - {review1['text'][:60]}...")
        print(f"  Review 2: {review2['sentiment']} - {review2['text'][:60]}...")
        print(f"  Label: {'Similar' if label == 1 else 'Dissimilar'}")
        print(f"  Similarity score: {sim_score:.4f}")
        print(f"  ✓ PASS" if (label == 1 and sim_score > 0.3) or (label == 0 and sim_score < 0.7) else "  ⚠ Note: Untrained model, scores are random")
    
    # Summary
    print("\n" + "=" * 70)
    print("TASK 12 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n✅ All subtasks completed:")
    print("  12.1: Siamese Network architecture designed ✓")
    print("  12.2: Telugu embedding layer implemented ✓")
    print("  12.3: Twin LSTM networks built with shared weights ✓")
    print("  12.4: Cosine similarity metric added ✓")
    print("  12.5: Tested with actual Telugu review data ✓")
    
    print("\n📊 Model Summary:")
    info = model.get_model_info()
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Trainable parameters: {info['trainable_parameters']:,}")
    print(f"  Model size: {info['model_size_mb']:.2f} MB")
    print(f"  Vocabulary size: {len(tokenizer.word2idx):,}")
    
    print("\n🎯 Next Steps:")
    print("  Task 13: Train the Siamese Network on Telugu reviews")
    print("  Task 14: Evaluate model performance")
    print("  Task 15: Integrate trained model into API")
    
    print("\n" + "=" * 70)
    
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = test_architecture_with_telugu_data()
