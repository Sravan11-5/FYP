"""
Data preparation and loading for Siamese Network training
=========================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import random
import numpy as np


class SiameseReviewDataset(Dataset):
    """
    Dataset for Siamese Network training.
    
    Creates pairs of reviews with labels:
    - Similar pairs (same sentiment): label = 0
    - Dissimilar pairs (different sentiment): label = 1
    
    Args:
        reviews: List of review dictionaries
        tokenizer: Tokenizer for encoding text
        max_length: Maximum sequence length
        num_pairs_per_review: Number of pairs to create per review
    """
    
    def __init__(
        self,
        reviews: List[Dict],
        tokenizer,
        max_length: int = 50,
        num_pairs_per_review: int = 2
    ):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_pairs_per_review = num_pairs_per_review
        
        # Group reviews by sentiment
        self.sentiment_groups = {}
        for review in reviews:
            sentiment = review['sentiment']
            if sentiment not in self.sentiment_groups:
                self.sentiment_groups[sentiment] = []
            self.sentiment_groups[sentiment].append(review)
        
        # Create pairs
        self.pairs = self._create_pairs()
        
        # Sentiment to index mapping
        self.sentiment_to_idx = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
    
    def _create_pairs(self) -> List[Tuple]:
        """Create review pairs (similar and dissimilar)."""
        pairs = []
        sentiments = list(self.sentiment_groups.keys())
        
        # Create similar pairs (same sentiment)
        for sentiment, group in self.sentiment_groups.items():
            if len(group) < 2:
                continue
            
            for i in range(len(group)):
                for _ in range(self.num_pairs_per_review // 2):
                    # Randomly select another review with same sentiment
                    j = random.choice([idx for idx in range(len(group)) if idx != i])
                    pairs.append((group[i], group[j], 0, sentiment))  # label=0 for similar
        
        # Create dissimilar pairs (different sentiment)
        for i, sent1 in enumerate(sentiments):
            for sent2 in sentiments[i+1:]:
                group1 = self.sentiment_groups[sent1]
                group2 = self.sentiment_groups[sent2]
                
                # Sample pairs
                num_pairs = min(len(group1), len(group2), 
                               len(group1) * self.num_pairs_per_review // 2)
                
                for _ in range(num_pairs):
                    review1 = random.choice(group1)
                    review2 = random.choice(group2)
                    # For dissimilar pairs, use first review's sentiment as label
                    pairs.append((review1, review2, 1, sent1))  # label=1 for dissimilar
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a pair of reviews.
        
        Returns:
            Dictionary with:
            - review1: Token indices [seq_length]
            - review2: Token indices [seq_length]
            - length1: Actual length of review1
            - length2: Actual length of review2
            - similarity_label: 0=similar, 1=dissimilar
            - sentiment_label: Sentiment class (0=neg, 1=neu, 2=pos)
        """
        review1, review2, similarity_label, sentiment = self.pairs[idx]
        
        # Encode reviews
        tokens1, length1 = self.tokenizer.encode(review1['text'], self.max_length)
        tokens2, length2 = self.tokenizer.encode(review2['text'], self.max_length)
        
        # Sentiment label
        sentiment_label = self.sentiment_to_idx[sentiment]
        
        return {
            'review1': torch.tensor(tokens1, dtype=torch.long),
            'review2': torch.tensor(tokens2, dtype=torch.long),
            'length1': torch.tensor(length1, dtype=torch.long),
            'length2': torch.tensor(length2, dtype=torch.long),
            'similarity_label': torch.tensor(similarity_label, dtype=torch.float),
            'sentiment_label': torch.tensor(sentiment_label, dtype=torch.long)
        }


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Handles variable-length sequences by padding.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched tensors
    """
    return {
        'review1': torch.stack([item['review1'] for item in batch]),
        'review2': torch.stack([item['review2'] for item in batch]),
        'length1': torch.stack([item['length1'] for item in batch]),
        'length2': torch.stack([item['length2'] for item in batch]),
        'similarity_label': torch.stack([item['similarity_label'] for item in batch]),
        'sentiment_label': torch.stack([item['sentiment_label'] for item in batch])
    }


def create_data_loaders(
    train_reviews: List[Dict],
    val_reviews: List[Dict],
    tokenizer,
    batch_size: int = 32,
    max_length: int = 50,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        train_reviews: Training reviews
        val_reviews: Validation reviews
        tokenizer: Tokenizer for encoding
        batch_size: Batch size
        max_length: Max sequence length
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = SiameseReviewDataset(
        train_reviews,
        tokenizer,
        max_length=max_length,
        num_pairs_per_review=2
    )
    
    val_dataset = SiameseReviewDataset(
        val_reviews,
        tokenizer,
        max_length=max_length,
        num_pairs_per_review=1
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ Train dataset: {len(train_dataset)} pairs")
    print(f"✓ Val dataset: {len(val_dataset)} pairs")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data preparation
    print("Testing data preparation...\n")
    
    # Dummy reviews
    reviews = [
        {'text': 'చాలా బాగుంది సినిమా', 'sentiment': 'positive', 'rating': 8},
        {'text': 'సూపర్ మూవీ అద్భుతం', 'sentiment': 'positive', 'rating': 9},
        {'text': 'బోరింగ్ చెడ్డది', 'sentiment': 'negative', 'rating': 3},
        {'text': 'వేస్ట్ టైం', 'sentiment': 'negative', 'rating': 2},
        {'text': 'ఓకే టైప్', 'sentiment': 'neutral', 'rating': 5},
        {'text': 'డిసెంట్ వాచ్', 'sentiment': 'neutral', 'rating': 6},
    ]
    
    # Simple tokenizer
    class DummyTokenizer:
        def __init__(self):
            self.word2idx = {"<PAD>": 0, "<UNK>": 1}
            idx = 2
            for review in reviews:
                for word in review['text'].split():
                    if word not in self.word2idx:
                        self.word2idx[word] = idx
                        idx += 1
        
        def encode(self, text, max_length=20):
            tokens = [self.word2idx.get(w, 1) for w in text.split()]
            length = min(len(tokens), max_length)
            if len(tokens) < max_length:
                tokens += [0] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
            return tokens, length
    
    tokenizer = DummyTokenizer()
    
    # Create dataset
    dataset = SiameseReviewDataset(reviews, tokenizer, max_length=20, num_pairs_per_review=2)
    
    print(f"Dataset size: {len(dataset)} pairs")
    print(f"Sentiment groups: {list(dataset.sentiment_groups.keys())}")
    
    # Sample a few pairs
    print("\nSample pairs:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nPair {i+1}:")
        print(f"  Review 1 tokens: {sample['review1'][:10].tolist()}")
        print(f"  Review 2 tokens: {sample['review2'][:10].tolist()}")
        print(f"  Length 1: {sample['length1'].item()}")
        print(f"  Length 2: {sample['length2'].item()}")
        print(f"  Similarity label: {sample['similarity_label'].item()} ({'similar' if sample['similarity_label'] == 0 else 'dissimilar'})")
        print(f"  Sentiment label: {sample['sentiment_label'].item()}")
    
    # Test data loader
    print("\n\nTesting DataLoader:")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_batch)
    
    for batch_idx, batch in enumerate(loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Review1 shape: {batch['review1'].shape}")
        print(f"  Review2 shape: {batch['review2'].shape}")
        print(f"  Length1 shape: {batch['length1'].shape}")
        print(f"  Length2 shape: {batch['length2'].shape}")
        print(f"  Similarity labels: {batch['similarity_label'].tolist()}")
        print(f"  Sentiment labels: {batch['sentiment_label'].tolist()}")
        break
    
    print("\n✅ Data preparation working correctly!")
