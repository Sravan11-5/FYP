"""
Telugu Movie Reviews Dataset Collection and Preparation - Task 11
Collect reviews from database and prepare dataset for Siamese Network training
"""
import asyncio
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import random

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.database import connect_to_mongo, get_database, close_mongo_connection
from app.services.tmdb_collector import TMDBDataCollector
from app.services.twitter_collector import TwitterDataCollector
from app.services.database_storage import DatabaseStorageService


class TeluguReviewsDatasetPreparer:
    """Prepare Telugu movie reviews dataset for ML training"""
    
    def __init__(self, db):
        """
        Initialize dataset preparer
        
        Args:
            db: MongoDB database connection
        """
        self.db = db
        self.reviews_collection = db.reviews
        self.movies_collection = db.movies
        
        # Telugu Unicode ranges
        self.telugu_char_range = range(0x0C00, 0x0C7F)
        
    async def collect_reviews_from_database(self) -> List[Dict[str, Any]]:
        """
        Collect all Telugu movie reviews from database
        
        Returns:
            List of review documents
        """
        print("\n[11.1] COLLECTING REVIEWS FROM DATABASE")
        print("-" * 70)
        
        # Fetch all reviews
        cursor = self.reviews_collection.find({})
        reviews = await cursor.to_list(length=None)
        
        print(f"✅ Collected {len(reviews)} reviews from database")
        
        # Get statistics
        if reviews:
            with_sentiment = sum(1 for r in reviews if 'sentiment_score' in r)
            with_text = sum(1 for r in reviews if r.get('text', '').strip())
            
            print(f"   - Reviews with text: {with_text}")
            print(f"   - Reviews with sentiment: {with_sentiment}")
        
        return reviews
    
    def clean_review_text(self, text: str) -> str:
        """
        Clean review text by removing noise and irrelevant characters
        
        Args:
            text: Raw review text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove reviews that are too short (likely spam or incomplete)
        if len(text) < 10:
            return ""
        
        return text
    
    def contains_telugu_text(self, text: str) -> bool:
        """
        Check if text contains Telugu characters
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains Telugu characters
        """
        if not text:
            return False
        
        # Check for Telugu Unicode characters
        for char in text:
            if ord(char) in self.telugu_char_range:
                return True
        
        return False
    
    def tokenize_telugu_text(self, text: str) -> List[str]:
        """
        Simple tokenization for Telugu text
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Simple whitespace tokenization
        # For production, consider using:
        # - Indic NLP Library
        # - IndicBERT tokenizer
        # - Custom Telugu morphological analyzer
        
        tokens = text.split()
        
        # Remove very short tokens (likely noise)
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    async def prepare_dataset(
        self,
        min_review_length: int = 10,
        max_review_length: int = 1000,
        include_english: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare cleaned and processed dataset
        
        Args:
            min_review_length: Minimum review length in characters
            max_review_length: Maximum review length in characters
            include_english: Whether to include English reviews
            
        Returns:
            Dictionary with processed dataset
        """
        print("\n[11.2] CLEANING AND PROCESSING REVIEWS")
        print("-" * 70)
        
        # Collect reviews from database
        raw_reviews = await self.collect_reviews_from_database()
        
        cleaned_reviews = []
        stats = {
            'total_raw': len(raw_reviews),
            'too_short': 0,
            'too_long': 0,
            'no_text': 0,
            'cleaned': 0,
            'telugu_only': 0,
            'mixed_language': 0
        }
        
        for review in raw_reviews:
            text = review.get('text', '')
            
            if not text or not text.strip():
                stats['no_text'] += 1
                continue
            
            # Clean the text
            cleaned_text = self.clean_review_text(text)
            
            if not cleaned_text:
                stats['no_text'] += 1
                continue
            
            # Check length constraints
            if len(cleaned_text) < min_review_length:
                stats['too_short'] += 1
                continue
            
            if len(cleaned_text) > max_review_length:
                stats['too_long'] += 1
                continue
            
            # Check for Telugu content
            has_telugu = self.contains_telugu_text(cleaned_text)
            
            if has_telugu:
                stats['telugu_only'] += 1
            elif not include_english:
                continue
            else:
                stats['mixed_language'] += 1
            
            # Create processed review entry
            processed_review = {
                'review_id': str(review.get('_id', '')),
                'tweet_id': review.get('tweet_id', ''),
                'tmdb_id': review.get('tmdb_id', 0),
                'text': cleaned_text,
                'cleaned_text': cleaned_text,
                'sentiment_score': review.get('sentiment_score', 0.0),
                'likes': review.get('likes', 0),
                'retweets': review.get('retweets', 0),
                'language': 'te' if has_telugu else review.get('language', 'en'),
                'has_telugu': has_telugu
            }
            
            cleaned_reviews.append(processed_review)
            stats['cleaned'] += 1
        
        print(f"\n✅ Data Cleaning Statistics:")
        print(f"   Total raw reviews: {stats['total_raw']}")
        print(f"   Cleaned reviews: {stats['cleaned']}")
        print(f"   - Telugu content: {stats['telugu_only']}")
        print(f"   - Mixed/English: {stats['mixed_language']}")
        print(f"   Filtered out:")
        print(f"   - No text: {stats['no_text']}")
        print(f"   - Too short (<{min_review_length} chars): {stats['too_short']}")
        print(f"   - Too long (>{max_review_length} chars): {stats['too_long']}")
        
        return {
            'reviews': cleaned_reviews,
            'statistics': stats
        }
    
    def tokenize_dataset(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tokenize all reviews in dataset
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            Reviews with tokenized text
        """
        print("\n[11.3] TOKENIZING REVIEWS")
        print("-" * 70)
        
        tokenized_reviews = []
        total_tokens = 0
        vocab = set()
        
        for review in reviews:
            text = review.get('cleaned_text', '')
            tokens = self.tokenize_telugu_text(text)
            
            review['tokens'] = tokens
            review['token_count'] = len(tokens)
            
            tokenized_reviews.append(review)
            total_tokens += len(tokens)
            vocab.update(tokens)
        
        avg_tokens = total_tokens / len(tokenized_reviews) if tokenized_reviews else 0
        
        print(f"✅ Tokenization Complete:")
        print(f"   Total reviews tokenized: {len(tokenized_reviews)}")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Average tokens per review: {avg_tokens:.1f}")
        print(f"   Vocabulary size: {len(vocab)}")
        
        # Show token distribution
        token_counts = [r['token_count'] for r in tokenized_reviews]
        if token_counts:
            print(f"   Token count distribution:")
            print(f"   - Min: {min(token_counts)}")
            print(f"   - Max: {max(token_counts)}")
            print(f"   - Median: {sorted(token_counts)[len(token_counts)//2]}")
        
        return tokenized_reviews
    
    def split_dataset(
        self,
        reviews: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            reviews: List of review dictionaries
            train_ratio: Training set ratio (default: 0.8)
            val_ratio: Validation set ratio (default: 0.1)
            test_ratio: Test set ratio (default: 0.1)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_set, val_set, test_set)
        """
        print("\n[11.4] SPLITTING DATASET")
        print("-" * 70)
        
        # Verify ratios sum to 1.0
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
            "Train, val, and test ratios must sum to 1.0"
        
        # Shuffle reviews with fixed seed for reproducibility
        random.seed(random_seed)
        shuffled_reviews = reviews.copy()
        random.shuffle(shuffled_reviews)
        
        # Calculate split indices
        total = len(shuffled_reviews)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split the data
        train_set = shuffled_reviews[:train_end]
        val_set = shuffled_reviews[train_end:val_end]
        test_set = shuffled_reviews[val_end:]
        
        print(f"✅ Dataset Split Complete:")
        print(f"   Total reviews: {total}")
        print(f"   Training set: {len(train_set)} ({len(train_set)/total*100:.1f}%)")
        print(f"   Validation set: {len(val_set)} ({len(val_set)/total*100:.1f}%)")
        print(f"   Test set: {len(test_set)} ({len(test_set)/total*100:.1f}%)")
        
        # Show sentiment distribution per set
        for set_name, dataset in [('Training', train_set), ('Validation', val_set), ('Test', test_set)]:
            if dataset:
                sentiments = [r.get('sentiment_score', 0) for r in dataset]
                positive = sum(1 for s in sentiments if s > 0.2)
                negative = sum(1 for s in sentiments if s < -0.2)
                neutral = len(sentiments) - positive - negative
                
                print(f"\n   {set_name} set sentiment distribution:")
                print(f"   - Positive: {positive} ({positive/len(dataset)*100:.1f}%)")
                print(f"   - Negative: {negative} ({negative/len(dataset)*100:.1f}%)")
                print(f"   - Neutral: {neutral} ({neutral/len(dataset)*100:.1f}%)")
        
        return train_set, val_set, test_set
    
    def save_dataset(
        self,
        train_set: List[Dict],
        val_set: List[Dict],
        test_set: List[Dict],
        output_dir: str = "data/telugu_reviews"
    ):
        """
        Save dataset splits to JSON files
        
        Args:
            train_set: Training set
            val_set: Validation set
            test_set: Test set
            output_dir: Output directory path
        """
        print("\n[11.5] SAVING DATASET")
        print("-" * 70)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each split
        splits = {
            'train': train_set,
            'validation': val_set,
            'test': test_set
        }
        
        for split_name, data in splits.items():
            file_path = output_path / f"{split_name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Saved {split_name} set: {file_path}")
            print(f"   - {len(data)} reviews")
        
        # Save dataset statistics
        stats_path = output_path / "dataset_stats.json"
        stats = {
            'total_reviews': len(train_set) + len(val_set) + len(test_set),
            'train_size': len(train_set),
            'validation_size': len(val_set),
            'test_size': len(test_set),
            'split_ratio': '80/10/10'
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved dataset statistics: {stats_path}")


async def main():
    """Main function to prepare Telugu reviews dataset"""
    
    print("=" * 80)
    print("TASK 11: COLLECT AND PREPARE TELUGU MOVIE REVIEWS DATASET")
    print("=" * 80)
    
    # Connect to database
    print("\n🔌 Connecting to MongoDB...")
    await connect_to_mongo()
    db = get_database()
    print("✅ Connected to MongoDB")
    
    # Initialize dataset preparer
    preparer = TeluguReviewsDatasetPreparer(db)
    
    # Check current database status
    print("\n📊 Checking current database status...")
    reviews_count = await db.reviews.count_documents({})
    movies_count = await db.movies.count_documents({})
    print(f"   Current movies in database: {movies_count}")
    print(f"   Current reviews in database: {reviews_count}")
    
    if reviews_count == 0:
        print("\n⚠️  No reviews found in database!")
        print("   To collect reviews, you need to:")
        print("   1. Run data collection script (Task 9)")
        print("   2. Or manually collect reviews using Twitter API")
        print("\n   For this test, we'll create a sample dataset...")
        
        # Close connection
        await close_mongo_connection()
        return
    
    # Prepare dataset
    dataset = await preparer.prepare_dataset(
        min_review_length=10,
        max_review_length=1000,
        include_english=True
    )
    
    reviews = dataset['reviews']
    
    if len(reviews) < 10:
        print(f"\n⚠️  Only {len(reviews)} reviews available (need 50+ for meaningful training)")
        print("   Consider collecting more reviews before training the model.")
    
    # Tokenize reviews
    tokenized_reviews = preparer.tokenize_dataset(reviews)
    
    # Split dataset
    train_set, val_set, test_set = preparer.split_dataset(
        tokenized_reviews,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
    )
    
    # Save dataset
    preparer.save_dataset(train_set, val_set, test_set)
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ TASK 11 COMPLETE - DATASET PREPARATION SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Subtask 11.1: Collected {len(reviews)} reviews from database")
    print(f"✅ Subtask 11.2: Cleaned and processed reviews")
    print(f"   - Removed noise, URLs, mentions")
    print(f"   - Filtered by length constraints")
    
    print(f"✅ Subtask 11.3: Tokenized all reviews")
    print(f"   - Average tokens per review: {sum(r['token_count'] for r in tokenized_reviews) / len(tokenized_reviews) if tokenized_reviews else 0:.1f}")
    
    print(f"✅ Subtask 11.4: Split dataset (80/10/10)")
    print(f"   - Training: {len(train_set)} reviews")
    print(f"   - Validation: {len(val_set)} reviews")
    print(f"   - Test: {len(test_set)} reviews")
    
    print(f"\n📁 Dataset saved to: data/telugu_reviews/")
    print(f"   - train.json ({len(train_set)} reviews)")
    print(f"   - validation.json ({len(val_set)} reviews)")
    print(f"   - test.json ({len(test_set)} reviews)")
    print(f"   - dataset_stats.json")
    
    if len(reviews) >= 50:
        print(f"\n🎉 Dataset is ready for Siamese Network training!")
    else:
        print(f"\n⚠️  Collect more reviews for better model performance")
        print(f"   Current: {len(reviews)} reviews")
        print(f"   Recommended: 1,000+ reviews")
    
    # Close database connection
    await close_mongo_connection()


if __name__ == "__main__":
    asyncio.run(main())
