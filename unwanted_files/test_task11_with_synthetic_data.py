"""
Task 11: Telugu Movie Reviews Dataset Preparation with Synthetic Data
====================================================================
Subtasks:
11.1: Collect reviews from database (or generate synthetic data if empty)
11.2: Clean data (remove URLs, mentions, special characters)
11.3: Tokenize Telugu text
11.4: Split dataset (80/10/10)
"""

import asyncio
import os
import json
import re
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()


class TeluguReviewsDatasetPreparer:
    """Prepares Telugu movie reviews dataset for sentiment analysis model training."""
    
    def __init__(self, mongodb_uri: str):
        """Initialize dataset preparer with MongoDB connection."""
        self.client = AsyncIOMotorClient(mongodb_uri)
        self.db = self.client["telugu_movie_db"]
        self.stats = {
            "total_raw": 0,
            "cleaned": 0,
            "too_short": 0,
            "too_long": 0,
            "no_text": 0,
            "telugu_only": 0,
            "mixed_language": 0
        }
    
    # Synthetic Telugu review templates (for testing when no real data)
    TELUGU_POSITIVE_TEMPLATES = [
        "చాలా బాగుంది ఈ సినిమా. {hero} నటన అద్భుతం. కథ చాలా బాగుంది.",
        "అద్భుతమైన సినిమా! {director} దర్శకత్వం సూపర్. ప్రతి సీన్ బాగుంది.",
        "ఎంత బావుంటుంది ఈ మూవీ! {hero} మరియు {heroine} జోడి చాలా బాగుంది.",
        "మరువలేని సినిమా. సంగీతం అద్భుతం. కథ చాలా touching.",
        "పర్ఫెక్ట్ ఎంటర్టైనర్! అన్ని విషయాల్లో 100/100.",
        "బ్లాక్‌బస్టర్ మూవీ! {hero} నటన చూసి ఆశ్చర్యపోయాను.",
        "అద్భుతమైన కథ! భావోద్వేగాలు చాలా బాగున్నాయి.",
        "సూపర్ సినిమా! కుటుంబంతో చూడవచ్చు. చాలా ఆనందంగా ఉంది.",
        "మాస్ మూవీ! యాక్షన్ సీన్లు అద్భుతం. థియేటర్ షేక్ అయింది.",
        "అద్భుతమైన సినిమా! {director} గారికి congratulations."
    ]
    
    TELUGU_NEGATIVE_TEMPLATES = [
        "చాలా బోరింగ్ సినిమా. కథ బాగోలేదు. {hero} నటన కూడా ఓకే.",
        "సమయం వృధా. కథ లేదు, సీన్లు లేవు. పూర్తిగా డిసాపాయింట్‌మెంట్.",
        "ఎంత చెడ్డ మూవీ! {director} దర్శకత్వం బాగోలేదు.",
        "వేస్ట్ ఆఫ్ మనీ. క్లైమాక్స్ చాలా చెడ్డది. ఏమీ అర్థం కాలేదు.",
        "చాలా నిరాశపరిచింది. అంచనాలు ఎక్కువ, కానీ delivery చాలా తక్కువ.",
        "{hero} నటన కూడా రక్షించలేకపోయింది. కథ చాలా weak.",
        "మొదటి హాఫ్ ఓకే కానీ రెండవ హాఫ్ చూడలేకపోయాను.",
        "ప్రొడక్షన్ వాల్యూస్ మంచివి కానీ కథ లేదు. పూర్తిగా డిసాపాయింట్‌మెంట్.",
        "ఓవర్‌హైప్డ్ మూవీ. నిజంగా ఏమీ స్పెషల్ లేదు.",
        "టైం పాస్ కూడా కాదు. థియేటర్ నుంచి వాక్ అవుట్ చేసుకున్నాను."
    ]
    
    TELUGU_NEUTRAL_TEMPLATES = [
        "ఓకే టైప్ మూవీ. {hero} నటన బాగుంది కానీ కథ average.",
        "చూడవచ్చు. స్పెషల్ ఏమీ లేదు కానీ చెడ్డది కూడా కాదు.",
        "సగటు సినిమా. కొన్ని సీన్లు బాగున్నాయి కొన్ని బాగోలేదు.",
        "మిక్స్డ్ ఫీలింగ్స్. మొదటి హాఫ్ బాగుంది, రెండవ హాఫ్ స్లో.",
        "పర్వాలేదు టైప్. {director} నుంచి బెటర్ expect చేశాం.",
        "టైం పాస్ మూవీ. అంచనాలు లేకుండా చూస్తే enjoy చేయవచ్చు.",
        "యావరేజ్ ఎంటర్టైనర్. {hero} మరియు {heroine} కెమిస్ట్రీ బాగుంది.",
        "ఓకే సినిమా. థియేటర్‌లో చూడకపోయినా పర్వాలేదు.",
        "డిసెంట్ వాచ్. కథ predictable కానీ execution బాగుంది.",
        "సగటు స్థాయి. ఒకసారి చూడవచ్చు కానీ రెండవసారి చూడలేం."
    ]
    
    TELUGU_MOVIES = [
        {"name": "బాహుబలి", "hero": "ప్రభాస్", "heroine": "అనుష్క", "director": "రాజమౌళి"},
        {"name": "RRR", "hero": "రామ్ చరణ్", "heroine": "ఆలియా భట్", "director": "రాజమౌళి"},
        {"name": "పుష్ప", "hero": "అల్లు అర్జున్", "heroine": "రష్మిక", "director": "సుకుమార్"},
        {"name": "సీతా రామం", "hero": "దుల్కర్ సల్మాన్", "heroine": "మృణాల్", "director": "హనుమాన్"},
        {"name": "అర్జున్ రెడ్డి", "hero": "విజయ్ దేవరకొండ", "heroine": "శాలిని", "director": "సంధీప్ వంగ"},
        {"name": "రంగస్థలం", "hero": "రామ్ చరణ్", "heroine": "సమంత", "director": "సుకుమార్"},
        {"name": "ఈగ", "hero": "నాని", "heroine": "సమంత", "director": "రాజమౌళి"},
        {"name": "జెర్సీ", "hero": "నాని", "heroine": "శ్రద్ధా శ్రీనాథ్", "director": "గౌతమ్"},
        {"name": "అల వైకుంఠపురములో", "hero": "అల్లు అర్జున్", "heroine": "పూజా హెగ్డే", "director": "త్రివిక్రమ్"},
        {"name": "మహర్షి", "hero": "మహేష్ బాబు", "heroine": "పూజా హెగ్డే", "director": "వంశీ"}
    ]
    
    async def generate_synthetic_reviews(self, count: int = 500) -> List[Dict[str, Any]]:
        """Generate synthetic Telugu movie reviews for testing."""
        print(f"Generating {count} synthetic Telugu movie reviews...")
        
        reviews = []
        for i in range(count):
            # Random sentiment distribution: 40% positive, 30% negative, 30% neutral
            sentiment_roll = random.random()
            if sentiment_roll < 0.4:
                template = random.choice(self.TELUGU_POSITIVE_TEMPLATES)
                sentiment = "positive"
                rating = random.randint(7, 10)
            elif sentiment_roll < 0.7:
                template = random.choice(self.TELUGU_NEGATIVE_TEMPLATES)
                sentiment = "negative"
                rating = random.randint(1, 4)
            else:
                template = random.choice(self.TELUGU_NEUTRAL_TEMPLATES)
                sentiment = "neutral"
                rating = random.randint(5, 7)
            
            # Pick random movie
            movie = random.choice(self.TELUGU_MOVIES)
            
            # Generate review text
            text = template.format(
                hero=movie["hero"],
                heroine=movie["heroine"],
                director=movie["director"]
            )
            
            # Add some noise (URLs, mentions, hashtags) randomly
            if random.random() < 0.2:
                text += f" https://example.com/review/{i}"
            if random.random() < 0.15:
                text += f" @{movie['hero'].replace(' ', '')}"
            if random.random() < 0.25:
                text += f" #{movie['name'].replace(' ', '')}"
            
            review = {
                "review_id": f"synthetic_{i}",
                "movie_title": movie["name"],
                "text": text,
                "sentiment": sentiment,
                "rating": rating,
                "language": "te",
                "source": "synthetic",
                "created_at": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
            }
            reviews.append(review)
        
        print(f"✓ Generated {len(reviews)} synthetic reviews")
        return reviews
    
    async def collect_reviews_from_database(self) -> List[Dict[str, Any]]:
        """
        Subtask 11.1: Collect reviews from database.
        If no reviews exist, generate synthetic data for testing.
        """
        print("\n=== Subtask 11.1: Collecting Reviews ===")
        
        # Check database for reviews
        reviews_count = await self.db.reviews.count_documents({})
        print(f"Reviews in database: {reviews_count}")
        
        if reviews_count == 0:
            print("⚠ No reviews in database. Generating synthetic data for testing...")
            reviews = await self.generate_synthetic_reviews(500)
            
            # Optionally store synthetic reviews in database
            if reviews:
                await self.db.reviews.insert_many(reviews)
                print(f"✓ Stored {len(reviews)} synthetic reviews in database")
        else:
            # Collect from database
            reviews = []
            cursor = self.db.reviews.find({})
            async for review in cursor:
                reviews.append(review)
            print(f"✓ Collected {len(reviews)} reviews from database")
        
        self.stats["total_raw"] = len(reviews)
        return reviews
    
    @staticmethod
    def clean_review_text(text: str) -> str:
        """
        Subtask 11.2: Clean review text.
        Remove URLs, mentions, hashtags, and normalize whitespace.
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text (e.g., #RRR -> RRR)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def contains_telugu_text(text: str) -> bool:
        """Check if text contains Telugu characters."""
        telugu_range = range(0x0C00, 0x0C7F)  # Telugu Unicode range
        return any(ord(char) in telugu_range for char in text)
    
    @staticmethod
    def tokenize_telugu_text(text: str) -> List[str]:
        """
        Subtask 11.3: Tokenize Telugu text.
        Simple whitespace tokenization (suitable for Telugu).
        """
        if not text:
            return []
        
        # Split on whitespace
        tokens = text.split()
        
        # Filter out very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    async def prepare_dataset(
        self,
        reviews: List[Dict[str, Any]],
        min_length: int = 10,
        max_length: int = 1000
    ) -> List[Dict[str, Any]]:
        """Process and clean reviews for dataset."""
        print("\n=== Subtask 11.2-11.3: Cleaning and Tokenizing ===")
        
        processed_reviews = []
        
        for review in reviews:
            text = review.get("text", "")
            
            if not text:
                self.stats["no_text"] += 1
                continue
            
            # Clean text
            cleaned_text = self.clean_review_text(text)
            
            # Check length
            if len(cleaned_text) < min_length:
                self.stats["too_short"] += 1
                continue
            
            if len(cleaned_text) > max_length:
                self.stats["too_long"] += 1
                continue
            
            # Tokenize
            tokens = self.tokenize_telugu_text(cleaned_text)
            
            if len(tokens) < 3:  # Need at least 3 tokens
                self.stats["too_short"] += 1
                continue
            
            # Check if contains Telugu
            has_telugu = self.contains_telugu_text(cleaned_text)
            if has_telugu:
                self.stats["telugu_only"] += 1
            else:
                self.stats["mixed_language"] += 1
            
            # Create processed review
            processed = {
                "review_id": review.get("review_id", review.get("_id")),
                "movie_title": review.get("movie_title", "Unknown"),
                "text": cleaned_text,
                "tokens": tokens,
                "token_count": len(tokens),
                "sentiment": review.get("sentiment", "unknown"),
                "rating": review.get("rating", 0),
                "has_telugu": has_telugu,
                "source": review.get("source", "unknown"),
                "created_at": review.get("created_at", "")
            }
            
            processed_reviews.append(processed)
            self.stats["cleaned"] += 1
        
        print(f"✓ Processed {len(processed_reviews)} reviews")
        print(f"  - Telugu reviews: {self.stats['telugu_only']}")
        print(f"  - Mixed language: {self.stats['mixed_language']}")
        print(f"  - Too short: {self.stats['too_short']}")
        print(f"  - Too long: {self.stats['too_long']}")
        print(f"  - No text: {self.stats['no_text']}")
        
        return processed_reviews
    
    @staticmethod
    def split_dataset(
        reviews: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Subtask 11.4: Split dataset into train/validation/test sets.
        """
        print("\n=== Subtask 11.4: Splitting Dataset ===")
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
        
        # Shuffle with fixed seed for reproducibility
        random.seed(random_seed)
        shuffled = reviews.copy()
        random.shuffle(shuffled)
        
        # Calculate split indices
        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split
        train_set = shuffled[:train_end]
        val_set = shuffled[train_end:val_end]
        test_set = shuffled[val_end:]
        
        print(f"✓ Dataset split:")
        print(f"  - Training: {len(train_set)} ({len(train_set)/total*100:.1f}%)")
        print(f"  - Validation: {len(val_set)} ({len(val_set)/total*100:.1f}%)")
        print(f"  - Test: {len(test_set)} ({len(test_set)/total*100:.1f}%)")
        
        return train_set, val_set, test_set
    
    @staticmethod
    def save_dataset(
        train_set: List[Dict],
        val_set: List[Dict],
        test_set: List[Dict],
        output_dir: str = "data/telugu_reviews"
    ):
        """Save dataset splits to JSON files."""
        print(f"\n=== Saving Dataset to {output_dir} ===")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        with open(output_path / "train.json", "w", encoding="utf-8") as f:
            json.dump(train_set, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved train.json ({len(train_set)} reviews)")
        
        with open(output_path / "validation.json", "w", encoding="utf-8") as f:
            json.dump(val_set, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved validation.json ({len(val_set)} reviews)")
        
        with open(output_path / "test.json", "w", encoding="utf-8") as f:
            json.dump(test_set, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved test.json ({len(test_set)} reviews)")
        
        # Save dataset statistics
        stats = {
            "total_reviews": len(train_set) + len(val_set) + len(test_set),
            "train_count": len(train_set),
            "validation_count": len(val_set),
            "test_count": len(test_set),
            "splits": {
                "train_ratio": len(train_set) / (len(train_set) + len(val_set) + len(test_set)),
                "val_ratio": len(val_set) / (len(train_set) + len(val_set) + len(test_set)),
                "test_ratio": len(test_set) / (len(train_set) + len(val_set) + len(test_set))
            },
            "created_at": datetime.now().isoformat()
        }
        
        with open(output_path / "dataset_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved dataset_stats.json")
    
    async def close(self):
        """Close MongoDB connection."""
        self.client.close()


async def main():
    """Main execution function for Task 11."""
    print("=" * 70)
    print("TASK 11: TELUGU MOVIE REVIEWS DATASET PREPARATION")
    print("=" * 70)
    
    # Initialize preparer
    mongo_uri = os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
    if not mongo_uri:
        print("❌ MONGODB_URL/MONGODB_URI not found in environment variables")
        return
    
    preparer = TeluguReviewsDatasetPreparer(mongo_uri)
    
    try:
        # Subtask 11.1: Collect reviews (or generate synthetic data)
        reviews = await preparer.collect_reviews_from_database()
        
        if not reviews:
            print("❌ No reviews available")
            return
        
        # Subtask 11.2-11.3: Clean and tokenize
        processed_reviews = await preparer.prepare_dataset(reviews)
        
        if not processed_reviews:
            print("❌ No reviews passed cleaning/tokenization")
            return
        
        # Subtask 11.4: Split dataset
        train_set, val_set, test_set = preparer.split_dataset(processed_reviews)
        
        # Save dataset
        preparer.save_dataset(train_set, val_set, test_set)
        
        # Print final summary
        print("\n" + "=" * 70)
        print("TASK 11 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Total reviews processed: {len(processed_reviews)}")
        print(f"Training set: {len(train_set)}")
        print(f"Validation set: {len(val_set)}")
        print(f"Test set: {len(test_set)}")
        print("\nDataset files created in: data/telugu_reviews/")
        print("  - train.json")
        print("  - validation.json")
        print("  - test.json")
        print("  - dataset_stats.json")
        print("\n✅ All 4 subtasks completed:")
        print("  11.1: Review collection ✓")
        print("  11.2: Data cleaning ✓")
        print("  11.3: Tokenization ✓")
        print("  11.4: Dataset split ✓")
        
    finally:
        await preparer.close()


if __name__ == "__main__":
    asyncio.run(main())
