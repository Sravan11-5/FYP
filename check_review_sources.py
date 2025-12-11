"""
Enhanced Dataset Preparer - Prioritizes Real Data Over Synthetic
=================================================================
This version checks for real Twitter reviews first, and only uses
synthetic data if absolutely no real data is available.
"""

import asyncio
import os
import json
from typing import List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()


async def check_data_sources():
    """Check what data sources are available in the database."""
    
    mongo_uri = os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
    if not mongo_uri:
        print("❌ MongoDB URI not found")
        return
    
    client = AsyncIOMotorClient(mongo_uri)
    db = client["telugu_movie_db"]
    
    print("=" * 70)
    print("DATABASE REVIEW SOURCES CHECK")
    print("=" * 70)
    
    # Count all reviews
    total_reviews = await db.reviews.count_documents({})
    print(f"\nTotal reviews in database: {total_reviews}")
    
    if total_reviews == 0:
        print("\n⚠️  No reviews found!")
        print("   Options:")
        print("   1. Wait for Twitter rate limits to reset (~15 minutes)")
        print("   2. Run Task 9 again without SKIP_TWITTER_COLLECTION")
        print("   3. Use synthetic data for development (already done)")
        client.close()
        return
    
    # Break down by source
    print("\nBreakdown by source:")
    
    # Count synthetic reviews
    synthetic_count = await db.reviews.count_documents({"source": "synthetic"})
    print(f"  📊 Synthetic: {synthetic_count} reviews")
    
    # Count Twitter reviews
    twitter_count = await db.reviews.count_documents({"source": "twitter"})
    print(f"  🐦 Twitter: {twitter_count} reviews")
    
    # Count TMDB reviews (if any)
    tmdb_count = await db.reviews.count_documents({"source": "tmdb"})
    if tmdb_count > 0:
        print(f"  🎬 TMDB: {tmdb_count} reviews")
    
    # Count other sources
    other_count = total_reviews - synthetic_count - twitter_count - tmdb_count
    if other_count > 0:
        print(f"  ❓ Other: {other_count} reviews")
    
    # Recommendation
    print("\n📋 Recommendation:")
    if twitter_count > 0:
        print(f"   ✅ You have {twitter_count} REAL Twitter reviews!")
        print(f"   → Dataset will use real data automatically")
        if synthetic_count > 0:
            print(f"   → {synthetic_count} synthetic reviews also available")
            print(f"   → You can filter by 'source' field if needed")
    elif synthetic_count > 0:
        print(f"   ⚠️  Only synthetic data available ({synthetic_count} reviews)")
        print(f"   → Good for development and testing")
        print(f"   → Collect real reviews for production use")
    
    # Sample review
    if total_reviews > 0:
        print("\n📝 Sample review:")
        sample = await db.reviews.find_one()
        print(f"   Source: {sample.get('source', 'unknown')}")
        print(f"   Movie: {sample.get('movie_title', 'unknown')}")
        print(f"   Text: {sample.get('text', '')[:80]}...")
        print(f"   Sentiment: {sample.get('sentiment', 'unknown')}")
        print(f"   Rating: {sample.get('rating', 0)}/10")
    
    client.close()


async def prepare_dataset_with_source_filter(
    use_real_only: bool = False,
    use_synthetic_only: bool = False,
    min_real_reviews: int = 100
):
    """
    Prepare dataset with source filtering options.
    
    Args:
        use_real_only: Only use real Twitter reviews
        use_synthetic_only: Only use synthetic reviews
        min_real_reviews: Minimum real reviews needed to skip synthetic
    """
    
    mongo_uri = os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
    if not mongo_uri:
        print("❌ MongoDB URI not found")
        return
    
    client = AsyncIOMotorClient(mongo_uri)
    db = client["telugu_movie_db"]
    
    print("=" * 70)
    print("DATASET PREPARATION WITH SOURCE FILTERING")
    print("=" * 70)
    
    # Count reviews by source
    total_reviews = await db.reviews.count_documents({})
    synthetic_count = await db.reviews.count_documents({"source": "synthetic"})
    real_count = total_reviews - synthetic_count
    
    print(f"\nAvailable reviews:")
    print(f"  Real (Twitter/TMDB): {real_count}")
    print(f"  Synthetic: {synthetic_count}")
    print(f"  Total: {total_reviews}")
    
    # Determine which data to use
    if use_real_only:
        print("\n🎯 Mode: REAL DATA ONLY")
        if real_count == 0:
            print("❌ No real reviews available!")
            print("   Cannot proceed without real data in this mode.")
            client.close()
            return
        query = {"source": {"$ne": "synthetic"}}
        print(f"✓ Will use {real_count} real reviews")
    
    elif use_synthetic_only:
        print("\n🎯 Mode: SYNTHETIC DATA ONLY")
        query = {"source": "synthetic"}
        print(f"✓ Will use {synthetic_count} synthetic reviews")
    
    else:
        # Smart mode: Prefer real data if we have enough
        print("\n🎯 Mode: SMART (Auto-select)")
        if real_count >= min_real_reviews:
            print(f"✓ Found {real_count} real reviews (≥ {min_real_reviews} minimum)")
            print("  → Using REAL DATA ONLY")
            query = {"source": {"$ne": "synthetic"}}
        elif real_count > 0:
            print(f"⚠️  Only {real_count} real reviews (< {min_real_reviews} minimum)")
            print(f"  → Using MIXED DATA (real + synthetic)")
            query = {}  # Use all reviews
        else:
            print(f"⚠️  No real reviews available")
            print(f"  → Using SYNTHETIC DATA for development")
            query = {"source": "synthetic"}
    
    # Collect selected reviews
    reviews = []
    cursor = db.reviews.find(query)
    async for review in cursor:
        reviews.append(review)
    
    print(f"\n✓ Collected {len(reviews)} reviews for dataset")
    
    # Show breakdown
    if len(reviews) > 0:
        sources = {}
        for review in reviews:
            source = review.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        print("\nDataset composition:")
        for source, count in sources.items():
            emoji = "🐦" if source == "twitter" else "📊" if source == "synthetic" else "🎬"
            print(f"  {emoji} {source}: {count} ({count/len(reviews)*100:.1f}%)")
    
    client.close()
    print("\n✅ Ready to proceed with dataset preparation!")
    print("   Run: python test_task11_with_synthetic_data.py")


async def main():
    """Main menu for data source checking and management."""
    
    print("\n" + "=" * 70)
    print("TASK 11: DATA SOURCE MANAGEMENT")
    print("=" * 70)
    print("\nOptions:")
    print("  1. Check current data sources in database")
    print("  2. Prepare dataset (smart mode - prefers real data)")
    print("  3. Prepare dataset (real data only)")
    print("  4. Prepare dataset (synthetic data only)")
    print("\n")
    
    # For demo, run option 1
    await check_data_sources()
    
    print("\n" + "=" * 70)
    print("\n💡 TIP: When Twitter rate limits reset:")
    print("   1. Set SKIP_TWITTER_COLLECTION=False in test_task9")
    print("   2. Run: python test_task9_data_collection.py")
    print("   3. Real reviews will be collected automatically")
    print("   4. Task 11 will detect and use them automatically!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
