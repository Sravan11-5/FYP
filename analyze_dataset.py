"""Analyze Task 11 dataset statistics."""
import json
from collections import Counter

# Load training data
with open('data/telugu_reviews/train.json', encoding='utf-8') as f:
    train_data = json.load(f)

with open('data/telugu_reviews/validation.json', encoding='utf-8') as f:
    val_data = json.load(f)

with open('data/telugu_reviews/test.json', encoding='utf-8') as f:
    test_data = json.load(f)

print("=" * 70)
print("TASK 11 DATASET ANALYSIS")
print("=" * 70)

print("\n1. Dataset Size:")
print(f"   Training: {len(train_data)} reviews")
print(f"   Validation: {len(val_data)} reviews")
print(f"   Test: {len(test_data)} reviews")
print(f"   Total: {len(train_data) + len(val_data) + len(test_data)} reviews")

print("\n2. Sentiment Distribution (Training Set):")
train_sentiments = Counter(r['sentiment'] for r in train_data)
for sentiment, count in sorted(train_sentiments.items()):
    print(f"   {sentiment.capitalize()}: {count} ({count/len(train_data)*100:.1f}%)")

print("\n3. Rating Distribution (Training Set):")
train_ratings = Counter(r['rating'] for r in train_data)
for rating in sorted(train_ratings.keys()):
    count = train_ratings[rating]
    print(f"   {rating}/10: {count} ({count/len(train_data)*100:.1f}%)")

print("\n4. Average Token Count:")
avg_tokens = sum(r['token_count'] for r in train_data) / len(train_data)
print(f"   {avg_tokens:.1f} tokens per review")

print("\n5. Movies Covered:")
movies = Counter(r['movie_title'] for r in train_data)
print(f"   {len(movies)} unique movies")
for movie, count in movies.most_common(5):
    print(f"   - {movie}: {count} reviews")

print("\n6. Sample Reviews:")
for i, review in enumerate(train_data[:3], 1):
    print(f"\n   Sample {i} ({review['sentiment']}, rating: {review['rating']}/10):")
    print(f"   Movie: {review['movie_title']}")
    print(f"   Text: {review['text']}")
    print(f"   Tokens: {review['token_count']}")

print("\n" + "=" * 70)
print("✅ Dataset ready for Siamese Network training (Task 12)")
print("=" * 70)
