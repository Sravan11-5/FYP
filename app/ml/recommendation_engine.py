"""
Recommendation Engine for Telugu Movie Recommendations
Uses Siamese Network sentiment analysis and similarity for context-aware recommendations
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from app.ml.inference import get_model_inference
from app.config import settings
from app.collectors.tmdb_collector import TMDBDataCollector
from app.utils.translator import get_translator

logger = logging.getLogger(__name__)


class MovieRecommendationEngine:
    """
    Context-aware movie recommendation engine that uses:
    1. Siamese Network sentiment analysis
    2. Review similarity computation
    3. Genre and rating matching
    """
    
    def __init__(self):
        """Initialize recommendation engine with ML model and database"""
        self.model_inference = get_model_inference()
        self.tmdb_collector = TMDBDataCollector()
        self.translator = get_translator()
        logger.info("Recommendation engine initialized")
    
    async def get_recommendations(
        self,
        movie_name: str,
        min_sentiment_score: float = 0.6,
        max_results: int = 10,
        genre_weight: float = 0.3,
        sentiment_weight: float = 0.4,
        similarity_weight: float = 0.3
    ) -> List[Dict]:
        """
        Generate movie recommendations based on input movie.
        
        Args:
            movie_name: Name of the movie to get recommendations for
            min_sentiment_score: Minimum positive sentiment threshold (0-1)
            max_results: Maximum number of recommendations to return
            genre_weight: Weight for genre matching (0-1)
            sentiment_weight: Weight for sentiment score (0-1)
            similarity_weight: Weight for review similarity (0-1)
            
        Returns:
            List of recommended movies with scores and reasoning
        """
        logger.info(f"Generating recommendations for movie: {movie_name}")
        
        try:
            # Get database connection
            client = AsyncIOMotorClient(settings.MONGODB_URL)
            db = client[settings.MONGODB_DB_NAME]
            movies_collection = db.movies
            reviews_collection = db.reviews
            
            # Step 1: Find the input movie (or fetch from TMDB if not found)
            input_movie = await movies_collection.find_one(
                {"title": {"$regex": movie_name, "$options": "i"}}
            )
            
            if not input_movie:
                logger.info(f"Movie '{movie_name}' not found in database. Fetching from TMDB...")
                
                # Fetch movie from TMDB and add to database
                input_movie = await self._fetch_and_store_movie(movie_name, movies_collection)
                
                if not input_movie:
                    logger.warning(f"Movie not found on TMDB: {movie_name}")
                    return []
                
                logger.info(f"Successfully added '{input_movie['title']}' to database from TMDB")
            
            logger.info(f"Found input movie: {input_movie['title']}")
            
            # Step 2: Get reviews for input movie (from embedded reviews array)
            input_reviews = input_movie.get('reviews', [])
            
            if not input_reviews:
                logger.warning(f"No reviews found for movie: {movie_name}")
                return []
            
            logger.info(f"Found {len(input_reviews)} reviews for input movie")
            
            # Step 3: Analyze sentiment of input movie's reviews
            input_sentiments = await self._analyze_movie_sentiment(input_reviews)
            avg_input_sentiment = input_sentiments['avg_positive_score']
            
            logger.info(
                f"Input movie sentiment: {input_sentiments['sentiment_distribution']}, "
                f"avg positive: {avg_input_sentiment:.2f}"
            )
            
            # Step 4: Find candidate movies (same genres, similar ratings)
            candidate_movies = await self._find_candidate_movies(
                input_movie,
                movies_collection,
                exclude_movie_id=input_movie["_id"]
            )
            
            logger.info(f"Found {len(candidate_movies)} candidate movies")
            
            # Step 5: Score and rank candidates
            recommendations = []
            
            for candidate in candidate_movies:
                # Get candidate reviews (from embedded reviews array)
                candidate_reviews = candidate.get('reviews', [])
                
                if not candidate_reviews:
                    continue
                
                # Analyze candidate sentiment
                candidate_sentiments = await self._analyze_movie_sentiment(candidate_reviews)
                avg_candidate_sentiment = candidate_sentiments['avg_positive_score']
                
                # Filter by minimum sentiment threshold
                if avg_candidate_sentiment < min_sentiment_score:
                    continue
                
                # Compute similarity between input and candidate reviews
                similarity_score = await self._compute_review_similarity(
                    input_reviews,
                    candidate_reviews
                )
                
                # Calculate genre match score
                genre_score = self._calculate_genre_match(
                    input_movie.get('genres', []),
                    candidate.get('genres', [])
                )
                
                # Calculate rating similarity (inverse of difference)
                rating_score = self._calculate_rating_similarity(
                    input_movie.get('vote_average', 0),
                    candidate.get('vote_average', 0)
                )
                
                # Weighted final score
                final_score = (
                    genre_weight * genre_score +
                    sentiment_weight * avg_candidate_sentiment +
                    similarity_weight * similarity_score
                )
                
                # Build recommendation object
                recommendation = {
                    'movie_id': str(candidate['_id']),
                    'tmdb_id': candidate.get('tmdb_id'),
                    'title': candidate['title'],
                    'genres': candidate.get('genres', []),
                    'vote_average': candidate.get('vote_average', 0),
                    'release_date': candidate.get('release_date'),
                    'overview': candidate.get('overview', ''),
                    'poster_path': candidate.get('poster_path'),
                    'recommendation_score': round(final_score, 4),
                    'sentiment_analysis': {
                        'avg_positive_score': round(avg_candidate_sentiment, 4),
                        'distribution': candidate_sentiments['sentiment_distribution'],
                        'total_reviews': len(candidate_reviews)
                    },
                    'similarity_score': round(similarity_score, 4),
                    'genre_match_score': round(genre_score, 4),
                    'rating_similarity': round(rating_score, 4),
                    'reasoning': self._generate_reasoning(
                        genre_score,
                        avg_candidate_sentiment,
                        similarity_score
                    )
                }
                
                recommendations.append(recommendation)
            
            # Sort by recommendation score (descending)
            recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            # Return top N results
            top_recommendations = recommendations[:max_results]
            
            logger.info(f"Generated {len(top_recommendations)} recommendations")
            
            return top_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}", exc_info=True)
            raise
    
    async def _analyze_movie_sentiment(self, reviews: List[Dict]) -> Dict:
        """
        Analyze sentiment for all reviews of a movie.
        
        Args:
            reviews: List of review documents
            
        Returns:
            Dictionary with sentiment statistics
        """
        if not reviews:
            return {
                'avg_positive_score': 0.0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
            }
        
        # Extract review texts
        review_texts = [review.get('text', '') for review in reviews if review.get('text')]
        
        if not review_texts:
            return {
                'avg_positive_score': 0.0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
            }
        
        # Batch predict sentiments
        predictions = self.model_inference.predict_sentiment_batch(
            review_texts,
            batch_size=32,
            return_confidence=True
        )
        
        # Calculate statistics
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        positive_scores = []
        
        for pred in predictions:
            sentiment_counts[pred['sentiment']] += 1
            
            # Get positive probability as score
            if 'probabilities' in pred:
                positive_scores.append(pred['probabilities'].get('positive', 0))
        
        avg_positive_score = sum(positive_scores) / len(positive_scores) if positive_scores else 0
        
        return {
            'avg_positive_score': avg_positive_score,
            'sentiment_distribution': sentiment_counts
        }
    
    async def _find_candidate_movies(
        self,
        input_movie: Dict,
        movies_collection,
        exclude_movie_id,
        max_candidates: int = 50
    ) -> List[Dict]:
        """
        Find candidate movies based on genre and rating similarity.
        
        Args:
            input_movie: Input movie document
            movies_collection: MongoDB movies collection
            exclude_movie_id: Movie ID to exclude (input movie)
            max_candidates: Maximum candidates to return
            
        Returns:
            List of candidate movie documents
        """
        input_genres = input_movie.get('genres', [])
        input_rating = input_movie.get('vote_average', 0)
        
        # Query for movies with matching genres
        query = {
            '_id': {'$ne': exclude_movie_id}
        }
        
        if input_genres:
            query['genres'] = {'$in': input_genres}
        
        # Find movies with similar ratings (±2.0 range)
        if input_rating > 0:
            query['vote_average'] = {
                '$gte': input_rating - 2.0,
                '$lte': input_rating + 2.0
            }
        
        # Fetch candidates
        candidates = await movies_collection.find(query).to_list(length=max_candidates)
        
        return candidates
    
    async def _compute_review_similarity(
        self,
        input_reviews: List[Dict],
        candidate_reviews: List[Dict],
        sample_size: int = 10
    ) -> float:
        """
        Compute average similarity between input and candidate movie reviews.
        
        Args:
            input_reviews: Reviews of input movie
            candidate_reviews: Reviews of candidate movie
            sample_size: Number of reviews to sample for comparison
            
        Returns:
            Average similarity score (0-1)
        """
        # Sample reviews for efficiency
        input_texts = [
            r.get('text', '') 
            for r in input_reviews[:sample_size] 
            if r.get('text')
        ]
        candidate_texts = [
            r.get('text', '') 
            for r in candidate_reviews[:sample_size] 
            if r.get('text')
        ]
        
        if not input_texts or not candidate_texts:
            return 0.0
        
        # Compute pairwise similarities
        similarities = []
        
        for input_text in input_texts:
            for candidate_text in candidate_texts:
                try:
                    similarity = self.model_inference.compute_similarity(
                        input_text,
                        candidate_text
                    )
                    similarities.append(similarity)
                except Exception as e:
                    logger.warning(f"Similarity computation failed: {e}")
                    continue
        
        # Return average similarity
        if similarities:
            return sum(similarities) / len(similarities)
        return 0.0
    
    def _calculate_genre_match(
        self,
        input_genres: List[str],
        candidate_genres: List[str]
    ) -> float:
        """
        Calculate genre match score using Jaccard similarity.
        
        Args:
            input_genres: Genres of input movie
            candidate_genres: Genres of candidate movie
            
        Returns:
            Genre match score (0-1)
        """
        if not input_genres or not candidate_genres:
            return 0.0
        
        input_set = set(input_genres)
        candidate_set = set(candidate_genres)
        
        intersection = len(input_set & candidate_set)
        union = len(input_set | candidate_set)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_rating_similarity(
        self,
        input_rating: float,
        candidate_rating: float
    ) -> float:
        """
        Calculate rating similarity (inverse of normalized difference).
        
        Args:
            input_rating: Rating of input movie (0-10)
            candidate_rating: Rating of candidate movie (0-10)
            
        Returns:
            Rating similarity score (0-1)
        """
        if input_rating == 0 or candidate_rating == 0:
            return 0.5  # Neutral score if no rating
        
        # Normalize difference to 0-1 scale (max difference is 10)
        difference = abs(input_rating - candidate_rating)
        similarity = 1.0 - (difference / 10.0)
        
        return max(0.0, similarity)
    
    def _generate_reasoning(
        self,
        genre_score: float,
        sentiment_score: float,
        similarity_score: float
    ) -> str:
        """
        Generate human-readable reasoning for recommendation.
        
        Args:
            genre_score: Genre match score
            sentiment_score: Sentiment score
            similarity_score: Review similarity score
            
        Returns:
            Reasoning text
        """
        reasons = []
        
        if genre_score > 0.7:
            reasons.append("strong genre match")
        elif genre_score > 0.4:
            reasons.append("moderate genre match")
        
        if sentiment_score > 0.75:
            reasons.append("highly positive reviews")
        elif sentiment_score > 0.6:
            reasons.append("mostly positive reviews")
        
        if similarity_score > 0.8:
            reasons.append("very similar audience reception")
        elif similarity_score > 0.6:
            reasons.append("similar audience reception")
        
        if not reasons:
            return "Based on overall compatibility"
        
        return f"Recommended due to: {', '.join(reasons)}"
    
    async def _fetch_and_store_movie(self, movie_name: str, movies_collection) -> Optional[Dict]:
        """
        Fetch movie from TMDB API and store in database with Telugu-translated reviews.
        
        Args:
            movie_name: Name of the movie to search for
            movies_collection: MongoDB movies collection
            
        Returns:
            Stored movie document or None if not found
        """
        try:
            logger.info(f"Searching TMDB for: {movie_name}")
            
            # Search for movie on TMDB using the collector's search method
            search_results = await self.tmdb_collector.search_movie(movie_name, language="en")
            
            if not search_results:
                logger.warning(f"No results found on TMDB for: {movie_name}")
                return None
            
            # Get the first (most relevant) result
            tmdb_movie = search_results[0]
            tmdb_id = tmdb_movie['id']
            
            logger.info(f"Found movie on TMDB: {tmdb_movie['title']} (ID: {tmdb_id})")
            
            # Fetch full movie details
            movie_details = await self.tmdb_collector.get_movie_details(tmdb_id)
            
            if not movie_details:
                logger.error(f"Failed to fetch movie details for TMDB ID: {tmdb_id}")
                return None
            
            # Fetch and translate reviews
            logger.info(f"Fetching reviews for: {movie_details['title']}")
            translated_reviews = await self.tmdb_collector.get_reviews_and_translate(
                tmdb_id, 
                max_reviews=20
            )
            
            if not translated_reviews:
                logger.warning(f"No reviews found for movie: {movie_details['title']}")
                return None
            
            logger.info(f"Successfully translated {len(translated_reviews)} reviews to Telugu")
            
            # Prepare movie document for storage
            primary_genre = movie_details['genres'][0]['name'] if movie_details.get('genres') else 'Unknown'
            
            movie_doc = {
                'tmdb_id': tmdb_id,
                'title': movie_details['title'],
                'original_title': movie_details.get('original_title', movie_details['title']),
                'genre': primary_genre,
                'genres': [g['name'] for g in movie_details.get('genres', [])],
                'rating': movie_details.get('vote_average', 0),
                'overview': movie_details.get('overview', ''),
                'release_date': movie_details.get('release_date', ''),
                'poster_path': movie_details.get('poster_path'),
                'backdrop_path': movie_details.get('backdrop_path'),
                'reviews': translated_reviews,
                'review_count': len(translated_reviews),
                'has_reviews': True,
                'popularity': movie_details.get('popularity', 0),
                'vote_count': movie_details.get('vote_count', 0),
                'original_language': movie_details.get('original_language', 'en'),
                'added_on_demand': True,  # Flag to track auto-fetched movies
                'added_at': datetime.utcnow()
            }
            
            # Insert into database
            result = await movies_collection.insert_one(movie_doc)
            movie_doc['_id'] = result.inserted_id
            
            logger.info(
                f"Successfully stored movie '{movie_details['title']}' with "
                f"{len(translated_reviews)} Telugu reviews"
            )
            
            return movie_doc
            
        except Exception as e:
            logger.error(f"Error fetching and storing movie '{movie_name}': {e}", exc_info=True)
            return None


# Singleton instance getter
_recommendation_engine = None

def get_recommendation_engine() -> MovieRecommendationEngine:
    """
    Get singleton instance of recommendation engine.
    
    Returns:
        MovieRecommendationEngine instance
    """
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = MovieRecommendationEngine()
    return _recommendation_engine
