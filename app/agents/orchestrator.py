"""
Agentic AI Orchestrator
Coordinates the end-to-end movie recommendation workflow
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.collectors import get_tmdb_collector, get_twitter_collector
from app.services import get_storage_service
from app.ml.inference import get_model_inference
from app.ml.recommendation_engine import get_recommendation_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Forward declaration for automation manager and decision maker
_automation_manager = None
_decision_maker = None


class AgentRole:
    """Define agent roles in the system"""
    DATA_COLLECTOR = "data_collector"
    ANALYZER = "analyzer"
    RECOMMENDER = "recommender"


class AgenticOrchestrator:
    """
    Agentic AI Orchestrator for movie recommendation system.
    
    Coordinates between three main agents:
    1. Data Collector Agent - Fetches movie data from TMDB and Twitter
    2. Analyzer Agent - Processes reviews with ML model for sentiment analysis
    3. Recommender Agent - Generates personalized recommendations
    """
    
    def __init__(self):
        """Initialize the orchestrator with all required tools"""
        self.tmdb_collector = get_tmdb_collector()
        self.twitter_collector = get_twitter_collector()
        self.storage_service = get_storage_service()
        self.ml_inference = get_model_inference()
        self.recommendation_engine = get_recommendation_engine()
        
        # Initialize decision maker
        from app.agents.decision_maker import get_decision_maker
        self.decision_maker = get_decision_maker(self.storage_service)
        
        logger.info("Agentic AI Orchestrator initialized with all tools")
        
        # Agent state tracking
        self.current_agent = None
        self.workflow_state = {}
    
    async def execute_workflow(
        self,
        movie_name: str,
        collect_new_data: bool = True,
        max_reviews: int = 10
    ) -> Dict[str, Any]:
        """
        Execute the complete end-to-end workflow.
        
        Args:
            movie_name: Name of the movie to process
            collect_new_data: Whether to collect fresh data or use cached
            max_reviews: Maximum number of reviews to collect
            
        Returns:
            Dict containing complete workflow results
        """
        workflow_id = f"workflow_{datetime.now().timestamp()}"
        logger.info(f"Starting workflow {workflow_id} for movie: {movie_name}")
        
        self.workflow_state = {
            "workflow_id": workflow_id,
            "movie_name": movie_name,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "agents_executed": []
        }
        
        try:
            # Phase 1: Data Collection Agent
            logger.info(f"[AGENT: {AgentRole.DATA_COLLECTOR}] Starting data collection")
            self.current_agent = AgentRole.DATA_COLLECTOR
            
            collection_result = await self._data_collector_agent(
                movie_name=movie_name,
                collect_new_data=collect_new_data,
                max_reviews=max_reviews
            )
            
            self.workflow_state["agents_executed"].append({
                "agent": AgentRole.DATA_COLLECTOR,
                "status": "completed",
                "result": collection_result
            })
            
            if not collection_result.get("success"):
                raise Exception(f"Data collection failed: {collection_result.get('error')}")
            
            movie_id = collection_result.get("movie_id")
            reviews_count = collection_result.get("reviews_count", 0)
            
            # Phase 2: Analyzer Agent
            logger.info(f"[AGENT: {AgentRole.ANALYZER}] Starting review analysis")
            self.current_agent = AgentRole.ANALYZER
            
            analysis_result = await self._analyzer_agent(
                movie_id=movie_id,
                movie_name=movie_name
            )
            
            self.workflow_state["agents_executed"].append({
                "agent": AgentRole.ANALYZER,
                "status": "completed",
                "result": analysis_result
            })
            
            # Phase 3: Recommender Agent
            logger.info(f"[AGENT: {AgentRole.RECOMMENDER}] Generating recommendations")
            self.current_agent = AgentRole.RECOMMENDER
            
            recommendation_result = await self._recommender_agent(
                movie_id=movie_id,
                movie_name=movie_name,
                sentiment_analysis=analysis_result
            )
            
            self.workflow_state["agents_executed"].append({
                "agent": AgentRole.RECOMMENDER,
                "status": "completed",
                "result": recommendation_result
            })
            
            # Complete workflow
            self.workflow_state["status"] = "completed"
            self.workflow_state["completed_at"] = datetime.now().isoformat()
            
            logger.info(f"Workflow {workflow_id} completed successfully")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "movie": {
                    "id": movie_id,
                    "name": movie_name,
                    "reviews_analyzed": reviews_count
                },
                "analysis": analysis_result,
                "recommendations": recommendation_result.get("recommendations", []),
                "workflow_state": self.workflow_state
            }
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {str(e)}", exc_info=True)
            self.workflow_state["status"] = "failed"
            self.workflow_state["error"] = str(e)
            self.workflow_state["failed_at"] = datetime.now().isoformat()
            
            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e),
                "workflow_state": self.workflow_state
            }
    
    async def _data_collector_agent(
        self,
        movie_name: str,
        collect_new_data: bool,
        max_reviews: int
    ) -> Dict[str, Any]:
        """
        Data Collector Agent - Autonomous decision making for data collection.
        
        Decides whether to:
        - Fetch new data from APIs
        - Use cached data from database
        - Handle rate limits and errors
        """
        try:
            # Autonomous Decision 1: Determine data fetching strategy
            logger.info(f"[DATA_COLLECTOR] Making autonomous decision for: {movie_name}")
            
            user_pref = "fresh" if collect_new_data else "cached"
            movie_decision = await self.decision_maker.decide_data_strategy(
                data_type="movie",
                identifier=movie_name,
                user_preference=user_pref
            )
            
            logger.info(f"[DATA_COLLECTOR] Decision: {movie_decision['action']}")
            logger.info(f"[DATA_COLLECTOR] Reasoning: {'; '.join(movie_decision['reasoning'])}")
            
            # Search TMDB for movie
            movie_results = await self.tmdb_collector.search_movie(movie_name)
            
            if not movie_results:
                logger.warning(f"[DATA_COLLECTOR] Movie not found on TMDB: {movie_name}")
                return {
                    "success": False,
                    "error": "Movie not found on TMDB"
                }
            
            # Get movie details
            movie_data = movie_results[0]
            tmdb_id = movie_data.get('id')
            
            detailed_movie = await self.tmdb_collector.get_movie_details(tmdb_id)
            
            if not detailed_movie:
                return {
                    "success": False,
                    "error": "Failed to get movie details"
                }
            
            # Store movie
            parsed_movie = self.tmdb_collector.parse_movie_data(detailed_movie)
            movie_id = await self.storage_service.store_movie(parsed_movie)
            
            logger.info(f"[DATA_COLLECTOR] Movie stored with ID: {movie_id}")
            
            # Autonomous Decision 2: Reviews collection strategy
            reviews_decision = await self.decision_maker.decide_data_strategy(
                data_type="reviews",
                identifier=str(movie_id),
                user_preference=user_pref
            )
            
            logger.info(f"[DATA_COLLECTOR] Reviews decision: {reviews_decision['action']}")
            logger.info(f"[DATA_COLLECTOR] Reasoning: {'; '.join(reviews_decision['reasoning'])}")
            
            reviews_count = 0
            
            if reviews_decision["action"] == "fetch_new":
                logger.info(f"[DATA_COLLECTOR] Collecting fresh reviews (max: {max_reviews})")
                
                # Collect from Twitter
                reviews = await self.twitter_collector.search_movie_reviews(
                    movie_name=movie_name,
                    max_results=max_reviews,
                    language="te"
                )
                
                if reviews:
                    parsed_reviews = [
                        self.twitter_collector.parse_tweet_data(review, str(movie_id))
                        for review in reviews
                    ]
                    
                    await self.storage_service.store_reviews_batch(parsed_reviews)
                    reviews_count = len(parsed_reviews)
                    
                    logger.info(f"[DATA_COLLECTOR] Collected {reviews_count} new reviews")
                else:
                    logger.warning(f"[DATA_COLLECTOR] No new reviews found")
            else:
                # Use cached data
                logger.info(f"[DATA_COLLECTOR] Using cached data from database")
                cached_reviews = await self.storage_service.get_reviews_for_movie(str(movie_id))
                reviews_count = len(cached_reviews) if cached_reviews else 0
            
            return {
                "success": True,
                "movie_id": str(movie_id),
                "tmdb_id": parsed_movie.get('tmdb_id'),
                "movie_title": parsed_movie.get('title'),
                "reviews_count": reviews_count,
                "data_source": "fresh" if collect_new_data else "cached"
            }
            
        except Exception as e:
            logger.error(f"[DATA_COLLECTOR] Error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _analyzer_agent(
        self,
        movie_id: str,
        movie_name: str
    ) -> Dict[str, Any]:
        """
        Analyzer Agent - Processes reviews with ML model.
        
        Autonomous decisions:
        - Batch processing strategy
        - Handle missing/corrupted data
        - Aggregate sentiment scores
        """
        try:
            logger.info(f"[ANALYZER] Analyzing reviews for movie ID: {movie_id}")
            
            # Get reviews from database
            reviews = await self.storage_service.get_reviews_for_movie(movie_id)
            
            if not reviews:
                logger.warning(f"[ANALYZER] No reviews found for analysis")
                return {
                    "success": True,
                    "reviews_analyzed": 0,
                    "sentiment_distribution": {
                        "positive": 0,
                        "negative": 0,
                        "neutral": 0
                    },
                    "average_sentiment": 0.0
                }
            
            # Decision: Use batch processing for efficiency
            review_texts = [review.get('text', '') for review in reviews]
            
            logger.info(f"[ANALYZER] Processing {len(review_texts)} reviews in batch")
            
            # Analyze with ML model
            sentiments = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for text in review_texts:
                if not text:
                    continue
                
                try:
                    result = self.ml_inference.predict_sentiment(text)
                    sentiment_label = result.get('sentiment', 'neutral')
                    sentiment_score = result.get('confidence', 0.0)
                    
                    sentiments.append(sentiment_score)
                    
                    if sentiment_label == 'positive':
                        positive_count += 1
                    elif sentiment_label == 'negative':
                        negative_count += 1
                    else:
                        neutral_count += 1
                        
                except Exception as e:
                    logger.error(f"[ANALYZER] Error analyzing review: {str(e)}")
                    continue
            
            # Calculate aggregate metrics
            total_reviews = len(sentiments)
            average_sentiment = sum(sentiments) / total_reviews if total_reviews > 0 else 0.0
            
            sentiment_distribution = {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "positive_percentage": (positive_count / total_reviews * 100) if total_reviews > 0 else 0,
                "negative_percentage": (negative_count / total_reviews * 100) if total_reviews > 0 else 0,
                "neutral_percentage": (neutral_count / total_reviews * 100) if total_reviews > 0 else 0
            }
            
            logger.info(f"[ANALYZER] Analysis complete: {total_reviews} reviews processed")
            logger.info(f"[ANALYZER] Average sentiment: {average_sentiment:.2f}")
            logger.info(f"[ANALYZER] Distribution: {sentiment_distribution}")
            
            return {
                "success": True,
                "reviews_analyzed": total_reviews,
                "sentiment_distribution": sentiment_distribution,
                "average_sentiment": average_sentiment
            }
            
        except Exception as e:
            logger.error(f"[ANALYZER] Error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _recommender_agent(
        self,
        movie_id: str,
        movie_name: str,
        sentiment_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recommender Agent - Generates personalized recommendations.
        
        Autonomous decisions:
        - Recommendation strategy based on sentiment
        - Number of recommendations to return
        - Explanation generation
        """
        try:
            logger.info(f"[RECOMMENDER] Generating recommendations for: {movie_name}")
            
            # Decision 1: Determine recommendation strategy
            avg_sentiment = sentiment_analysis.get('average_sentiment', 0.0)
            reviews_count = sentiment_analysis.get('reviews_analyzed', 0)
            
            # Get movie details for context
            movie = await self.storage_service.get_movie_by_tmdb_id(movie_id)
            
            if not movie:
                logger.warning(f"[RECOMMENDER] Movie not found in database")
                return {
                    "success": False,
                    "error": "Movie not found"
                }
            
            # Decision 2: Get recommendations using the engine
            logger.info(f"[RECOMMENDER] Using recommendation engine")
            
            recommendations = await self.recommendation_engine.get_recommendations(
                movie_id=movie_id,
                max_results=10
            )
            
            # Decision 3: Generate explanations
            explanations = []
            for rec in recommendations:
                explanation = self._generate_explanation(
                    rec,
                    sentiment_analysis,
                    movie
                )
                rec['explanation'] = explanation
                explanations.append(explanation)
            
            logger.info(f"[RECOMMENDER] Generated {len(recommendations)} recommendations")
            
            return {
                "success": True,
                "recommendations": recommendations,
                "recommendation_count": len(recommendations),
                "strategy": "sentiment_based" if reviews_count > 0 else "genre_based"
            }
            
        except Exception as e:
            logger.error(f"[RECOMMENDER] Error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_explanation(
        self,
        recommendation: Dict[str, Any],
        sentiment_analysis: Dict[str, Any],
        source_movie: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation for recommendation"""
        reasons = []
        
        # Genre match
        source_genres = set(source_movie.get('genres', []))
        rec_genres = set(recommendation.get('genres', []))
        genre_match = source_genres & rec_genres
        
        if genre_match:
            reasons.append(f"Same genre: {', '.join(list(genre_match)[:2])}")
        
        # Rating similarity
        source_rating = source_movie.get('rating', 0)
        rec_rating = recommendation.get('rating', 0)
        
        if rec_rating >= source_rating:
            reasons.append(f"Higher rating: {rec_rating}/10")
        
        # Sentiment-based
        avg_sentiment = sentiment_analysis.get('average_sentiment', 0.0)
        if avg_sentiment > 0.6:
            reasons.append("Positive reviews in Telugu")
        
        # Recommendation score
        score = recommendation.get('recommendation_score', 0)
        if score > 80:
            reasons.append("Highly recommended")
        
        if not reasons:
            reasons.append("Similar movie based on genre and ratings")
        
        return " | ".join(reasons)
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of a specific workflow"""
        if self.workflow_state.get("workflow_id") == workflow_id:
            return self.workflow_state
        else:
            return {
                "error": "Workflow not found",
                "workflow_id": workflow_id
            }
    
    def get_automation_manager(self):
        """Get the automation manager for this orchestrator"""
        global _automation_manager
        if _automation_manager is None:
            from app.agents.automation import get_workflow_manager
            _automation_manager = get_workflow_manager(self)
        return _automation_manager
    
    def get_decision_maker(self):
        """Get the decision maker for this orchestrator"""
        return self.decision_maker


# Singleton instance
_orchestrator_instance = None


def get_orchestrator() -> AgenticOrchestrator:
    """Get the singleton orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AgenticOrchestrator()
    return _orchestrator_instance
