"""
Telugu Movie Recommendation System - Main Application
FastAPI backend for intelligent movie recommendations using Agentic AI and Deep Learning
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
from pathlib import Path
from app.config import settings
from app.database import connect_to_mongo, close_mongo_connection
from app.middleware.performance import PerformanceMonitoringMiddleware, set_performance_middleware
from app.routes.test_routes import router as test_router
from app.routes.api import router as api_router
from app.api.routes.sentiment import router as sentiment_router
from app.api.routes.recommendations import router as recommendations_router
from app.api.routes.search import router as search_router
from app.api.routes.orchestrator import router as orchestrator_router
from app.api.routes.system import router as system_router
from app.api.routes.feedback import router as feedback_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Intelligent movie recommendations using Agentic AI and Deep Learning for Telugu language reviews",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS (Subtask 1.3)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add performance monitoring middleware
performance_middleware = PerformanceMonitoringMiddleware(app)
app.add_middleware(PerformanceMonitoringMiddleware)
set_performance_middleware(performance_middleware)

# Global Exception Handlers (Subtask 1.4)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.error(f"Validation error: {exc.errors()}")
    # Convert errors to JSON-serializable format
    errors = []
    for error in exc.errors():
        error_dict = dict(error)
        # Convert ValueError objects to strings
        if 'ctx' in error_dict and 'error' in error_dict['ctx']:
            error_dict['ctx']['error'] = str(error_dict['ctx']['error'])
        errors.append(error_dict)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors, "body": exc.body},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred"},
    )

# Application startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Connect to MongoDB
    try:
        await connect_to_mongo()
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.warning("Application will continue without database connection")
    
    # Load ML model
    try:
        from app.ml.inference import get_model_inference
        model = get_model_inference()
        logger.info(f"ML model loaded successfully on device: {model.device}")
        logger.info(f"Model vocabulary size: {model.vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        logger.warning("Sentiment analysis endpoints may not work")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info(f"Shutting down {settings.APP_NAME}")
    
    # Close MongoDB connection
    await close_mongo_connection()

# Include routers
app.include_router(test_router)
app.include_router(api_router)
app.include_router(sentiment_router)
app.include_router(recommendations_router)
app.include_router(search_router)
app.include_router(orchestrator_router)
app.include_router(system_router)
app.include_router(feedback_router)

# Mount static files (CSS, JS, Images)
app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")
app.mount("/images", StaticFiles(directory="frontend/images"), name="images")

# Root endpoints
@app.get("/")
async def root():
    """Serve the frontend HTML"""
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        return FileResponse(frontend_path)
    # Fallback to API response if frontend not found
    logger.info("Root endpoint accessed")
    return {
        "message": f"{settings.APP_NAME} API",
        "status": "online",
        "version": settings.APP_VERSION
    }

@app.get("/api")
async def api_root():
    """API health check"""
    logger.info("API endpoint accessed")
    return {
        "message": f"{settings.APP_NAME} API",
        "status": "online",
        "version": settings.APP_VERSION
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
