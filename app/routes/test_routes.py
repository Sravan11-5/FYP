"""
Database Test Routes
Test CRUD operations for all models
"""
from fastapi import APIRouter, HTTPException
from app.models import MovieModel, ReviewModel, UserSearchModel, GenreModel
from app.database import get_database
from datetime import datetime

router = APIRouter(prefix="/api/test", tags=["Testing"])

@router.post("/movie", response_model=dict)
async def create_test_movie():
    """Create a test movie document"""
    try:
        db = get_database()
        
        test_movie = {
            "tmdb_id": 999999,
            "title": "Test Movie KGF",
            "original_title": "టెస్ట్ మూవీ కెజిఎఫ్",
            "genres": ["Action", "Drama"],
            "rating": 8.5,
            "vote_count": 1000,
            "release_date": "2024-01-01",
            "overview": "This is a test movie for database verification",
            "language": "te",
            "total_reviews": 0,
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow()
        }
        
        result = await db.movies.insert_one(test_movie)
        return {"success": True, "inserted_id": str(result.inserted_id), "message": "Test movie created successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating test movie: {str(e)}")

@router.get("/movie/{tmdb_id}", response_model=dict)
async def get_test_movie(tmdb_id: int):
    """Get a movie by TMDB ID"""
    try:
        db = get_database()
        movie = await db.movies.find_one({"tmdb_id": tmdb_id})
        
        if not movie:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        movie["_id"] = str(movie["_id"])
        return {"success": True, "movie": movie}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching movie: {str(e)}")

@router.put("/movie/{tmdb_id}", response_model=dict)
async def update_test_movie(tmdb_id: int):
    """Update a test movie"""
    try:
        db = get_database()
        
        result = await db.movies.update_one(
            {"tmdb_id": tmdb_id},
            {"$set": {"total_reviews": 5, "last_updated": datetime.utcnow()}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        return {"success": True, "modified_count": result.modified_count, "message": "Movie updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating movie: {str(e)}")

@router.delete("/movie/{tmdb_id}", response_model=dict)
async def delete_test_movie(tmdb_id: int):
    """Delete a test movie"""
    try:
        db = get_database()
        
        result = await db.movies.delete_one({"tmdb_id": tmdb_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        return {"success": True, "deleted_count": result.deleted_count, "message": "Movie deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting movie: {str(e)}")

@router.get("/collections", response_model=dict)
async def list_collections():
    """List all MongoDB collections"""
    try:
        db = get_database()
        collections = await db.list_collection_names()
        return {"success": True, "collections": collections}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

@router.get("/indexes/{collection}", response_model=dict)
async def list_indexes(collection: str):
    """List indexes for a collection"""
    try:
        db = get_database()
        indexes = await db[collection].index_information()
        return {"success": True, "collection": collection, "indexes": indexes}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing indexes: {str(e)}")
