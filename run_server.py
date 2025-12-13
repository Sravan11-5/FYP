"""
Simple server runner
"""
import uvicorn
import asyncio

if __name__ == "__main__":
    # Ensure Windows event loop policy for Python 3.8+
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
