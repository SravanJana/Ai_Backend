"""
AI Trading Copilot - Main Application
Production-ready AI-powered trading intelligence system.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime

from config import settings
from api.chat import router as chat_router
from api.portfolio import router as portfolio_router
from api.stocks import router as stocks_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("🚀 AI Trading Copilot starting up...")
    print(f"📊 Environment: {'Development' if settings.debug else 'Production'}")
    print(f"🔑 Groq configured: {bool(settings.groq_api_key)}")
    yield
    # Shutdown
    print("👋 AI Trading Copilot shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AI Trading Copilot",
    description="""
## AI-Powered Trading Intelligence System

The AI Trading Copilot provides intelligent insights for your investment portfolio.

### Features:
- 🤖 **AI Chatbot** - Ask questions about your portfolio and stocks
- 📊 **Portfolio Analysis** - Deep dive into your holdings
- 📈 **Stock Analysis** - Technical indicators and signals
- 📉 **Risk Assessment** - Comprehensive risk metrics
- 📰 **Sentiment Analysis** - News-based market sentiment
- 🏛️ **Market Overview** - Real-time market data

### Example Usage:
```python
# Chat with AI
POST /ai/chat
{
    "user_id": 1,
    "message": "Should I buy Infosys?"
}

# Get portfolio summary
GET /portfolio/ai/portfolio-summary/1

# Analyze stock
GET /ai/stock-analysis/INFY
```
    """,
    version="1.0.0",
    contact={
        "name": "AI Trading Copilot",
        "email": "support@tradingcopilot.ai"
    },
    license_info={
        "name": "MIT License"
    },
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Include routers
app.include_router(chat_router)
app.include_router(portfolio_router)
app.include_router(stocks_router)


# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - Health check."""
    return {
        "status": "healthy",
        "service": "AI Trading Copilot",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "openai": "configured" if settings.openai_api_key else "not configured",
            "market_data": "available"
        },
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
