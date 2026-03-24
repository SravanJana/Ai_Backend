"""
AI Trading Copilot - Chat API Endpoints
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Dict, Any

from models import ChatMessage, ChatResponse
from services.chatbot import trading_copilot
from services.pdf_service import pdf_service

router = APIRouter(prefix="/ai", tags=["AI Chat"])


class ChatRequest(BaseModel):
    """Chat request model."""
    user_id: int
    message: str
    context: Optional[Dict[str, Any]] = None


class PDFQuestionRequest(BaseModel):
    """PDF Q&A request model."""
    document_id: str
    question: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    AI Chatbot endpoint for trading insights.
    
    Users can ask questions like:
    - "Should I buy HDFC Bank?"
    - "Why is my portfolio risky?"
    - "Which stock should I sell?"
    - "How is the market today?"
    
    The AI analyzes user holdings, market data, and sentiment to provide insights.
    """
    print(f"[API] Chat endpoint called with message: {request.message}", flush=True)
    try:
        message = ChatMessage(
            user_id=request.user_id,
            message=request.message,
            context=request.context
        )
        
        print(f"[API] Calling trading_copilot.chat()", flush=True)
        response = await trading_copilot.chat(message)
        print(f"[API] Got response: {response.response[:50]}...", flush=True)
        return response
        
    except Exception as e:
        print(f"[API] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.delete("/chat/{user_id}/history")
async def clear_chat_history(user_id: int):
    """Clear conversation history for a user."""
    trading_copilot.clear_conversation(user_id)
    return {"message": "Conversation history cleared"}


@router.get("/suggested-prompts")
async def get_suggested_prompts():
    """Get suggested conversation starters."""
    return {
        "prompts": [
            "Analyze my portfolio",
            "What's my risk level?",
            "Should I buy Infosys?",
            "How is the market today?",
            "Best stocks to buy today",
            "Why is my portfolio risky?",
            "Compare TCS vs Infosys",
            "What should I sell?",
            "Portfolio rebalancing suggestions",
            "Market sentiment analysis"
        ]
    }


@router.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Analyze a PDF document and extract financial insights.
    
    Supports:
    - Earnings reports
    - Annual reports
    - Balance sheets
    - Income statements
    - Portfolio statements
    - Research reports
    - Mutual fund statements
    - Demat statements
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )
    
    # Check file size (max 10MB)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )
    
    try:
        result = await pdf_service.analyze_pdf(content, file.filename)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing PDF: {str(e)}"
        )


@router.post("/pdf-question")
async def ask_pdf_question(request: PDFQuestionRequest):
    """
    Ask a question about a previously analyzed PDF document.
    
    Requires:
    - document_id: The ID returned from /analyze-pdf
    - question: The question to ask about the document
    """
    try:
        result = await pdf_service.ask_question(request.document_id, request.question)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error answering question: {str(e)}"
        )
