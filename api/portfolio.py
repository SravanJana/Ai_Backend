"""
AI Trading Copilot - Portfolio API Endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from models import Portfolio, Holding, PortfolioSummary, RiskMetrics
from services.portfolio_service import portfolio_service
from services.risk_engine import risk_engine

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


class AddHoldingRequest(BaseModel):
    """Request model for adding a holding."""
    symbol: str
    quantity: int
    avg_price: float


class HoldingResponse(BaseModel):
    """Response model for holdings."""
    symbol: str
    qty: int
    avg_price: float
    current_price: Optional[float] = None
    current_value: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    sector: Optional[str] = None


@router.get("/{user_id}", response_model=List[HoldingResponse])
async def get_holdings(user_id: int):
    """
    Get user holdings.
    
    Returns list of holdings with basic info:
    - symbol
    - quantity
    - average price
    """
    try:
        holdings = portfolio_service.get_holdings(user_id)
        
        return [
            HoldingResponse(
                symbol=h.symbol,
                qty=h.quantity,
                avg_price=h.average_price
            )
            for h in holdings
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching holdings: {str(e)}")


@router.get("/{user_id}/detailed")
async def get_portfolio_detailed(user_id: int):
    """
    Get detailed portfolio with current prices and P&L.
    
    Returns enriched portfolio data including:
    - Current prices
    - Current values
    - P&L for each holding
    - Total portfolio metrics
    """
    try:
        portfolio = portfolio_service.get_portfolio(user_id)
        
        return {
            "user_id": portfolio.user_id,
            "total_value": portfolio.total_value,
            "total_investment": portfolio.total_investment,
            "total_pnl": portfolio.total_pnl,
            "total_pnl_percentage": portfolio.total_pnl_percentage,
            "last_updated": portfolio.last_updated.isoformat(),
            "holdings": [
                {
                    "symbol": h.symbol,
                    "quantity": h.quantity,
                    "average_price": h.average_price,
                    "current_price": h.current_price,
                    "current_value": h.current_value,
                    "pnl": h.pnl,
                    "pnl_percentage": h.pnl_percentage,
                    "sector": h.sector
                }
                for h in portfolio.holdings
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio: {str(e)}")


@router.get("/{user_id}/risk", response_model=RiskMetrics)
async def get_portfolio_risk(user_id: int):
    """
    Get portfolio risk metrics.
    
    Returns:
    - Risk score (0-1)
    - Risk level (Low/Moderate/High/Very High)
    - Volatility
    - Sharpe ratio
    - Max drawdown
    - Beta
    - Sector exposure
    - Suggestions
    """
    try:
        holdings = portfolio_service.get_holdings(user_id)
        
        if not holdings:
            raise HTTPException(status_code=404, detail="No holdings found")
        
        risk_metrics = risk_engine.analyze_portfolio(holdings, user_id)
        return risk_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing risk: {str(e)}")


@router.get("/ai/portfolio-summary/{user_id}")
async def get_portfolio_summary(user_id: int):
    """
    Get comprehensive AI-powered portfolio summary.
    
    Returns:
    - Complete portfolio data
    - Risk metrics
    - Health assessment
    - Top/worst performers
    - AI insights
    - Rebalancing suggestions
    """
    try:
        summary = portfolio_service.get_portfolio_summary(user_id)
        
        return {
            "user_id": summary.user_id,
            "portfolio": {
                "total_value": summary.portfolio.total_value,
                "total_investment": summary.portfolio.total_investment,
                "total_pnl": summary.portfolio.total_pnl,
                "total_pnl_percentage": summary.portfolio.total_pnl_percentage,
                "holdings_count": len(summary.portfolio.holdings)
            },
            "risk": {
                "risk_score": summary.risk_metrics.risk_score,
                "risk_level": summary.risk_metrics.risk_level.value,
                "volatility": summary.risk_metrics.volatility,
                "sharpe_ratio": summary.risk_metrics.sharpe_ratio,
                "beta": summary.risk_metrics.beta,
                "sector_exposure": [
                    {"sector": s.sector, "percentage": s.percentage}
                    for s in summary.risk_metrics.sector_exposure
                ]
            },
            "health": {
                "score": summary.health.health_score,
                "status": summary.health.health_status,
                "strengths": summary.health.strengths,
                "weaknesses": summary.health.weaknesses,
                "recommendations": summary.health.recommendations
            },
            "top_performers": [
                {
                    "symbol": h.symbol,
                    "pnl_percentage": h.pnl_percentage
                }
                for h in summary.top_performers
            ],
            "worst_performers": [
                {
                    "symbol": h.symbol,
                    "pnl_percentage": h.pnl_percentage
                }
                for h in summary.worst_performers
            ],
            "ai_insights": summary.ai_insights,
            "rebalance_suggestions": summary.rebalance_suggestions,
            "timestamp": summary.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@router.post("/{user_id}/holdings")
async def add_holding(user_id: int, request: AddHoldingRequest):
    """Add a new holding to user's portfolio."""
    try:
        success = portfolio_service.add_holding(
            user_id,
            request.symbol,
            request.quantity,
            request.avg_price
        )
        
        if success:
            return {"message": f"Added {request.quantity} shares of {request.symbol}"}
        else:
            raise HTTPException(status_code=400, detail="Failed to add holding")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding holding: {str(e)}")


@router.delete("/{user_id}/holdings/{symbol}")
async def remove_holding(user_id: int, symbol: str):
    """Remove a holding from user's portfolio."""
    try:
        success = portfolio_service.remove_holding(user_id, symbol)
        
        if success:
            return {"message": f"Removed {symbol} from portfolio"}
        else:
            raise HTTPException(status_code=404, detail=f"Holding {symbol} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing holding: {str(e)}")
