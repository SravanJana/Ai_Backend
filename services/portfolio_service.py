"""
AI Trading Copilot - Portfolio Service
Manages user portfolios and provides analytics.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime

from models import (
    Portfolio, Holding, PortfolioSummary, PortfolioHealth
)
from services.market_data import market_data_service
from services.risk_engine import risk_engine


# Mock database of user holdings (in production, this would come from actual trading platform DB)
MOCK_HOLDINGS_DB: Dict[int, List[Dict]] = {
    1: [
        {"symbol": "INFY", "qty": 50, "avg_price": 1450.00},
        {"symbol": "RELIANCE", "qty": 20, "avg_price": 2500.00},
        {"symbol": "HDFCBANK", "qty": 30, "avg_price": 1650.00},
        {"symbol": "TCS", "qty": 15, "avg_price": 3600.00},
        {"symbol": "ITC", "qty": 100, "avg_price": 420.00},
        {"symbol": "ICICIBANK", "qty": 40, "avg_price": 980.00},
        {"symbol": "BHARTIARTL", "qty": 25, "avg_price": 1100.00},
    ],
    2: [
        {"symbol": "WIPRO", "qty": 80, "avg_price": 450.00},
        {"symbol": "HCLTECH", "qty": 30, "avg_price": 1350.00},
        {"symbol": "SBIN", "qty": 100, "avg_price": 620.00},
        {"symbol": "MARUTI", "qty": 5, "avg_price": 10500.00},
    ],
    3: [
        {"symbol": "TCS", "qty": 100, "avg_price": 3500.00},
        {"symbol": "INFY", "qty": 200, "avg_price": 1400.00},
        {"symbol": "WIPRO", "qty": 150, "avg_price": 440.00},
    ]
}


class PortfolioService:
    """Service for managing and analyzing user portfolios."""
    
    def __init__(self):
        self.market_data = market_data_service
        self.risk_engine = risk_engine
    
    def get_holdings(self, user_id: int) -> List[Holding]:
        """Get user holdings from database."""
        holdings_data = MOCK_HOLDINGS_DB.get(user_id, [])
        
        holdings = []
        for h in holdings_data:
            holding = Holding(
                symbol=h["symbol"],
                qty=h["qty"],
                avg_price=h["avg_price"]
            )
            holdings.append(holding)
        
        return holdings
    
    def get_portfolio(self, user_id: int) -> Portfolio:
        """Get complete portfolio with current values."""
        holdings = self.get_holdings(user_id)
        
        # Enrich holdings with current prices
        portfolio_data = self.risk_engine.calculate_portfolio_value(holdings)
        
        return Portfolio(
            user_id=user_id,
            holdings=portfolio_data["holdings"],
            total_value=portfolio_data["total_value"],
            total_investment=portfolio_data["total_investment"],
            total_pnl=portfolio_data["total_pnl"],
            total_pnl_percentage=portfolio_data["total_pnl_percentage"]
        )
    
    def get_portfolio_summary(self, user_id: int) -> PortfolioSummary:
        """Get comprehensive portfolio analysis summary."""
        portfolio = self.get_portfolio(user_id)
        
        # Calculate risk metrics
        risk_metrics = self.risk_engine.analyze_portfolio(
            portfolio.holdings, 
            user_id
        )
        
        # Get portfolio health
        health = self.risk_engine.get_portfolio_health(
            portfolio.holdings,
            risk_metrics
        )
        
        # Find top and worst performers
        sorted_by_pnl = sorted(
            portfolio.holdings,
            key=lambda h: h.pnl_percentage or 0,
            reverse=True
        )
        
        top_performers = sorted_by_pnl[:3]
        worst_performers = sorted_by_pnl[-3:][::-1]
        
        # Generate AI insights
        ai_insights = self._generate_ai_insights(
            portfolio, risk_metrics, health
        )
        
        # Generate rebalance suggestions
        rebalance = self._generate_rebalance_suggestions(
            portfolio, risk_metrics
        )
        
        return PortfolioSummary(
            user_id=user_id,
            portfolio=portfolio,
            risk_metrics=risk_metrics,
            health=health,
            top_performers=top_performers,
            worst_performers=worst_performers,
            ai_insights=ai_insights,
            rebalance_suggestions=rebalance
        )
    
    def _generate_ai_insights(
        self,
        portfolio: Portfolio,
        risk_metrics,
        health: PortfolioHealth
    ) -> str:
        """Generate AI-powered insights summary."""
        insights = []
        
        # Portfolio value insight
        if portfolio.total_pnl_percentage > 10:
            insights.append(
                f"Your portfolio is performing well with {portfolio.total_pnl_percentage:.1f}% returns."
            )
        elif portfolio.total_pnl_percentage > 0:
            insights.append(
                f"Your portfolio is in profit with {portfolio.total_pnl_percentage:.1f}% returns."
            )
        else:
            insights.append(
                f"Your portfolio is currently at {portfolio.total_pnl_percentage:.1f}% loss. "
                "Consider reviewing underperforming positions."
            )
        
        # Risk insight
        if risk_metrics.risk_level.value in ["High", "Very High"]:
            insights.append(
                f"Risk level is {risk_metrics.risk_level.value}. "
                "Consider defensive measures or position sizing."
            )
        else:
            insights.append(
                f"Risk level is {risk_metrics.risk_level.value}, which is manageable."
            )
        
        # Sector insight
        if risk_metrics.sector_exposure:
            top_sector = risk_metrics.sector_exposure[0]
            insights.append(
                f"Highest allocation is in {top_sector.sector} ({top_sector.percentage:.1f}%)."
            )
        
        # Sharpe ratio insight
        if risk_metrics.sharpe_ratio > 1:
            insights.append("Risk-adjusted returns are excellent.")
        elif risk_metrics.sharpe_ratio > 0:
            insights.append("Risk-adjusted returns are positive but can be improved.")
        else:
            insights.append("Consider rebalancing to improve risk-adjusted returns.")
        
        return " ".join(insights)
    
    def _generate_rebalance_suggestions(
        self,
        portfolio: Portfolio,
        risk_metrics
    ) -> List[Dict[str, Any]]:
        """Generate specific rebalance recommendations."""
        suggestions = []
        
        # Check for overweight positions
        total_value = portfolio.total_value
        
        for holding in portfolio.holdings:
            if holding.current_value is None:
                continue
                
            weight = (holding.current_value / total_value * 100) if total_value > 0 else 0
            
            # If any position > 25%, suggest trimming
            if weight > 25:
                suggestions.append({
                    "action": "REDUCE",
                    "symbol": holding.symbol,
                    "current_weight": round(weight, 1),
                    "target_weight": 20,
                    "reason": f"Position too concentrated at {weight:.1f}%"
                })
        
        # Check sector concentration
        for exposure in risk_metrics.sector_exposure:
            if exposure.percentage > 40:
                # Find stocks in this sector
                sector_stocks = [
                    h.symbol for h in portfolio.holdings
                    if market_data_service.get_sector(h.symbol) == exposure.sector
                ]
                
                suggestions.append({
                    "action": "SECTOR_REDUCE",
                    "sector": exposure.sector,
                    "current_exposure": exposure.percentage,
                    "target_exposure": 30,
                    "affected_stocks": sector_stocks,
                    "reason": f"Sector overweight at {exposure.percentage:.1f}%"
                })
        
        # Suggest adding defensive stocks if volatility is high
        if risk_metrics.volatility > 0.30:
            suggestions.append({
                "action": "ADD",
                "sector": "FMCG",
                "reason": "Add defensive stocks to reduce volatility",
                "suggested_stocks": ["ITC", "HINDUNILVR"]
            })
        
        return suggestions[:5]
    
    def add_holding(
        self, 
        user_id: int, 
        symbol: str, 
        quantity: int, 
        avg_price: float
    ) -> bool:
        """Add new holding to user portfolio."""
        if user_id not in MOCK_HOLDINGS_DB:
            MOCK_HOLDINGS_DB[user_id] = []
        
        # Check if stock already exists
        for h in MOCK_HOLDINGS_DB[user_id]:
            if h["symbol"].upper() == symbol.upper():
                # Update existing holding
                total_qty = h["qty"] + quantity
                total_cost = (h["qty"] * h["avg_price"]) + (quantity * avg_price)
                h["qty"] = total_qty
                h["avg_price"] = total_cost / total_qty
                return True
        
        # Add new holding
        MOCK_HOLDINGS_DB[user_id].append({
            "symbol": symbol.upper(),
            "qty": quantity,
            "avg_price": avg_price
        })
        
        return True
    
    def remove_holding(self, user_id: int, symbol: str) -> bool:
        """Remove holding from user portfolio."""
        if user_id not in MOCK_HOLDINGS_DB:
            return False
        
        original_len = len(MOCK_HOLDINGS_DB[user_id])
        MOCK_HOLDINGS_DB[user_id] = [
            h for h in MOCK_HOLDINGS_DB[user_id]
            if h["symbol"].upper() != symbol.upper()
        ]
        
        return len(MOCK_HOLDINGS_DB[user_id]) < original_len


# Global instance
portfolio_service = PortfolioService()
