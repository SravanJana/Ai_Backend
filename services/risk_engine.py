"""
AI Trading Copilot - Portfolio Risk Engine
Calculates risk metrics, sector exposure, and generates recommendations.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from models import (
    Portfolio, Holding, RiskMetrics, RiskLevel, SectorExposure,
    PortfolioHealth, PortfolioSummary
)
from services.market_data import market_data_service


class PortfolioRiskEngine:
    """Engine for calculating portfolio risk metrics and analytics."""
    
    # Risk-free rate (approximate Indian 10-year government bond yield)
    RISK_FREE_RATE = 0.07  # 7% annual
    
    # Risk thresholds
    VOLATILITY_THRESHOLDS = {
        "low": 0.15,
        "moderate": 0.25,
        "high": 0.35
    }
    
    # Sector concentration thresholds
    MAX_SECTOR_CONCENTRATION = 0.40  # 40%
    MIN_DIVERSIFICATION_STOCKS = 5
    
    def __init__(self):
        self.market_data = market_data_service
    
    def calculate_portfolio_value(self, holdings: List[Holding]) -> Dict[str, Any]:
        """Calculate current portfolio value and P&L."""
        total_value = 0.0
        total_investment = 0.0
        
        enriched_holdings = []
        
        for holding in holdings:
            current_price = self.market_data.get_current_price(holding.symbol)
            
            if current_price is None:
                current_price = holding.average_price  # Fallback
            
            investment = holding.quantity * holding.average_price
            current_value = holding.quantity * current_price
            pnl = current_value - investment
            pnl_percentage = (pnl / investment * 100) if investment > 0 else 0
            
            holding.current_price = current_price
            holding.current_value = current_value
            holding.pnl = pnl
            holding.pnl_percentage = pnl_percentage
            holding.sector = self.market_data.get_sector(holding.symbol)
            
            total_value += current_value
            total_investment += investment
            enriched_holdings.append(holding)
        
        total_pnl = total_value - total_investment
        total_pnl_percentage = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        
        return {
            "total_value": round(total_value, 2),
            "total_investment": round(total_investment, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_percentage": round(total_pnl_percentage, 2),
            "holdings": enriched_holdings
        }
    
    def calculate_portfolio_volatility(self, holdings: List[Holding]) -> float:
        """Calculate portfolio volatility based on historical returns."""
        if not holdings:
            return 0.0
        
        # Get historical data for each holding
        returns_data = []
        weights = []
        
        total_value = sum(
            h.quantity * (self.market_data.get_current_price(h.symbol) or h.average_price)
            for h in holdings
        )
        
        for holding in holdings:
            df = self.market_data.get_stock_data(holding.symbol, period="1y")
            
            if not df.empty and len(df) > 20:
                # Calculate daily returns
                daily_returns = df['Close'].pct_change().dropna()
                returns_data.append(daily_returns)
                
                # Calculate weight
                current_price = self.market_data.get_current_price(holding.symbol) or holding.average_price
                weight = (holding.quantity * current_price) / total_value if total_value > 0 else 0
                weights.append(weight)
        
        if not returns_data:
            return 0.0
        
        # Align all return series to same index
        min_length = min(len(r) for r in returns_data)
        aligned_returns = [r.tail(min_length).values for r in returns_data]
        
        # Create returns matrix
        returns_matrix = np.column_stack(aligned_returns)
        weights_array = np.array(weights)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(returns_matrix.T)
        
        # Portfolio variance (annualized)
        if weights_array.shape[0] == cov_matrix.shape[0]:
            portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
            # Annualize (252 trading days)
            portfolio_volatility = np.sqrt(portfolio_variance * 252)
        else:
            # Simple average volatility
            individual_vols = [np.std(r) * np.sqrt(252) for r in returns_data]
            portfolio_volatility = np.average(individual_vols, weights=weights)
        
        return float(round(portfolio_volatility, 4))
    
    def calculate_sharpe_ratio(
        self, holdings: List[Holding], 
        volatility: Optional[float] = None
    ) -> float:
        """Calculate portfolio Sharpe ratio."""
        if not holdings:
            return 0.0
        
        # Calculate portfolio return
        total_investment = sum(h.quantity * h.average_price for h in holdings)
        total_value = sum(
            h.quantity * (self.market_data.get_current_price(h.symbol) or h.average_price)
            for h in holdings
        )
        
        if total_investment <= 0:
            return 0.0
        
        # Calculate annualized return (assuming 1-year holding)
        portfolio_return = (total_value - total_investment) / total_investment
        
        if volatility is None:
            volatility = self.calculate_portfolio_volatility(holdings)
        
        if volatility <= 0:
            return 0.0
        
        sharpe_ratio = (portfolio_return - self.RISK_FREE_RATE) / volatility
        
        return round(sharpe_ratio, 3)
    
    def calculate_max_drawdown(self, holdings: List[Holding]) -> float:
        """Calculate maximum drawdown of the portfolio."""
        if not holdings:
            return 0.0
        
        # Build portfolio value series
        total_value = sum(
            h.quantity * (self.market_data.get_current_price(h.symbol) or h.average_price)
            for h in holdings
        )
        
        # Get historical data and calculate weighted portfolio value
        portfolio_values = None
        
        for holding in holdings:
            df = self.market_data.get_stock_data(holding.symbol, period="1y")
            
            if not df.empty:
                weight = (holding.quantity * (self.market_data.get_current_price(holding.symbol) or holding.average_price)) / total_value
                weighted_values = df['Close'] * weight
                
                if portfolio_values is None:
                    portfolio_values = weighted_values
                else:
                    # Align indices
                    common_index = portfolio_values.index.intersection(weighted_values.index)
                    portfolio_values = portfolio_values.loc[common_index] + weighted_values.loc[common_index]
        
        if portfolio_values is None or len(portfolio_values) < 2:
            return 0.0
        
        # Calculate drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        return round(max_drawdown, 4)
    
    def calculate_beta(self, holdings: List[Holding]) -> float:
        """Calculate portfolio beta relative to NIFTY 50."""
        if not holdings:
            return 1.0
        
        # Get NIFTY 50 data
        nifty_df = self.market_data.get_stock_data("^NSEI", period="1y")
        
        if nifty_df.empty:
            return 1.0
        
        nifty_returns = nifty_df['Close'].pct_change().dropna()
        
        # Calculate weighted portfolio beta
        betas = []
        weights = []
        
        total_value = sum(
            h.quantity * (self.market_data.get_current_price(h.symbol) or h.average_price)
            for h in holdings
        )
        
        for holding in holdings:
            df = self.market_data.get_stock_data(holding.symbol, period="1y")
            
            if not df.empty and len(df) > 20:
                stock_returns = df['Close'].pct_change().dropna()
                
                # Align returns
                common_index = stock_returns.index.intersection(nifty_returns.index)
                
                if len(common_index) > 20:
                    aligned_stock = stock_returns.loc[common_index]
                    aligned_nifty = nifty_returns.loc[common_index]
                    
                    # Calculate beta using covariance
                    covariance = np.cov(aligned_stock, aligned_nifty)[0, 1]
                    nifty_variance = np.var(aligned_nifty)
                    
                    if nifty_variance > 0:
                        beta = covariance / nifty_variance
                        betas.append(beta)
                        
                        current_price = self.market_data.get_current_price(holding.symbol) or holding.average_price
                        weight = (holding.quantity * current_price) / total_value
                        weights.append(weight)
        
        if betas and weights:
            portfolio_beta = np.average(betas, weights=weights)
            return float(round(portfolio_beta, 3))
        
        return 1.0
    
    def calculate_sector_exposure(self, holdings: List[Holding]) -> List[SectorExposure]:
        """Calculate sector-wise allocation."""
        sector_values = {}
        total_value = 0.0
        
        for holding in holdings:
            sector = self.market_data.get_sector(holding.symbol)
            current_price = self.market_data.get_current_price(holding.symbol) or holding.average_price
            value = holding.quantity * current_price
            
            if sector not in sector_values:
                sector_values[sector] = 0.0
            sector_values[sector] += value
            total_value += value
        
        exposures = []
        for sector, value in sector_values.items():
            percentage = (value / total_value * 100) if total_value > 0 else 0
            exposures.append(SectorExposure(
                sector=sector,
                percentage=round(percentage, 2),
                value=round(value, 2)
            ))
        
        # Sort by percentage descending
        exposures.sort(key=lambda x: x.percentage, reverse=True)
        
        return exposures
    
    def calculate_concentration_risk(self, holdings: List[Holding]) -> float:
        """Calculate portfolio concentration risk using Herfindahl Index."""
        if not holdings:
            return 0.0
        
        total_value = sum(
            h.quantity * (self.market_data.get_current_price(h.symbol) or h.average_price)
            for h in holdings
        )
        
        if total_value <= 0:
            return 0.0
        
        # Calculate weights
        weights = []
        for holding in holdings:
            current_price = self.market_data.get_current_price(holding.symbol) or holding.average_price
            weight = (holding.quantity * current_price) / total_value
            weights.append(weight)
        
        # Herfindahl Index (sum of squared weights)
        hhi = sum(w ** 2 for w in weights)
        
        # Normalize: 0 = perfectly diversified, 1 = single stock
        # Minimum HHI = 1/n where n is number of stocks
        n = len(holdings)
        if n > 1:
            normalized_hhi = (hhi - 1/n) / (1 - 1/n)
        else:
            normalized_hhi = 1.0
        
        return round(max(0, normalized_hhi), 4)
    
    def calculate_risk_score(
        self, 
        volatility: float, 
        concentration: float, 
        sector_concentration: float,
        beta: float
    ) -> float:
        """Calculate overall risk score (0-1)."""
        # Weight different risk factors
        vol_weight = 0.30
        conc_weight = 0.25
        sector_weight = 0.25
        beta_weight = 0.20
        
        # Normalize volatility (typical range 0-0.5)
        vol_score = min(volatility / 0.5, 1.0)
        
        # Concentration is already 0-1
        conc_score = concentration
        
        # Sector concentration
        sector_score = min(sector_concentration / 0.6, 1.0)
        
        # Beta (typical range 0-2, with 1 being neutral)
        beta_score = min(abs(beta - 1) / 1, 1.0)  # Deviation from 1
        
        risk_score = (
            vol_weight * vol_score +
            conc_weight * conc_score +
            sector_weight * sector_score +
            beta_weight * beta_score
        )
        
        return round(risk_score, 3)
    
    def get_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert risk score to risk level."""
        if risk_score < 0.25:
            return RiskLevel.LOW
        elif risk_score < 0.50:
            return RiskLevel.MODERATE
        elif risk_score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def generate_suggestions(
        self,
        holdings: List[Holding],
        sector_exposure: List[SectorExposure],
        concentration_risk: float,
        volatility: float,
        beta: float
    ) -> List[str]:
        """Generate portfolio improvement suggestions."""
        suggestions = []
        
        # Diversification suggestions
        if len(holdings) < self.MIN_DIVERSIFICATION_STOCKS:
            suggestions.append(
                f"Consider adding more stocks. Current portfolio has only {len(holdings)} stocks. "
                f"Aim for at least {self.MIN_DIVERSIFICATION_STOCKS} for better diversification."
            )
        
        # Concentration suggestions
        if concentration_risk > 0.5:
            top_holding = max(holdings, key=lambda h: h.quantity * (h.current_price or h.average_price))
            suggestions.append(
                f"High concentration risk detected. Consider reducing position in {top_holding.symbol}."
            )
        
        # Sector exposure suggestions
        for exposure in sector_exposure:
            if exposure.percentage > self.MAX_SECTOR_CONCENTRATION * 100:
                suggestions.append(
                    f"High exposure to {exposure.sector} sector ({exposure.percentage:.1f}%). "
                    f"Consider diversifying into other sectors."
                )
        
        # Missing sectors
        all_sectors = {"IT", "Banking", "FMCG", "Pharma", "Energy", "Automobile"}
        current_sectors = {e.sector for e in sector_exposure}
        missing_sectors = all_sectors - current_sectors
        
        if len(missing_sectors) >= 3:
            suggestions.append(
                f"Consider adding exposure to: {', '.join(list(missing_sectors)[:3])} sectors."
            )
        
        # Volatility suggestions
        if volatility > self.VOLATILITY_THRESHOLDS["high"]:
            suggestions.append(
                "Portfolio volatility is high. Consider adding defensive stocks or blue-chips."
            )
        
        # Beta suggestions
        if beta > 1.3:
            suggestions.append(
                f"Portfolio beta is {beta:.2f}, indicating higher market risk. "
                "Consider adding low-beta defensive stocks."
            )
        elif beta < 0.7:
            suggestions.append(
                f"Portfolio beta is {beta:.2f}. If you want market-linked returns, "
                "consider adding some high-growth stocks."
            )
        
        return suggestions[:5]  # Limit suggestions
    
    def analyze_portfolio(self, holdings: List[Holding], user_id: int) -> RiskMetrics:
        """Perform complete risk analysis on portfolio."""
        if not holdings:
            return RiskMetrics(
                risk_score=0,
                risk_level=RiskLevel.LOW,
                volatility=0,
                sharpe_ratio=0,
                max_drawdown=0,
                beta=1,
                sector_exposure=[],
                concentration_risk=0,
                suggestions=["No holdings found. Add stocks to start analysis."]
            )
        
        # Calculate metrics
        volatility = self.calculate_portfolio_volatility(holdings)
        sharpe_ratio = self.calculate_sharpe_ratio(holdings, volatility)
        max_drawdown = self.calculate_max_drawdown(holdings)
        beta = self.calculate_beta(holdings)
        sector_exposure = self.calculate_sector_exposure(holdings)
        concentration_risk = self.calculate_concentration_risk(holdings)
        
        # Get max sector concentration
        max_sector_conc = sector_exposure[0].percentage / 100 if sector_exposure else 0
        
        # Calculate overall risk score
        risk_score = self.calculate_risk_score(
            volatility, concentration_risk, max_sector_conc, beta
        )
        
        risk_level = self.get_risk_level(risk_score)
        
        # Generate suggestions
        suggestions = self.generate_suggestions(
            holdings, sector_exposure, concentration_risk, volatility, beta
        )
        
        return RiskMetrics(
            risk_score=risk_score,
            risk_level=risk_level,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            beta=beta,
            sector_exposure=sector_exposure,
            concentration_risk=concentration_risk,
            suggestions=suggestions
        )
    
    def get_portfolio_health(
        self, 
        holdings: List[Holding],
        risk_metrics: RiskMetrics
    ) -> PortfolioHealth:
        """Assess overall portfolio health."""
        # Base health score (0-100)
        base_score = 70
        
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Adjust based on risk
        if risk_metrics.risk_level == RiskLevel.LOW:
            base_score += 15
            strengths.append("Low overall risk profile")
        elif risk_metrics.risk_level == RiskLevel.MODERATE:
            base_score += 5
            strengths.append("Balanced risk profile")
        elif risk_metrics.risk_level == RiskLevel.HIGH:
            base_score -= 10
            weaknesses.append("High risk profile")
        else:
            base_score -= 20
            weaknesses.append("Very high risk requires immediate attention")
        
        # Adjust based on Sharpe ratio
        if risk_metrics.sharpe_ratio > 1.5:
            base_score += 10
            strengths.append(f"Excellent risk-adjusted returns (Sharpe: {risk_metrics.sharpe_ratio})")
        elif risk_metrics.sharpe_ratio > 0.5:
            base_score += 5
            strengths.append("Good risk-adjusted returns")
        elif risk_metrics.sharpe_ratio < 0:
            base_score -= 10
            weaknesses.append("Negative risk-adjusted returns")
        
        # Adjust based on diversification
        if len(holdings) >= 10:
            base_score += 5
            strengths.append("Well-diversified across multiple stocks")
        elif len(holdings) < 5:
            base_score -= 10
            weaknesses.append("Portfolio lacks diversification")
            recommendations.append("Add more stocks to reduce concentration risk")
        
        # Sector diversification
        sectors = {e.sector for e in risk_metrics.sector_exposure}
        if len(sectors) >= 5:
            base_score += 5
            strengths.append("Good sector diversification")
        elif len(sectors) <= 2:
            base_score -= 10
            weaknesses.append("Limited sector exposure")
            recommendations.append("Diversify across more sectors")
        
        # Adjust based on concentration
        if risk_metrics.concentration_risk < 0.3:
            base_score += 5
            strengths.append("Low concentration risk")
        elif risk_metrics.concentration_risk > 0.6:
            base_score -= 10
            weaknesses.append("High concentration in few stocks")
        
        # Add risk engine suggestions to recommendations
        recommendations.extend(risk_metrics.suggestions)
        
        # Determine health status
        if base_score >= 80:
            health_status = "Excellent"
        elif base_score >= 60:
            health_status = "Good"
        elif base_score >= 40:
            health_status = "Fair"
        else:
            health_status = "Poor"
        
        return PortfolioHealth(
            health_score=max(0, min(100, base_score)),
            health_status=health_status,
            strengths=strengths[:5],
            weaknesses=weaknesses[:5],
            recommendations=recommendations[:5]
        )


# Global instance
risk_engine = PortfolioRiskEngine()
