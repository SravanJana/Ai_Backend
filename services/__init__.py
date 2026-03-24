"""
AI Trading Copilot - Services Module
"""
from services.market_data import market_data_service, MarketDataService
from services.news_service import news_service, sentiment_analyzer, NewsService, SentimentAnalyzer
from services.risk_engine import risk_engine, PortfolioRiskEngine
from services.portfolio_service import portfolio_service, PortfolioService
from services.chatbot import trading_copilot, TradingCopilot

__all__ = [
    "market_data_service",
    "MarketDataService",
    "news_service",
    "sentiment_analyzer",
    "NewsService",
    "SentimentAnalyzer",
    "risk_engine",
    "PortfolioRiskEngine",
    "portfolio_service",
    "PortfolioService",
    "trading_copilot",
    "TradingCopilot"
]
