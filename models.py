"""
AI Trading Copilot - Data Models
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============= Enums =============

class RiskLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"


class TrendDirection(str, Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"


class SignalType(str, Enum):
    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"
    STRONG_BUY = "Strong Buy"
    STRONG_SELL = "Strong Sell"


class SentimentType(str, Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"


# ============= Holdings Models =============

class Holding(BaseModel):
    """Individual stock holding."""
    symbol: str
    quantity: int = Field(alias="qty", default=0)
    average_price: float = Field(alias="avg_price", default=0.0)
    current_price: Optional[float] = None
    current_value: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    sector: Optional[str] = None
    
    class Config:
        populate_by_name = True


class Portfolio(BaseModel):
    """User portfolio containing multiple holdings."""
    user_id: int
    holdings: List[Holding]
    total_value: float = 0.0
    total_investment: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percentage: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.now)


# ============= Risk Models =============

class SectorExposure(BaseModel):
    """Sector allocation data."""
    sector: str
    percentage: float
    value: float


class RiskMetrics(BaseModel):
    """Portfolio risk metrics."""
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    sector_exposure: List[SectorExposure]
    concentration_risk: float
    suggestions: List[str]


# ============= Stock Analysis Models =============

class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    atr: float
    volume_sma: float


class StockAnalysis(BaseModel):
    """Complete stock analysis result."""
    symbol: str
    name: Optional[str] = None
    current_price: float
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    year_high: Optional[float] = None
    year_low: Optional[float] = None
    trend: TrendDirection
    signal: SignalType
    confidence: float = Field(ge=0, le=1)
    technical_indicators: TechnicalIndicators
    support_level: float
    resistance_level: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    analysis_summary: str
    timestamp: datetime = Field(default_factory=datetime.now)


# ============= Sentiment Models =============

class NewsSentiment(BaseModel):
    """News sentiment analysis result."""
    headline: str
    source: str
    sentiment: SentimentType
    confidence: float
    url: Optional[str] = None
    published_at: Optional[datetime] = None


class OverallSentiment(BaseModel):
    """Aggregated sentiment for a stock."""
    symbol: str
    overall_sentiment: SentimentType
    sentiment_score: float = Field(ge=-1, le=1)
    confidence: float
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    news_items: List[NewsSentiment]


# ============= Chat Models =============

class ChatMessage(BaseModel):
    """Chat message from user."""
    user_id: int
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """AI chatbot response."""
    response: str
    suggestions: Optional[List[str]] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ============= Market Models =============

class MarketIndex(BaseModel):
    """Market index data."""
    symbol: str
    name: str
    value: float
    change: float
    change_percentage: float
    trend: TrendDirection


class StockMover(BaseModel):
    """Top gainer/loser stock."""
    symbol: str
    name: str
    price: float
    change: float
    change_percentage: float
    volume: int


class MarketOverview(BaseModel):
    """Overall market summary."""
    indices: List[MarketIndex]
    top_gainers: List[StockMover]
    top_losers: List[StockMover]
    market_breadth: Dict[str, int]
    sector_performance: Dict[str, float]
    market_sentiment: SentimentType
    timestamp: datetime = Field(default_factory=datetime.now)


# ============= Portfolio Summary Models =============

class PortfolioHealth(BaseModel):
    """Portfolio health assessment."""
    health_score: float = Field(ge=0, le=100)
    health_status: str
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


class PortfolioSummary(BaseModel):
    """Complete portfolio analysis summary."""
    user_id: int
    portfolio: Portfolio
    risk_metrics: RiskMetrics
    health: PortfolioHealth
    top_performers: List[Holding]
    worst_performers: List[Holding]
    ai_insights: str
    rebalance_suggestions: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ============= Watchlist Models =============

class WatchlistItem(BaseModel):
    """Watchlist stock item."""
    symbol: str
    name: str
    current_price: float
    change_percentage: float
    alert_price: Optional[float] = None
    notes: Optional[str] = None


class Alert(BaseModel):
    """Trading alert."""
    alert_id: str
    symbol: str
    alert_type: str
    message: str
    priority: str
    triggered_at: datetime = Field(default_factory=datetime.now)
    is_read: bool = False
