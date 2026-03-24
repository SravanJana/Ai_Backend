"""
AI Trading Copilot - Stocks API Endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional

from models import StockAnalysis, MarketOverview, OverallSentiment
from services.market_data import market_data_service
from services.news_service import news_service

router = APIRouter(prefix="/ai", tags=["Stock Analysis"])


@router.get("/stock-analysis/{symbol}", response_model=StockAnalysis)
async def get_stock_analysis(symbol: str):
    """
    Get comprehensive AI-powered stock analysis.
    
    Returns:
    - Current price
    - Trend direction (Bullish/Bearish/Neutral)
    - Trading signal (Buy/Sell/Hold)
    - Technical indicators (RSI, MACD, Moving Averages)
    - Support and resistance levels
    - Target price and stop loss
    - Analysis summary
    """
    try:
        analysis = market_data_service.analyze_stock(symbol)
        
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Unable to analyze {symbol}. Stock not found or insufficient data."
            )
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing stock: {str(e)}")


@router.get("/stock-info/{symbol}")
async def get_stock_info(symbol: str):
    """Get basic stock information."""
    try:
        info = market_data_service.get_stock_info(symbol)
        
        if "error" in info:
            raise HTTPException(status_code=404, detail=info["error"])
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock info: {str(e)}")


@router.get("/stock-price/{symbol}")
async def get_stock_price(symbol: str):
    """Get current stock price."""
    try:
        price = market_data_service.get_current_price(symbol)
        
        if price is None:
            raise HTTPException(status_code=404, detail=f"Price not available for {symbol}")
        
        return {
            "symbol": symbol.upper(),
            "price": price
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching price: {str(e)}")


@router.get("/stock-sentiment/{symbol}", response_model=OverallSentiment)
async def get_stock_sentiment(symbol: str):
    """
    Get news sentiment analysis for a stock.
    
    Returns:
    - Overall sentiment (Positive/Negative/Neutral)
    - Sentiment score (-1 to 1)
    - Confidence level
    - Recent news with individual sentiment
    """
    try:
        sentiment = news_service.analyze_news_sentiment(symbol)
        return sentiment
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")


@router.get("/stock-history/{symbol}")
async def get_stock_history(symbol: str, period: str = "3mo"):
    """
    Get historical price data for charting.
    
    Args:
        symbol: Stock symbol (e.g., RELIANCE, TCS)
        period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y)
    
    Returns:
        List of OHLCV data points for charting
    """
    try:
        df = market_data_service.get_stock_data(symbol, period)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to list of dicts for JSON response
        history = []
        for date, row in df.iterrows():
            # Convert pandas Timestamp to string
            date_str = str(date)[:10]  # Get YYYY-MM-DD format
            history.append({
                "date": date_str,
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]) if row["Volume"] else 0
            })
        
        return {
            "symbol": symbol.upper(),
            "period": period,
            "data": history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


@router.get("/market-overview", response_model=MarketOverview)
async def get_market_overview():
    """
    Get comprehensive market overview.
    
    Returns:
    - Market indices (NIFTY, SENSEX, etc.)
    - Top gainers and losers
    - Market breadth
    - Sector performance
    - Overall market sentiment
    """
    try:
        overview = market_data_service.get_market_overview()
        return overview
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market overview: {str(e)}")


@router.get("/market-sentiment")
async def get_market_sentiment():
    """Get overall market sentiment from news analysis."""
    try:
        sentiment = news_service.get_market_sentiment()
        return sentiment
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing market sentiment: {str(e)}")


@router.get("/market-news")
async def get_market_news(limit: int = 10):
    """
    Get latest market news with sentiment analysis.
    
    Args:
        limit: Maximum number of news items to return (default: 10)
    
    Returns:
        List of news items with headline, source, time, and sentiment
    """
    try:
        from datetime import datetime
        
        news_items = news_service.fetch_news(limit=limit)
        
        # Add sentiment analysis to each news item
        analyzed_news = []
        for item in news_items:
            sentiment_result = news_service.sentiment_analyzer.analyze(item.get("headline", ""))
            sentiment = sentiment_result["sentiment"].value if hasattr(sentiment_result["sentiment"], "value") else str(sentiment_result["sentiment"])
            
            # Parse published time
            published = item.get("published", "")
            try:
                # Try to parse the date and format it as relative time
                if published:
                    from dateutil import parser
                    pub_date = parser.parse(published)
                    now = datetime.now(pub_date.tzinfo) if pub_date.tzinfo else datetime.now()
                    diff = now - pub_date
                    
                    if diff.days > 0:
                        time_ago = f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
                    elif diff.seconds >= 3600:
                        hours = diff.seconds // 3600
                        time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
                    elif diff.seconds >= 60:
                        minutes = diff.seconds // 60
                        time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
                    else:
                        time_ago = "Just now"
                else:
                    time_ago = "Recently"
            except:
                time_ago = "Recently"
            
            analyzed_news.append({
                "title": item.get("headline", ""),
                "source": item.get("source", "Unknown"),
                "time": time_ago,
                "sentiment": sentiment.lower() if isinstance(sentiment, str) else "neutral",
                "url": item.get("url", "")
            })
        
        return {"news": analyzed_news}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market news: {str(e)}")


@router.get("/top-movers")
async def get_top_movers(count: int = 5, sector: str = "allSec"):
    """
    Get top gainers and losers.
    
    Args:
        count: Number of stocks to return in each category (default: 5)
        sector: Market sector/index to filter by:
            - NIFTY: NIFTY 50 stocks
            - BANKNIFTY: Bank Nifty stocks
            - NIFTYNEXT50: NIFTY Next 50 stocks
            - SecGtr20: Securities > Rs 20
            - SecLwr20: Securities < Rs 20
            - FOSec: F&O Securities
            - allSec: All Securities (default)
    """
    try:
        movers = market_data_service.get_top_movers(count=count, sector=sector)
        return movers
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching movers: {str(e)}")


@router.get("/indices")
async def get_indices():
    """Get major market indices."""
    try:
        indices = market_data_service.get_market_indices()
        return {"indices": indices}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching indices: {str(e)}")


@router.get("/compare/{symbol1}/{symbol2}")
async def compare_stocks(symbol1: str, symbol2: str):
    """
    Compare two stocks side by side.
    
    Returns technical analysis and metrics for both stocks.
    """
    try:
        analysis1 = market_data_service.analyze_stock(symbol1)
        analysis2 = market_data_service.analyze_stock(symbol2)
        
        if not analysis1:
            raise HTTPException(status_code=404, detail=f"Unable to analyze {symbol1}")
        if not analysis2:
            raise HTTPException(status_code=404, detail=f"Unable to analyze {symbol2}")
        
        # Build detailed comparison
        comparison = {
            "stocks": [
                {
                    "symbol": analysis1.symbol,
                    "name": analysis1.name,
                    "price": analysis1.current_price,
                    "day_high": analysis1.day_high,
                    "day_low": analysis1.day_low,
                    "year_high": analysis1.year_high,
                    "year_low": analysis1.year_low,
                    "trend": analysis1.trend.value,
                    "signal": analysis1.signal.value,
                    "confidence": analysis1.confidence,
                    "technical_indicators": {
                        "rsi": analysis1.technical_indicators.rsi,
                        "macd": analysis1.technical_indicators.macd,
                        "sma_20": analysis1.technical_indicators.sma_20,
                        "sma_50": analysis1.technical_indicators.sma_50,
                        "sma_200": analysis1.technical_indicators.sma_200,
                    },
                    "support": analysis1.support_level,
                    "resistance": analysis1.resistance_level,
                    "target_price": analysis1.target_price,
                    "stop_loss": analysis1.stop_loss,
                },
                {
                    "symbol": analysis2.symbol,
                    "name": analysis2.name,
                    "price": analysis2.current_price,
                    "day_high": analysis2.day_high,
                    "day_low": analysis2.day_low,
                    "year_high": analysis2.year_high,
                    "year_low": analysis2.year_low,
                    "trend": analysis2.trend.value,
                    "signal": analysis2.signal.value,
                    "confidence": analysis2.confidence,
                    "technical_indicators": {
                        "rsi": analysis2.technical_indicators.rsi,
                        "macd": analysis2.technical_indicators.macd,
                        "sma_20": analysis2.technical_indicators.sma_20,
                        "sma_50": analysis2.technical_indicators.sma_50,
                        "sma_200": analysis2.technical_indicators.sma_200,
                    },
                    "support": analysis2.support_level,
                    "resistance": analysis2.resistance_level,
                    "target_price": analysis2.target_price,
                    "stop_loss": analysis2.stop_loss,
                },
            ],
            "comparison_metrics": [],
            "recommendation": "",
            "winner": None
        }
        
        # Generate comparison metrics
        metrics = []
        
        # RSI comparison
        if analysis1.technical_indicators.rsi < 30:
            rsi1_status = "Oversold (Bullish)"
        elif analysis1.technical_indicators.rsi > 70:
            rsi1_status = "Overbought (Bearish)"
        else:
            rsi1_status = "Neutral"
            
        if analysis2.technical_indicators.rsi < 30:
            rsi2_status = "Oversold (Bullish)"
        elif analysis2.technical_indicators.rsi > 70:
            rsi2_status = "Overbought (Bearish)"
        else:
            rsi2_status = "Neutral"
        
        metrics.append({
            "name": "RSI",
            "stock1_value": round(analysis1.technical_indicators.rsi, 2),
            "stock2_value": round(analysis2.technical_indicators.rsi, 2),
            "stock1_status": rsi1_status,
            "stock2_status": rsi2_status,
        })
        
        # Trend comparison
        metrics.append({
            "name": "Trend",
            "stock1_value": analysis1.trend.value,
            "stock2_value": analysis2.trend.value,
            "stock1_status": "Positive" if analysis1.trend.value == "Bullish" else "Negative" if analysis1.trend.value == "Bearish" else "Neutral",
            "stock2_status": "Positive" if analysis2.trend.value == "Bullish" else "Negative" if analysis2.trend.value == "Bearish" else "Neutral",
        })
        
        # Signal comparison
        metrics.append({
            "name": "Signal",
            "stock1_value": analysis1.signal.value,
            "stock2_value": analysis2.signal.value,
            "stock1_status": "Buy" if "Buy" in analysis1.signal.value else "Sell" if "Sell" in analysis1.signal.value else "Hold",
            "stock2_status": "Buy" if "Buy" in analysis2.signal.value else "Sell" if "Sell" in analysis2.signal.value else "Hold",
        })
        
        # Price vs SMA200 (above = bullish)
        above_sma200_1 = analysis1.current_price > analysis1.technical_indicators.sma_200
        above_sma200_2 = analysis2.current_price > analysis2.technical_indicators.sma_200
        
        metrics.append({
            "name": "Above 200 SMA",
            "stock1_value": "Yes" if above_sma200_1 else "No",
            "stock2_value": "Yes" if above_sma200_2 else "No",
            "stock1_status": "Bullish" if above_sma200_1 else "Bearish",
            "stock2_status": "Bullish" if above_sma200_2 else "Bearish",
        })
        
        # MACD
        macd1_status = "Bullish" if analysis1.technical_indicators.macd > 0 else "Bearish"
        macd2_status = "Bullish" if analysis2.technical_indicators.macd > 0 else "Bearish"
        
        metrics.append({
            "name": "MACD",
            "stock1_value": round(analysis1.technical_indicators.macd, 2),
            "stock2_value": round(analysis2.technical_indicators.macd, 2),
            "stock1_status": macd1_status,
            "stock2_status": macd2_status,
        })
        
        comparison["comparison_metrics"] = metrics
        
        # Calculate overall score
        def calculate_score(analysis):
            score = 0
            # Signal score
            if analysis.signal.value == "Strong Buy":
                score += 3
            elif analysis.signal.value == "Buy":
                score += 2
            elif analysis.signal.value == "Hold":
                score += 1
            elif analysis.signal.value == "Sell":
                score -= 2
            elif analysis.signal.value == "Strong Sell":
                score -= 3
            
            # Trend score
            if analysis.trend.value == "Bullish":
                score += 2
            elif analysis.trend.value == "Bearish":
                score -= 2
            
            # RSI score
            if 30 <= analysis.technical_indicators.rsi <= 70:
                score += 1
            elif analysis.technical_indicators.rsi < 30:
                score += 2  # Oversold is opportunity
            else:
                score -= 1  # Overbought
            
            # Above SMA200
            if analysis.current_price > analysis.technical_indicators.sma_200:
                score += 1
            
            # MACD positive
            if analysis.technical_indicators.macd > 0:
                score += 1
                
            return score
        
        score1 = calculate_score(analysis1)
        score2 = calculate_score(analysis2)
        
        if score1 > score2:
            comparison["winner"] = analysis1.symbol
            comparison["recommendation"] = f"**{analysis1.symbol}** appears to be the stronger pick based on technical analysis. It has a {analysis1.trend.value.lower()} trend with a {analysis1.signal.value.lower()} signal and better overall technical positioning."
        elif score2 > score1:
            comparison["winner"] = analysis2.symbol
            comparison["recommendation"] = f"**{analysis2.symbol}** appears to be the stronger pick based on technical analysis. It has a {analysis2.trend.value.lower()} trend with a {analysis2.signal.value.lower()} signal and better overall technical positioning."
        else:
            comparison["winner"] = "Tie"
            comparison["recommendation"] = f"Both **{analysis1.symbol}** and **{analysis2.symbol}** have similar technical profiles. Consider other factors like fundamentals, sector outlook, and your portfolio allocation before making a decision."
        
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing stocks: {str(e)}")


@router.get("/available-stocks")
async def get_available_stocks():
    """Get list of available stocks for analysis."""
    from services.market_data import INDIAN_STOCKS, STOCK_SECTORS
    
    stocks = []
    for symbol in INDIAN_STOCKS.keys():
        stocks.append({
            "symbol": symbol,
            "sector": STOCK_SECTORS.get(symbol, "Other")
        })
    
    return {"stocks": stocks}
