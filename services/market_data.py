"""
AI Trading Copilot - Market Data Service
Fetches live market data from Yahoo Finance with Alpha Vantage fallback.
"""
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import ta

from models import (
    StockAnalysis, TechnicalIndicators, TrendDirection, SignalType,
    MarketIndex, StockMover, MarketOverview, SentimentType
)


# Comprehensive Indian Stock List - NIFTY 50 + NIFTY Next 50 + Popular BSE Stocks (~150 stocks)
ALL_INDIAN_STOCKS = {
    # NIFTY 50
    "ADANIENT": "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "AXISBANK": "AXISBANK.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "BPCL": "BPCL.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "CIPLA": "CIPLA.NS",
    "COALINDIA": "COALINDIA.NS",
    "DIVISLAB": "DIVISLAB.NS",
    "DRREDDY": "DRREDDY.NS",
    "EICHERMOT": "EICHERMOT.NS",
    "GRASIM": "GRASIM.NS",
    "HCLTECH": "HCLTECH.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "HDFCLIFE": "HDFCLIFE.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "HINDALCO": "HINDALCO.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "INFY": "INFY.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "LT": "LT.NS",
    "M&M": "M&M.NS",
    "MARUTI": "MARUTI.NS",
    "NTPC": "NTPC.NS",
    "NESTLEIND": "NESTLEIND.NS",
    "ONGC": "ONGC.NS",
    "POWERGRID": "POWERGRID.NS",
    "RELIANCE": "RELIANCE.NS",
    "SBILIFE": "SBILIFE.NS",
    "SBIN": "SBIN.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TCS": "TCS.NS",
    "TATACONSUM": "TATACONSUM.NS",
    "TATAMTRDVR": "TATAMTRDVR.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "TECHM": "TECHM.NS",
    "TITAN": "TITAN.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "UPL": "UPL.NS",
    "WIPRO": "WIPRO.NS",
    
    # NIFTY Next 50
    "ABB": "ABB.NS",
    "ADANIGREEN": "ADANIGREEN.NS",
    "ATGL": "ATGL.NS",
    "AMBUJACEM": "AMBUJACEM.NS",
    "AUROPHARMA": "AUROPHARMA.NS",
    "BAJAJHLDNG": "BAJAJHLDNG.NS",
    "BANKBARODA": "BANKBARODA.NS",
    "BERGEPAINT": "BERGEPAINT.NS",
    "BIOCON": "BIOCON.NS",
    "BOSCHLTD": "BOSCHLTD.NS",
    "CANBK": "CANBK.NS",
    "CHOLAFIN": "CHOLAFIN.NS",
    "COLPAL": "COLPAL.NS",
    "CONCOR": "CONCOR.NS",
    "DLF": "DLF.NS",
    "DABUR": "DABUR.NS",
    "DMART": "DMART.NS",
    "GAIL": "GAIL.NS",
    "GODREJCP": "GODREJCP.NS",
    "GODREJPROP": "GODREJPROP.NS",
    "HAVELLS": "HAVELLS.NS",
    "HINDPETRO": "HINDPETRO.NS",
    "ICICIGI": "ICICIGI.NS",
    "ICICIPRULI": "ICICIPRULI.NS",
    "INDUSTOWER": "INDUSTOWER.NS",
    "IOC": "IOC.NS",
    "IRCTC": "IRCTC.NS",
    "JINDALSTEL": "JINDALSTEL.NS",
    "JUBLFOOD": "JUBLFOOD.NS",
    "LICI": "LICI.NS",
    "LUPIN": "LUPIN.NS",
    "MARICO": "MARICO.NS",
    "UNITDSPR": "UNITDSPR.NS",
    "MOTHERSON": "MOTHERSON.NS",
    "MUTHOOTFIN": "MUTHOOTFIN.NS",
    "NAUKRI": "NAUKRI.NS",
    "NMDC": "NMDC.NS",
    "OBEROIRLTY": "OBEROIRLTY.NS",
    "OFSS": "OFSS.NS",
    "PAGEIND": "PAGEIND.NS",
    "PIRAMAL": "PIRAMAL.NS",
    "PETRONET": "PETRONET.NS",
    "PIDILITIND": "PIDILITIND.NS",
    "PNB": "PNB.NS",
    "SAIL": "SAIL.NS",
    "SHREECEM": "SHREECEM.NS",
    "SIEMENS": "SIEMENS.NS",
    "SRF": "SRF.NS",
    "TATAPOWER": "TATAPOWER.NS",
    "TORNTPHARM": "TORNTPHARM.NS",
    "TRENT": "TRENT.NS",
    "VEDL": "VEDL.NS",
    "ZOMATO": "ZOMATO.NS",
    "ZYDUSLIFE": "ZYDUSLIFE.NS",
    
    # Other Popular NSE/BSE Stocks
    "ACC": "ACC.NS",
    "ADANIPOWER": "ADANIPOWER.NS",
    "ALKEM": "ALKEM.NS",
    "ASHOKLEY": "ASHOKLEY.NS",
    "ASTRAL": "ASTRAL.NS",
    "ATUL": "ATUL.NS",
    "AUBANK": "AUBANK.NS",
    "BALKRISIND": "BALKRISIND.NS",
    "BANDHANBNK": "BANDHANBNK.NS",
    "BEL": "BEL.NS",
    "BHEL": "BHEL.NS",
    # CADILAHC merged into ZYDUSLIFE (already in list)
    "CANFINHOME": "CANFINHOME.NS",
    "CENTRALBK": "CENTRALBK.NS",
    "COFORGE": "COFORGE.NS",
    "CUMMINSIND": "CUMMINSIND.NS",
    "DEEPAKNTR": "DEEPAKNTR.NS",
    "DELTACORP": "DELTACORP.NS",
    "DIXON": "DIXON.NS",
    "ESCORTS": "ESCORTS.NS",
    "EXIDEIND": "EXIDEIND.NS",
    "FEDERALBNK": "FEDERALBNK.NS",
    "GLENMARK": "GLENMARK.NS",
    "GMRAIRPORT": "GMRAIRPORT.NS",
    "GRANULES": "GRANULES.NS",
    "HAL": "HAL.NS",
    "IDFCFIRSTB": "IDFCFIRSTB.NS",
    "IEX": "IEX.NS",
    "INDHOTEL": "INDHOTEL.NS",
    "INDIAMART": "INDIAMART.NS",
    "INDIANB": "INDIANB.NS",
    "INDIGO": "INDIGO.NS",
    "IPCALAB": "IPCALAB.NS",
    "IRFC": "IRFC.NS",
    "JKCEMENT": "JKCEMENT.NS",
    "JSWENERGY": "JSWENERGY.NS",
    "LAURUSLABS": "LAURUSLABS.NS",
    "LICHSGFIN": "LICHSGFIN.NS",
    "LTIM": "LTIM.NS",
    "LTTS": "LTTS.NS",
    "MANAPPURAM": "MANAPPURAM.NS",
    "MFSL": "MFSL.NS",
    "MGL": "MGL.NS",
    # MINDTREE merged into LTIM (already in list)
    "MPHASIS": "MPHASIS.NS",
    "NAM-INDIA": "NAM-INDIA.NS",
    "NATIONALUM": "NATIONALUM.NS",
    "NAVINFLUOR": "NAVINFLUOR.NS",
    "NYKAA": "NYKAA.NS",
    "PAYTM": "PAYTM.NS",
    "PERSISTENT": "PERSISTENT.NS",
    "PIIND": "PIIND.NS",
    "POLYCAB": "POLYCAB.NS",
    "PVRINOX": "PVRINOX.NS",
    "RAMCOCEM": "RAMCOCEM.NS",
    "RBLBANK": "RBLBANK.NS",
    "RECLTD": "RECLTD.NS",
    "SBICARD": "SBICARD.NS",
    "SCHAEFFLER": "SCHAEFFLER.NS",
    "SHRIRAMFIN": "SHRIRAMFIN.NS",
    "STAR": "STAR.NS",
    "SUNTV": "SUNTV.NS",
    "TATACOMM": "TATACOMM.NS",
    "TATAELXSI": "TATAELXSI.NS",
    "TATACHEM": "TATACHEM.NS",
    "TORNTPOWER": "TORNTPOWER.NS",
    "UNIONBANK": "UNIONBANK.NS",
    "UBL": "UBL.NS",
    "VOLTAS": "VOLTAS.NS",
    "WHIRLPOOL": "WHIRLPOOL.NS",
    "YESBANK": "YESBANK.NS",
}

# Legacy mappings for backward compatibility
NIFTY_50_STOCKS = ALL_INDIAN_STOCKS
INDIAN_STOCKS = ALL_INDIAN_STOCKS

# Sector mapping for all stocks
STOCK_SECTORS = {
    # NIFTY 50
    "ADANIENT": "Infrastructure",
    "ADANIPORTS": "Infrastructure",
    "APOLLOHOSP": "Healthcare",
    "ASIANPAINT": "Consumer Goods",
    "AXISBANK": "Banking",
    "BAJAJ-AUTO": "Automobile",
    "BAJFINANCE": "Financial Services",
    "BAJAJFINSV": "Financial Services",
    "BPCL": "Energy",
    "BHARTIARTL": "Telecom",
    "BRITANNIA": "FMCG",
    "CIPLA": "Pharma",
    "COALINDIA": "Mining",
    "DIVISLAB": "Pharma",
    "DRREDDY": "Pharma",
    "EICHERMOT": "Automobile",
    "GRASIM": "Cement",
    "HCLTECH": "IT",
    "HDFCBANK": "Banking",
    "HDFCLIFE": "Insurance",
    "HEROMOTOCO": "Automobile",
    "HINDALCO": "Metals",
    "HINDUNILVR": "FMCG",
    "ICICIBANK": "Banking",
    "ITC": "FMCG",
    "INDUSINDBK": "Banking",
    "INFY": "IT",
    "JSWSTEEL": "Metals",
    "KOTAKBANK": "Banking",
    "LT": "Infrastructure",
    "M&M": "Automobile",
    "MARUTI": "Automobile",
    "NTPC": "Power",
    "NESTLEIND": "FMCG",
    "ONGC": "Energy",
    "POWERGRID": "Power",
    "RELIANCE": "Energy",
    "SBILIFE": "Insurance",
    "SBIN": "Banking",
    "SUNPHARMA": "Pharma",
    "TCS": "IT",
    "TATACONSUM": "FMCG",
    "TATAMTRDVR": "Automobile",
    "TATASTEEL": "Metals",
    "TECHM": "IT",
    "TITAN": "Consumer Goods",
    "ULTRACEMCO": "Cement",
    "UPL": "Chemicals",
    "WIPRO": "IT",
    # NIFTY Next 50
    "ABB": "Engineering",
    "ADANIGREEN": "Energy",
    "ATGL": "Energy",
    "AMBUJACEM": "Cement",
    "AUROPHARMA": "Pharma",
    "BAJAJHLDNG": "Financial Services",
    "BANKBARODA": "Banking",
    "BERGEPAINT": "Consumer Goods",
    "BIOCON": "Pharma",
    "BOSCHLTD": "Automobile",
    "CANBK": "Banking",
    "CHOLAFIN": "Financial Services",
    "COLPAL": "FMCG",
    "CONCOR": "Logistics",
    "DLF": "Real Estate",
    "DABUR": "FMCG",
    "DMART": "Retail",
    "GAIL": "Energy",
    "GODREJCP": "FMCG",
    "GODREJPROP": "Real Estate",
    "HAVELLS": "Consumer Durables",
    "HINDPETRO": "Energy",
    "ICICIGI": "Insurance",
    "ICICIPRULI": "Insurance",
    "INDUSTOWER": "Telecom",
    "IOC": "Energy",
    "IRCTC": "Travel",
    "JINDALSTEL": "Metals",
    "JUBLFOOD": "FMCG",
    "LICI": "Insurance",
    "LUPIN": "Pharma",
    "MARICO": "FMCG",
    "UNITDSPR": "FMCG",
    "MOTHERSON": "Automobile",
    "MUTHOOTFIN": "Financial Services",
    "NAUKRI": "IT",
    "NMDC": "Mining",
    "OBEROIRLTY": "Real Estate",
    "OFSS": "IT",
    "PAGEIND": "Textiles",
    "PIRAMAL": "Pharma",
    "PETRONET": "Energy",
    "PIDILITIND": "Chemicals",
    "PNB": "Banking",
    "SAIL": "Metals",
    "SHREECEM": "Cement",
    "SIEMENS": "Engineering",
    "SRF": "Chemicals",
    "TATAPOWER": "Power",
    "TORNTPHARM": "Pharma",
    "TRENT": "Retail",
    "VEDL": "Metals",
    "ZOMATO": "Tech",
    "ZYDUSLIFE": "Pharma",
    # Other stocks
    "ACC": "Cement",
    "ADANIPOWER": "Power",
    "ALKEM": "Pharma",
    "ASHOKLEY": "Automobile",
    "ASTRAL": "Chemicals",
    "ATUL": "Chemicals",
    "AUBANK": "Banking",
    "BALKRISIND": "Automobile",
    "BANDHANBNK": "Banking",
    "BEL": "Defence",
    "BHEL": "Engineering",
    "CANFINHOME": "Financial Services",
    # CADILAHC -> ZYDUSLIFE already
    "GMRAIRPORT": "Infrastructure",
    "CENTRALBK": "Banking",
    "COFORGE": "IT",
    "CUMMINSIND": "Engineering",
    "DEEPAKNTR": "Chemicals",
    "DELTACORP": "Hotels",
    "DIXON": "Consumer Durables",
    "ESCORTS": "Automobile",
    "EXIDEIND": "Automobile",
    "FEDERALBNK": "Banking",
    "GLENMARK": "Pharma",
    # GMRINFRA -> GMRAIRPORT already
    "GRANULES": "Pharma",
    "HAL": "Defence",
    "IDFCFIRSTB": "Banking",
    "IEX": "Exchange",
    "INDHOTEL": "Hotels",
    "INDIAMART": "IT",
    "INDIANB": "Banking",
    "INDIGO": "Aviation",
    "IPCALAB": "Pharma",
    "IRFC": "Financial Services",
    "JKCEMENT": "Cement",
    "JSWENERGY": "Power",
    "LAURUSLABS": "Pharma",
    "LICHSGFIN": "Financial Services",
    "LTIM": "IT",
    "LTTS": "IT",
    "MANAPPURAM": "Financial Services",
    "MFSL": "Financial Services",
    "MGL": "Energy",
    # MINDTREE -> LTIM already
    "MPHASIS": "IT",
    "NAM-INDIA": "Financial Services",
    "NATIONALUM": "Metals",
    "NAVINFLUOR": "Chemicals",
    "NYKAA": "Retail",
    "PAYTM": "Fintech",
    "PERSISTENT": "IT",
    "PIIND": "Chemicals",
    "POLYCAB": "Consumer Durables",
    "PVRINOX": "Entertainment",
    "RAMCOCEM": "Cement",
    "RBLBANK": "Banking",
    "RECLTD": "Financial Services",
    "SBICARD": "Financial Services",
    "SCHAEFFLER": "Engineering",
    "SHRIRAMFIN": "Financial Services",
    "STAR": "Media",
    "SUNTV": "Media",
    "TATACOMM": "Telecom",
    "TATAELXSI": "IT",
    "TATACHEM": "Chemicals",
    "TORNTPOWER": "Power",
    "UNIONBANK": "Banking",
    "UBL": "FMCG",
    "VOLTAS": "Consumer Durables",
    "WHIRLPOOL": "Consumer Durables",
    "YESBANK": "Banking",
}

# Market indices
INDICES = {
    "^NSEI": "NIFTY 50",
    "^BSESN": "SENSEX",
    "^NSEBANK": "Bank Nifty",
    "^CNXIT": "Nifty IT",
}


class MarketDataService:
    """Service for fetching and processing market data."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=15)  # Longer cache for slow Yahoo Finance
    
    def _get_yahoo_symbol(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance format."""
        symbol = symbol.upper().strip()
        if symbol in INDIAN_STOCKS:
            return INDIAN_STOCKS[symbol]
        if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
            # Try NSE first
            return f"{symbol}.NS"
        return symbol
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_time:
            return False
        return datetime.now() - self._cache_time[key] < self._cache_duration
    
    def get_stock_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """Fetch historical stock data."""
        cache_key = f"stock_data_{symbol}_{period}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        yahoo_symbol = self._get_yahoo_symbol(symbol)
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                # Try without suffix
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
            
            self._cache[cache_key] = df
            self._cache_time[cache_key] = datetime.now()
            
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price with caching."""
        cache_key = f"price_{symbol}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        yahoo_symbol = self._get_yahoo_symbol(symbol)
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            price = None
            
            # Try multiple price sources
            try:
                price = ticker.fast_info.get("lastPrice") or ticker.fast_info.get("regularMarketPrice")
            except:
                pass
            
            if not price:
                try:
                    info = ticker.info
                    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
                except:
                    pass
            
            # Fallback: get last close from history (works during non-market hours)
            if not price:
                try:
                    hist = ticker.history(period="5d")
                    if not hist.empty:
                        price = hist['Close'].iloc[-1]
                        print(f"Using historical close for {symbol} (market closed)")
                except:
                    pass
            
            if price:
                self._cache[cache_key] = price
                self._cache_time[cache_key] = datetime.now()
                return price
                
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
        
        # Return fallback price for common Indian stocks
        fallback_prices = {
            "INFY": 1520.0, "RELIANCE": 2650.0, "HDFCBANK": 1700.0,
            "TCS": 3750.0, "ITC": 445.0, "ICICIBANK": 1050.0,
            "BHARTIARTL": 1150.0, "WIPRO": 480.0, "HCLTECH": 1400.0,
            "SBIN": 650.0, "MARUTI": 11000.0, "KOTAKBANK": 1800.0,
            "AXISBANK": 1100.0, "BAJFINANCE": 7000.0, "LT": 3500.0,
            "SUNPHARMA": 1200.0, "TITAN": 3200.0, "ASIANPAINT": 2900.0,
            "HINDUNILVR": 2400.0, "ULTRACEMCO": 10500.0
        }
        return fallback_prices.get(symbol.upper())
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock information with fallback for non-market hours."""
        yahoo_symbol = self._get_yahoo_symbol(symbol)
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            # Get current price with fallback to previous close or history
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            previous_close = info.get("previousClose")
            
            # If no current price, try history (works during non-market hours)
            if not current_price:
                try:
                    hist = ticker.history(period="5d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        if not previous_close and len(hist) > 1:
                            previous_close = hist['Close'].iloc[-2]
                except:
                    pass
            
            # Final fallback
            if not current_price:
                current_price = previous_close or self.get_current_price(symbol)
            
            return {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "sector": STOCK_SECTORS.get(symbol.upper(), info.get("sector", "Unknown")),
                "industry": info.get("industry", "Unknown"),
                "current_price": current_price,
                "previous_close": previous_close or current_price,
                "open": info.get("open") or current_price,
                "day_high": info.get("dayHigh") or current_price,
                "day_low": info.get("dayLow") or current_price,
                "volume": info.get("volume") or 0,
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "beta": info.get("beta"),
            }
        except Exception as e:
            print(f"Error getting info for {symbol}: {e}")
            # Return basic fallback data
            fallback_price = self.get_current_price(symbol)
            return {
                "symbol": symbol,
                "name": symbol,
                "sector": STOCK_SECTORS.get(symbol.upper(), "Unknown"),
                "industry": "Unknown",
                "current_price": fallback_price,
                "previous_close": fallback_price,
                "error": str(e)
            }
    
    def calculate_technical_indicators(self, symbol: str) -> Optional[TechnicalIndicators]:
        """Calculate technical indicators for a stock."""
        df = self.get_stock_data(symbol, period="6mo")
        
        if df.empty or len(df) < 50:
            return None
        
        try:
            # RSI
            rsi = ta.momentum.RSIIndicator(df['Close'], window=14)
            rsi_value = rsi.rsi().iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            macd_value = macd.macd().iloc[-1]
            macd_signal = macd.macd_signal().iloc[-1]
            macd_histogram = macd.macd_diff().iloc[-1]
            
            # Moving Averages
            sma_20 = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator().iloc[-1]
            sma_50 = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator().iloc[-1]
            sma_200 = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator().iloc[-1] if len(df) >= 200 else sma_50
            ema_12 = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator().iloc[-1]
            ema_26 = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator().iloc[-1]
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'], window=20)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_middle = bb.bollinger_mavg().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            
            # ATR
            atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range().iloc[-1]
            
            # Volume SMA
            volume_sma = df['Volume'].rolling(window=20).mean().iloc[-1]
            
            return TechnicalIndicators(
                rsi=round(rsi_value, 2),
                macd=round(macd_value, 4),
                macd_signal=round(macd_signal, 4),
                macd_histogram=round(macd_histogram, 4),
                sma_20=round(sma_20, 2),
                sma_50=round(sma_50, 2),
                sma_200=round(sma_200, 2),
                ema_12=round(ema_12, 2),
                ema_26=round(ema_26, 2),
                bollinger_upper=round(bb_upper, 2),
                bollinger_middle=round(bb_middle, 2),
                bollinger_lower=round(bb_lower, 2),
                atr=round(atr, 2),
                volume_sma=round(volume_sma, 0)
            )
        except Exception as e:
            print(f"Error calculating indicators for {symbol}: {e}")
            return None
    
    def analyze_stock(self, symbol: str) -> Optional[StockAnalysis]:
        """Perform complete technical analysis on a stock."""
        stock_info = self.get_stock_info(symbol)
        indicators = self.calculate_technical_indicators(symbol)
        
        if not indicators or not stock_info.get("current_price"):
            return None
        
        current_price = stock_info["current_price"]
        
        # Determine trend
        trend = TrendDirection.NEUTRAL
        if current_price > indicators.sma_50 > indicators.sma_200:
            trend = TrendDirection.BULLISH
        elif current_price < indicators.sma_50 < indicators.sma_200:
            trend = TrendDirection.BEARISH
        
        # Generate signal based on indicators
        signal_score = 0
        
        # RSI signals
        if indicators.rsi < 30:
            signal_score += 2  # Oversold - buy signal
        elif indicators.rsi > 70:
            signal_score -= 2  # Overbought - sell signal
        elif indicators.rsi < 40:
            signal_score += 1
        elif indicators.rsi > 60:
            signal_score -= 1
        
        # MACD signals
        if indicators.macd > indicators.macd_signal:
            signal_score += 1
        else:
            signal_score -= 1
        
        if indicators.macd_histogram > 0:
            signal_score += 0.5
        else:
            signal_score -= 0.5
        
        # Moving average signals
        if current_price > indicators.sma_20:
            signal_score += 0.5
        else:
            signal_score -= 0.5
        
        if current_price > indicators.sma_50:
            signal_score += 0.5
        else:
            signal_score -= 0.5
        
        # Bollinger Band signals
        if current_price < indicators.bollinger_lower:
            signal_score += 1  # Potential bounce
        elif current_price > indicators.bollinger_upper:
            signal_score -= 1  # Potential pullback
        
        # Determine signal
        if signal_score >= 3:
            signal = SignalType.STRONG_BUY
        elif signal_score >= 1:
            signal = SignalType.BUY
        elif signal_score <= -3:
            signal = SignalType.STRONG_SELL
        elif signal_score <= -1:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD
        
        # Calculate confidence
        confidence = min(abs(signal_score) / 5, 0.95)
        
        # Calculate support and resistance
        df = self.get_stock_data(symbol)
        if not df.empty:
            recent_lows = df['Low'].tail(30).min()
            recent_highs = df['High'].tail(30).max()
            support = round(recent_lows * 0.98, 2)
            resistance = round(recent_highs * 1.02, 2)
        else:
            support = round(current_price * 0.95, 2)
            resistance = round(current_price * 1.05, 2)
        
        # Calculate target and stop loss
        if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            target_price = round(current_price * 1.1, 2)
            stop_loss = round(current_price * 0.95, 2)
        elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
            target_price = round(current_price * 0.9, 2)
            stop_loss = round(current_price * 1.05, 2)
        else:
            target_price = None
            stop_loss = None
        
        # Generate summary
        summary = self._generate_analysis_summary(symbol, trend, signal, indicators, current_price)
        
        return StockAnalysis(
            symbol=symbol.upper(),
            name=stock_info.get("name", symbol),
            current_price=current_price,
            day_high=stock_info.get("day_high"),
            day_low=stock_info.get("day_low"),
            year_high=stock_info.get("52_week_high"),
            year_low=stock_info.get("52_week_low"),
            trend=trend,
            signal=signal,
            confidence=round(confidence, 2),
            technical_indicators=indicators,
            support_level=support,
            resistance_level=resistance,
            target_price=target_price,
            stop_loss=stop_loss,
            analysis_summary=summary
        )
    
    def _generate_analysis_summary(
        self, symbol: str, trend: TrendDirection, 
        signal: SignalType, indicators: TechnicalIndicators,
        current_price: float
    ) -> str:
        """Generate human-readable analysis summary."""
        
        trend_desc = {
            TrendDirection.BULLISH: "positive uptrend",
            TrendDirection.BEARISH: "negative downtrend",
            TrendDirection.NEUTRAL: "sideways consolidation"
        }
        
        rsi_desc = ""
        if indicators.rsi < 30:
            rsi_desc = "The stock is oversold, indicating potential buying opportunity."
        elif indicators.rsi > 70:
            rsi_desc = "The stock is overbought, suggesting caution for new entries."
        else:
            rsi_desc = f"RSI at {indicators.rsi} indicates neutral momentum."
        
        ma_desc = ""
        if current_price > indicators.sma_50:
            ma_desc = "Price is trading above key moving averages, supporting bullish bias."
        else:
            ma_desc = "Price is below key moving averages, indicating bearish pressure."
        
        return f"""{symbol} is showing a {trend_desc[trend]} in the current market. {rsi_desc} {ma_desc} The overall signal is {signal.value} based on technical analysis."""
    
    def get_market_indices(self) -> List[MarketIndex]:
        """Get major market indices with fallback for non-market hours."""
        indices = []
        
        # Fallback data for non-market hours (last known values)
        fallback_indices = {
            "^NSEI": {"value": 22150.0, "prev": 22100.0},
            "^BSESN": {"value": 73200.0, "prev": 73100.0},
            "^NSEBANK": {"value": 47500.0, "prev": 47400.0},
            "^CNXIT": {"value": 35800.0, "prev": 35700.0},
        }
        
        for symbol, name in INDICES.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current = info.get("regularMarketPrice") or info.get("previousClose") or 0
                previous = info.get("previousClose") or info.get("regularMarketPreviousClose") or current
                
                # Use fallback if data is missing
                if not current or current == 0:
                    fallback = fallback_indices.get(symbol, {"value": 0, "prev": 0})
                    current = fallback["value"]
                    previous = fallback["prev"]
                    print(f"Using fallback data for {name} (market closed)")
                
                change = current - previous if previous else 0
                change_pct = (change / previous * 100) if previous else 0
                
                trend = TrendDirection.BULLISH if change > 0 else (
                    TrendDirection.BEARISH if change < 0 else TrendDirection.NEUTRAL
                )
                
                indices.append(MarketIndex(
                    symbol=symbol,
                    name=name,
                    value=round(current, 2),
                    change=round(change, 2),
                    change_percentage=round(change_pct, 2),
                    trend=trend
                ))
            except Exception as e:
                print(f"Error fetching index {symbol}: {e}")
                # Add fallback on error
                fallback = fallback_indices.get(symbol, {"value": 0, "prev": 0})
                indices.append(MarketIndex(
                    symbol=symbol,
                    name=name,
                    value=fallback["value"],
                    change=round(fallback["value"] - fallback["prev"], 2),
                    change_percentage=round((fallback["value"] - fallback["prev"]) / fallback["prev"] * 100, 2) if fallback["prev"] else 0,
                    trend=TrendDirection.NEUTRAL
                ))
        
        return indices
    
    def get_top_movers(self, count: int = 5, sector: str = "allSec") -> Dict[str, List[StockMover]]:
        """Get top gainers and losers directly from NSE/BSE APIs.
        
        Args:
            count: Number of stocks to return
            sector: Market sector/index to filter by:
                NSE Sectors:
                - NIFTY: NIFTY 50 stocks
                - BANKNIFTY: Bank Nifty stocks (uses index API)
                - NIFTYNEXT50: NIFTY Next 50 stocks
                - SecGtr20: Securities > Rs 20
                - SecLwr20: Securities < Rs 20
                - FOSec: F&O Securities
                - allSec: All Securities (default)
                - NIFTY_IT, NIFTY_PHARMA, etc: Sectoral indices
                
                BSE Sectors:
                - BSE_ALL: All BSE stocks
                - SENSEX: BSE SENSEX 30 stocks
        """
        
        # BSE sectors - use TradingView API with BSE filter
        BSE_SECTORS = ['BSE_ALL', 'SENSEX']
        if sector in BSE_SECTORS:
            try:
                result = self._fetch_from_tradingview_bse(count, sector)
                if result and (result.get("gainers") or result.get("losers")):
                    return result
            except Exception as e:
                print(f"BSE/TradingView API failed: {e}")
            return {"gainers": [], "losers": []}
        
        # NSE sectors - use NSE India API
        try:
            result = self._fetch_from_nse_api(count, sector)
            if result and (result.get("gainers") or result.get("losers")):
                return result
        except Exception as e:
            print(f"NSE API failed: {e}")
        
        # Fallback to TradingView if NSE fails (only for allSec)
        if sector == "allSec":
            try:
                result = self._fetch_from_tradingview(count)
                if result and (result.get("gainers") or result.get("losers")):
                    return result
            except Exception as e:
                print(f"TradingView API failed: {e}")
        
        # Return empty if all APIs fail
        return {"gainers": [], "losers": []}
    
    def _fetch_from_nse_api(self, count: int = 5, sector: str = "allSec") -> Dict[str, List[StockMover]]:
        """Fetch top movers directly from NSE India official API.
        
        Args:
            count: Number of stocks to return
            sector: Sector key to filter by:
                - NIFTY, BANKNIFTY, NIFTYNEXT50, SecGtr20, SecLwr20, FOSec, allSec (from gainers API)
                - NIFTY_BANK_INDEX: Bank Nifty constituents (from equity-stockIndices API)
                - SENSEX: BSE SENSEX (from BSE API)
                - NIFTY_IT, NIFTY_PHARMA, etc: Sectoral indices
        """
        import requests
        
        print(f"🔄 Attempting NSE India API for sector: {sector}...")
        
        gainers = []
        losers = []
        
        # Sectors that need specific index API calls (not available in gainers/losers API)
        INDEX_API_SECTORS = {
            'BANKNIFTY': 'NIFTY BANK',  # Bank Nifty returns empty in gainers API
            'NIFTY_IT': 'NIFTY IT',
            'NIFTY_PHARMA': 'NIFTY PHARMA',
            'NIFTY_AUTO': 'NIFTY AUTO',
            'NIFTY_FMCG': 'NIFTY FMCG',
            'NIFTY_METAL': 'NIFTY METAL',
            'NIFTY_REALTY': 'NIFTY REALTY',
            'NIFTY_ENERGY': 'NIFTY ENERGY',
            'NIFTY_INFRA': 'NIFTY INFRA',
            'NIFTY_PSE': 'NIFTY PSE',
            'NIFTY_PSU_BANK': 'NIFTY PSU BANK',
            'NIFTY_PRIVATE_BANK': 'NIFTY PVT BANK',
            'NIFTY_FIN_SERVICE': 'NIFTY FIN SERVICE',
            'NIFTY_MEDIA': 'NIFTY MEDIA',
        }
        
        # Valid sector keys for standard gainers/losers API
        GAINERS_API_SECTORS = ['NIFTY', 'NIFTYNEXT50', 'SecGtr20', 'SecLwr20', 'FOSec', 'allSec']
        
        # NSE requires session cookies - create a session and get cookies first
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/market-data/live-equity-market',
            'Connection': 'keep-alive',
        }
        session.headers.update(headers)
        
        try:
            # Step 1: Get session cookies by visiting main page
            print("  Step 1: Getting NSE cookies...")
            session.get('https://www.nseindia.com', timeout=10)
            print(f"  Got {len(session.cookies)} cookies")
            
            # Check if this sector needs the index constituents API
            if sector in INDEX_API_SECTORS:
                return self._fetch_from_index_api(session, INDEX_API_SECTORS[sector], count)
            
            # Fallback to allSec for unknown sectors
            if sector not in GAINERS_API_SECTORS:
                sector = 'allSec'
            
            # Step 2: Fetch gainers
            print(f"  Step 2: Fetching gainers for {sector}...")
            gainers_resp = session.get(
                'https://www.nseindia.com/api/live-analysis-variations?index=gainers',
                timeout=15
            )
            print(f"  Gainers response: {gainers_resp.status_code}")
            
            if gainers_resp.status_code == 200:
                data = gainers_resp.json()
                
                # Get stocks from the selected sector
                sector_data = data.get(sector, {}).get('data', [])
                
                # If no data in selected sector, fallback to allSec
                if not sector_data and sector != 'allSec':
                    print(f"  No data for {sector}, falling back to allSec")
                    sector_data = data.get('allSec', {}).get('data', [])
                
                # Sort by percentage change (descending for gainers)
                sector_data.sort(key=lambda x: float(x.get('perChange', 0) or x.get('net_price', 0)), reverse=True)
                
                for stock in sector_data[:count]:
                    symbol = stock.get('symbol', '')
                    gainers.append(StockMover(
                        symbol=symbol,
                        name=symbol,  # NSE API doesn't return full name
                        price=round(float(stock.get('ltp', 0)), 2),
                        change=round(float(stock.get('ltp', 0)) - float(stock.get('prev_price', 0)), 2),
                        change_percentage=round(float(stock.get('perChange', 0) or stock.get('net_price', 0)), 2),
                        volume=int(stock.get('trade_quantity', 0))
                    ))
                print(f"✅ NSE API: Fetched {len(gainers)} gainers from {sector}")
            
            # Step 3: Fetch losers
            print(f"  Step 3: Fetching losers for {sector}...")
            losers_resp = session.get(
                'https://www.nseindia.com/api/live-analysis-variations?index=loosers',
                timeout=15
            )
            
            if losers_resp.status_code == 200:
                data = losers_resp.json()
                
                # Get stocks from the selected sector
                sector_data = data.get(sector, {}).get('data', [])
                
                # If no data in selected sector, fallback to allSec
                if not sector_data and sector != 'allSec':
                    sector_data = data.get('allSec', {}).get('data', [])
                
                # Sort by percentage change (ascending for losers - most negative first)
                sector_data.sort(key=lambda x: float(x.get('perChange', 0) or x.get('net_price', 0)))
                
                for stock in sector_data[:count]:
                    symbol = stock.get('symbol', '')
                    losers.append(StockMover(
                        symbol=symbol,
                        name=symbol,
                        price=round(float(stock.get('ltp', 0)), 2),
                        change=round(float(stock.get('ltp', 0)) - float(stock.get('prev_price', 0)), 2),
                        change_percentage=round(float(stock.get('perChange', 0) or stock.get('net_price', 0)), 2),
                        volume=int(stock.get('trade_quantity', 0))
                    ))
                print(f"✅ NSE API: Fetched {len(losers)} losers from {sector}")
            
            if gainers or losers:
                return {"gainers": gainers, "losers": losers}
                
        except Exception as e:
            print(f"NSE API error: {e}")
        
        return {"gainers": [], "losers": []}
    
    def _fetch_from_index_api(self, session, index_name: str, count: int = 5) -> Dict[str, List[StockMover]]:
        """Fetch top movers from NSE index constituents API.
        
        This is used for indices like Bank Nifty that don't return data in the gainers/losers API.
        Uses the equity-stockIndices API to get all constituents and sorts by pChange.
        
        Args:
            session: Requests session with NSE cookies
            index_name: Index name like 'NIFTY BANK', 'NIFTY IT', etc.
            count: Number of stocks to return
        """
        import urllib.parse
        
        print(f"  Using index constituents API for: {index_name}")
        
        gainers = []
        losers = []
        
        try:
            # URL encode the index name (spaces become %20)
            encoded_index = urllib.parse.quote(index_name)
            url = f'https://www.nseindia.com/api/equity-stockIndices?index={encoded_index}'
            
            print(f"  Fetching: {url}")
            resp = session.get(url, timeout=15)
            print(f"  Response: {resp.status_code}")
            
            if resp.status_code == 200:
                data = resp.json()
                stocks_data = data.get('data', [])
                
                # Filter out the index itself (priority == 1)
                stocks_data = [s for s in stocks_data if s.get('priority', 0) != 1]
                
                if stocks_data:
                    # Sort by pChange for gainers (descending)
                    gainers_sorted = sorted(stocks_data, key=lambda x: float(x.get('pChange', 0)), reverse=True)
                    
                    for stock in gainers_sorted[:count]:
                        symbol = stock.get('symbol', '')
                        meta = stock.get('meta', {})
                        company_name = meta.get('companyName', symbol) if meta else symbol
                        
                        gainers.append(StockMover(
                            symbol=symbol,
                            name=company_name,
                            price=round(float(stock.get('lastPrice', 0)), 2),
                            change=round(float(stock.get('change', 0)), 2),
                            change_percentage=round(float(stock.get('pChange', 0)), 2),
                            volume=int(stock.get('totalTradedVolume', 0))
                        ))
                    
                    # Sort by pChange for losers (ascending)
                    losers_sorted = sorted(stocks_data, key=lambda x: float(x.get('pChange', 0)))
                    
                    for stock in losers_sorted[:count]:
                        symbol = stock.get('symbol', '')
                        meta = stock.get('meta', {})
                        company_name = meta.get('companyName', symbol) if meta else symbol
                        
                        losers.append(StockMover(
                            symbol=symbol,
                            name=company_name,
                            price=round(float(stock.get('lastPrice', 0)), 2),
                            change=round(float(stock.get('change', 0)), 2),
                            change_percentage=round(float(stock.get('pChange', 0)), 2),
                            volume=int(stock.get('totalTradedVolume', 0))
                        ))
                    
                    print(f"✅ Index API: Fetched {len(gainers)} gainers and {len(losers)} losers from {index_name}")
                    return {"gainers": gainers, "losers": losers}
                else:
                    print(f"  No stock data found for {index_name}")
                    
        except Exception as e:
            print(f"Index API error for {index_name}: {e}")
        
        return {"gainers": [], "losers": []}
    
    def _fetch_from_yahoo_screener(self, count: int = 5) -> Dict[str, List[StockMover]]:
        """Fetch ALL Indian stocks using Yahoo Finance Screener API - covers entire NSE/BSE."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        
        gainers = []
        losers = []
        
        try:
            # Yahoo Finance Screener API - Query ALL Indian stocks sorted by day change
            # Top Gainers - sorted by regularMarketChangePercent descending
            gainers_url = 'https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved'
            
            # Use yfinance's built-in screener for Indian market
            import yfinance as yf
            
            # Get top gainers from NSE using screener
            gainers_query = {
                "size": count * 2,
                "offset": 0,
                "sortField": "percentchange",
                "sortType": "DESC",
                "quoteType": "EQUITY",
                "query": {
                    "operator": "AND",
                    "operands": [
                        {"operator": "EQ", "operands": ["exchange", "NSI"]},
                        {"operator": "GT", "operands": ["percentchange", 0]}
                    ]
                },
                "userId": "",
                "userIdType": "guid"
            }
            
            # Alternative: Use Yahoo Finance India gainers endpoint directly
            gainers_resp = requests.get(
                'https://query1.finance.yahoo.com/v1/finance/screener?crumb=&lang=en-IN&region=IN&formatted=true&corsDomain=in.finance.yahoo.com&enableNavLinks=false&includeFields=regularMarketChangePercent,regularMarketChange,regularMarketPrice,symbol,shortName,regularMarketVolume&sortField=regularmarketchangepercent&sortType=desc&count=50&offset=0&screeners=day_gainers&entityIdType=all',
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                },
                timeout=15
            )
            
            if gainers_resp.status_code == 200:
                data = gainers_resp.json()
                quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
                
                for stock in quotes[:count]:
                    symbol = stock.get('symbol', '').replace('.NS', '').replace('.BO', '')
                    gainers.append(StockMover(
                        symbol=symbol,
                        name=stock.get('shortName', stock.get('longName', symbol)),
                        price=round(float(stock.get('regularMarketPrice', {}).get('raw', 0)), 2),
                        change=round(float(stock.get('regularMarketChange', {}).get('raw', 0)), 2),
                        change_percentage=round(float(stock.get('regularMarketChangePercent', {}).get('raw', 0)), 2),
                        volume=int(stock.get('regularMarketVolume', {}).get('raw', 0))
                    ))
                print(f"✅ Yahoo Screener: Fetched {len(gainers)} gainers from ALL Indian stocks")
            
            # Get top losers
            losers_resp = requests.get(
                'https://query1.finance.yahoo.com/v1/finance/screener?crumb=&lang=en-IN&region=IN&formatted=true&corsDomain=in.finance.yahoo.com&enableNavLinks=false&includeFields=regularMarketChangePercent,regularMarketChange,regularMarketPrice,symbol,shortName,regularMarketVolume&sortField=regularmarketchangepercent&sortType=asc&count=50&offset=0&screeners=day_losers&entityIdType=all',
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                },
                timeout=15
            )
            
            if losers_resp.status_code == 200:
                data = losers_resp.json()
                quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
                
                for stock in quotes[:count]:
                    symbol = stock.get('symbol', '').replace('.NS', '').replace('.BO', '')
                    losers.append(StockMover(
                        symbol=symbol,
                        name=stock.get('shortName', stock.get('longName', symbol)),
                        price=round(float(stock.get('regularMarketPrice', {}).get('raw', 0)), 2),
                        change=round(float(stock.get('regularMarketChange', {}).get('raw', 0)), 2),
                        change_percentage=round(float(stock.get('regularMarketChangePercent', {}).get('raw', 0)), 2),
                        volume=int(stock.get('regularMarketVolume', {}).get('raw', 0))
                    ))
                print(f"✅ Yahoo Screener: Fetched {len(losers)} losers from ALL Indian stocks")
            
            if gainers or losers:
                return {"gainers": gainers, "losers": losers}
                
        except Exception as e:
            print(f"Yahoo Screener error: {e}")
        
        return {"gainers": [], "losers": []}
    
    def _fetch_from_tradingview(self, count: int = 5) -> Dict[str, List[StockMover]]:
        """Fetch ALL stocks from TradingView Scanner API - covers entire market."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Content-Type': 'application/json',
        }
        
        gainers = []
        losers = []
        
        try:
            # TradingView Scanner API - Query ALL Indian stocks
            scanner_url = 'https://scanner.tradingview.com/india/scan'
            
            # Top Gainers query
            gainers_payload = {
                "filter": [
                    {"left": "change", "operation": "greater", "right": 0},
                    {"left": "is_primary", "operation": "equal", "right": True}
                ],
                "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                "columns": ["name", "close", "change", "change_abs", "volume", "description"],
                "sort": {"sortBy": "change", "sortOrder": "desc"},
                "options": {"lang": "en"},
                "range": [0, count * 2]
            }
            
            resp = requests.post(scanner_url, json=gainers_payload, headers=headers, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                for stock in data.get('data', [])[:count]:
                    d = stock.get('d', [])
                    if len(d) >= 5:
                        symbol = stock.get('s', '').replace('NSE:', '').replace('BSE:', '')
                        gainers.append(StockMover(
                            symbol=symbol,
                            name=d[5] if len(d) > 5 else symbol,  # description
                            price=round(float(d[1] or 0), 2),  # close
                            change=round(float(d[3] or 0), 2),  # change_abs
                            change_percentage=round(float(d[2] or 0), 2),  # change %
                            volume=int(d[4] or 0)  # volume
                        ))
                print(f"✅ TradingView: Fetched {len(gainers)} gainers from ALL Indian stocks")
            
            # Top Losers query
            losers_payload = {
                "filter": [
                    {"left": "change", "operation": "less", "right": 0},
                    {"left": "is_primary", "operation": "equal", "right": True}
                ],
                "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                "columns": ["name", "close", "change", "change_abs", "volume", "description"],
                "sort": {"sortBy": "change", "sortOrder": "asc"},
                "options": {"lang": "en"},
                "range": [0, count * 2]
            }
            
            resp = requests.post(scanner_url, json=losers_payload, headers=headers, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                for stock in data.get('data', [])[:count]:
                    d = stock.get('d', [])
                    if len(d) >= 5:
                        symbol = stock.get('s', '').replace('NSE:', '').replace('BSE:', '')
                        losers.append(StockMover(
                            symbol=symbol,
                            name=d[5] if len(d) > 5 else symbol,
                            price=round(float(d[1] or 0), 2),
                            change=round(float(d[3] or 0), 2),
                            change_percentage=round(float(d[2] or 0), 2),
                            volume=int(d[4] or 0)
                        ))
                print(f"✅ TradingView: Fetched {len(losers)} losers from ALL Indian stocks")
            
            if gainers or losers:
                return {"gainers": gainers, "losers": losers}
                
        except Exception as e:
            print(f"TradingView error: {e}")
        
        return {"gainers": [], "losers": []}
    
    def _fetch_from_tradingview_bse(self, count: int = 5, sector: str = "BSE_ALL") -> Dict[str, List[StockMover]]:
        """Fetch BSE stocks from TradingView Scanner API.
        
        Args:
            count: Number of stocks to return
            sector: BSE sector - BSE_ALL for all BSE stocks, SENSEX for SENSEX 30
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Content-Type': 'application/json',
        }
        
        gainers = []
        losers = []
        
        # SENSEX 30 constituent symbols (using NSE symbols with TradingView tickers filter)
        SENSEX_TICKERS = [
            "NSE:RELIANCE", "NSE:TCS", "NSE:HDFCBANK", "NSE:ICICIBANK", "NSE:INFY",
            "NSE:HINDUNILVR", "NSE:ITC", "NSE:SBIN", "NSE:BHARTIARTL", "NSE:KOTAKBANK",
            "NSE:LT", "NSE:AXISBANK", "NSE:MARUTI", "NSE:BAJFINANCE", "NSE:ASIANPAINT",
            "NSE:HCLTECH", "NSE:SUNPHARMA", "NSE:TITAN", "NSE:WIPRO", "NSE:ULTRACEMCO",
            "NSE:TECHM", "NSE:NTPC", "NSE:M_M", "NSE:POWERGRID", "NSE:NESTLEIND",
            "NSE:TATASTEEL", "NSE:INDUSINDBK", "NSE:JSWSTEEL", "NSE:BAJAJFINSV", "NSE:TATAMOTORS"
        ]
        
        try:
            scanner_url = 'https://scanner.tradingview.com/india/scan'
            
            if sector == "SENSEX":
                # For SENSEX, use specific tickers list with NSE exchange
                # Top Gainers query for SENSEX
                gainers_payload = {
                    "filter": [{"left": "change", "operation": "greater", "right": 0}],
                    "symbols": {"tickers": SENSEX_TICKERS},
                    "columns": ["name", "close", "change", "change_abs", "volume", "description"],
                    "sort": {"sortBy": "change", "sortOrder": "desc"},
                    "options": {"lang": "en"},
                    "range": [0, 30]
                }
                
                print(f"🔄 Fetching SENSEX gainers...")
                resp = requests.post(scanner_url, json=gainers_payload, headers=headers, timeout=15)
                
                if resp.status_code == 200:
                    data = resp.json()
                    for stock in data.get('data', [])[:count]:
                        symbol_full = stock.get('s', '')
                        d = stock.get('d', [])
                        if len(d) >= 5:
                            symbol = symbol_full.replace('NSE:', '')
                            gainers.append(StockMover(
                                symbol=symbol,
                                name=d[5] if len(d) > 5 else symbol,
                                price=round(float(d[1] or 0), 2),
                                change=round(float(d[3] or 0), 2),
                                change_percentage=round(float(d[2] or 0), 2),
                                volume=int(d[4] or 0)
                            ))
                    print(f"✅ TradingView SENSEX: Fetched {len(gainers)} gainers")
                
                # Top Losers query for SENSEX
                losers_payload = {
                    "filter": [{"left": "change", "operation": "less", "right": 0}],
                    "symbols": {"tickers": SENSEX_TICKERS},
                    "columns": ["name", "close", "change", "change_abs", "volume", "description"],
                    "sort": {"sortBy": "change", "sortOrder": "asc"},
                    "options": {"lang": "en"},
                    "range": [0, 30]
                }
                
                resp = requests.post(scanner_url, json=losers_payload, headers=headers, timeout=15)
                
                if resp.status_code == 200:
                    data = resp.json()
                    for stock in data.get('data', [])[:count]:
                        symbol_full = stock.get('s', '')
                        d = stock.get('d', [])
                        if len(d) >= 5:
                            symbol = symbol_full.replace('NSE:', '')
                            losers.append(StockMover(
                                symbol=symbol,
                                name=d[5] if len(d) > 5 else symbol,
                                price=round(float(d[1] or 0), 2),
                                change=round(float(d[3] or 0), 2),
                                change_percentage=round(float(d[2] or 0), 2),
                                volume=int(d[4] or 0)
                            ))
                    print(f"✅ TradingView SENSEX: Fetched {len(losers)} losers")
                    
            else:
                # BSE_ALL - Filter for BSE stocks by exchange
                base_filter = [
                    {"left": "is_primary", "operation": "equal", "right": True},
                    {"left": "exchange", "operation": "equal", "right": "BSE"}
                ]
                
                # Top Gainers query
                gainers_filter = base_filter.copy()
                gainers_filter.append({"left": "change", "operation": "greater", "right": 0})
                
                gainers_payload = {
                    "filter": gainers_filter,
                    "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                    "columns": ["name", "close", "change", "change_abs", "volume", "description"],
                    "sort": {"sortBy": "change", "sortOrder": "desc"},
                    "options": {"lang": "en"},
                    "range": [0, count * 3]
                }
                
                print(f"🔄 Fetching BSE gainers for {sector}...")
                resp = requests.post(scanner_url, json=gainers_payload, headers=headers, timeout=15)
                
                if resp.status_code == 200:
                    data = resp.json()
                    added = 0
                    for stock in data.get('data', []):
                        if added >= count:
                            break
                        symbol_full = stock.get('s', '')
                        if 'BSE:' in symbol_full:
                            d = stock.get('d', [])
                            if len(d) >= 5:
                                symbol = symbol_full.replace('BSE:', '')
                                gainers.append(StockMover(
                                    symbol=symbol,
                                    name=d[5] if len(d) > 5 else symbol,
                                    price=round(float(d[1] or 0), 2),
                                    change=round(float(d[3] or 0), 2),
                                    change_percentage=round(float(d[2] or 0), 2),
                                    volume=int(d[4] or 0)
                                ))
                                added += 1
                    print(f"✅ TradingView BSE: Fetched {len(gainers)} gainers from {sector}")
                
                # Top Losers query
                losers_filter = base_filter.copy()
                losers_filter.append({"left": "change", "operation": "less", "right": 0})
                
                losers_payload = {
                    "filter": losers_filter,
                    "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                    "columns": ["name", "close", "change", "change_abs", "volume", "description"],
                    "sort": {"sortBy": "change", "sortOrder": "asc"},
                    "options": {"lang": "en"},
                    "range": [0, count * 3]
                }
                
                resp = requests.post(scanner_url, json=losers_payload, headers=headers, timeout=15)
                
                if resp.status_code == 200:
                    data = resp.json()
                    added = 0
                    for stock in data.get('data', []):
                        if added >= count:
                            break
                        symbol_full = stock.get('s', '')
                        if 'BSE:' in symbol_full:
                            d = stock.get('d', [])
                            if len(d) >= 5:
                                symbol = symbol_full.replace('BSE:', '')
                                losers.append(StockMover(
                                    symbol=symbol,
                                    name=d[5] if len(d) > 5 else symbol,
                                    price=round(float(d[1] or 0), 2),
                                    change=round(float(d[3] or 0), 2),
                                    change_percentage=round(float(d[2] or 0), 2),
                                    volume=int(d[4] or 0)
                                ))
                                added += 1
                    print(f"✅ TradingView BSE: Fetched {len(losers)} losers from {sector}")
            
            if gainers or losers:
                return {"gainers": gainers, "losers": losers}
                
        except Exception as e:
            print(f"TradingView BSE error: {e}")
        
        return {"gainers": [], "losers": []}

    def _fetch_from_moneycontrol(self, count: int = 5) -> Dict[str, List[StockMover]]:
        """Fetch ALL market top movers from Moneycontrol - covers entire NSE/BSE."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html',
            'Referer': 'https://www.moneycontrol.com/markets/indian-indices/',
        }
        
        gainers = []
        losers = []
        
        try:
            # Moneycontrol Top Gainers API - ALL NSE stocks
            gainers_url = 'https://api.moneycontrol.com/mcapi/v1/stock/top-stocks?type=topGainers&exch=nse&limit=20'
            resp = requests.get(gainers_url, headers=headers, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                for stock in data.get('data', [])[:count]:
                    gainers.append(StockMover(
                        symbol=stock.get('symbol', stock.get('sc_id', '')),
                        name=stock.get('company_name', stock.get('name', '')),
                        price=round(float(stock.get('last_price', stock.get('lp', 0))), 2),
                        change=round(float(stock.get('change', stock.get('ch', 0))), 2),
                        change_percentage=round(float(stock.get('percent_change', stock.get('pch', 0))), 2),
                        volume=int(float(stock.get('volume', 0)))
                    ))
                print(f"✅ Moneycontrol: Fetched {len(gainers)} gainers from entire NSE market")
            
            # Moneycontrol Top Losers API
            losers_url = 'https://api.moneycontrol.com/mcapi/v1/stock/top-stocks?type=topLosers&exch=nse&limit=20'
            resp = requests.get(losers_url, headers=headers, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                for stock in data.get('data', [])[:count]:
                    losers.append(StockMover(
                        symbol=stock.get('symbol', stock.get('sc_id', '')),
                        name=stock.get('company_name', stock.get('name', '')),
                        price=round(float(stock.get('last_price', stock.get('lp', 0))), 2),
                        change=round(float(stock.get('change', stock.get('ch', 0))), 2),
                        change_percentage=round(float(stock.get('percent_change', stock.get('pch', 0))), 2),
                        volume=int(float(stock.get('volume', 0)))
                    ))
                print(f"✅ Moneycontrol: Fetched {len(losers)} losers from entire NSE market")
            
            if gainers or losers:
                return {"gainers": gainers, "losers": losers}
                
        except Exception as e:
            print(f"Moneycontrol API error: {e}")
        
        return {"gainers": [], "losers": []}
    
    def _fetch_from_screener(self, count: int = 5) -> Dict[str, List[StockMover]]:
        """Fetch from Screener.in API - covers ALL listed Indian stocks."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        
        gainers = []
        losers = []
        
        try:
            # Screener API for top gainers (all stocks)
            url = 'https://www.screener.in/api/screens/71/'  # Top gainers screen
            resp = requests.get(url, headers=headers, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                for stock in data.get('results', [])[:count]:
                    change_pct = float(stock.get('change_pct', 0))
                    gainers.append(StockMover(
                        symbol=stock.get('symbol', ''),
                        name=stock.get('name', ''),
                        price=round(float(stock.get('price', 0)), 2),
                        change=round(float(stock.get('change', 0)), 2),
                        change_percentage=round(change_pct, 2),
                        volume=int(float(stock.get('volume', 0)))
                    ))
                print(f"✅ Screener.in: Fetched {len(gainers)} gainers")
            
            # Top losers screen
            url = 'https://www.screener.in/api/screens/72/'
            resp = requests.get(url, headers=headers, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                for stock in data.get('results', [])[:count]:
                    change_pct = float(stock.get('change_pct', 0))
                    losers.append(StockMover(
                        symbol=stock.get('symbol', ''),
                        name=stock.get('name', ''),
                        price=round(float(stock.get('price', 0)), 2),
                        change=round(float(stock.get('change', 0)), 2),
                        change_percentage=round(change_pct, 2),
                        volume=int(float(stock.get('volume', 0)))
                    ))
                print(f"✅ Screener.in: Fetched {len(losers)} losers")
            
            if gainers or losers:
                return {"gainers": gainers, "losers": losers}
                
        except Exception as e:
            print(f"Screener API error: {e}")
        
        return {"gainers": [], "losers": []}
    
    def _fetch_from_nse_equity(self, count: int = 5) -> Dict[str, List[StockMover]]:
        """Fetch from NSE equity market API - ALL listed NSE stocks."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.nseindia.com/market-data/live-equity-market',
            'Connection': 'keep-alive',
        }
        
        gainers = []
        losers = []
        
        try:
            session = requests.Session()
            # Get cookies first
            session.get('https://www.nseindia.com', headers=headers, timeout=10)
            
            # Fetch ALL equity data and sort for gainers/losers
            equity_url = 'https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O'
            resp = session.get(equity_url, headers=headers, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                stocks = data.get('data', [])
                
                # Sort by percentage change
                stocks_sorted = sorted(stocks, key=lambda x: float(x.get('pChange', 0) or 0), reverse=True)
                
                # Top gainers
                for stock in stocks_sorted[:count]:
                    if float(stock.get('pChange', 0) or 0) > 0:
                        gainers.append(StockMover(
                            symbol=stock.get('symbol', ''),
                            name=stock.get('meta', {}).get('companyName', stock.get('symbol', '')),
                            price=round(float(stock.get('lastPrice', 0) or 0), 2),
                            change=round(float(stock.get('change', 0) or 0), 2),
                            change_percentage=round(float(stock.get('pChange', 0) or 0), 2),
                            volume=int(float(stock.get('totalTradedVolume', 0) or 0))
                        ))
                
                # Top losers
                for stock in stocks_sorted[-count:]:
                    if float(stock.get('pChange', 0) or 0) < 0:
                        losers.append(StockMover(
                            symbol=stock.get('symbol', ''),
                            name=stock.get('meta', {}).get('companyName', stock.get('symbol', '')),
                            price=round(float(stock.get('lastPrice', 0) or 0), 2),
                            change=round(float(stock.get('change', 0) or 0), 2),
                            change_percentage=round(float(stock.get('pChange', 0) or 0), 2),
                            volume=int(float(stock.get('totalTradedVolume', 0) or 0))
                        ))
                
                losers.reverse()
                print(f"✅ NSE Equity API: Fetched {len(gainers)} gainers, {len(losers)} losers")
            
            session.close()
            
            if gainers or losers:
                return {"gainers": gainers, "losers": losers}
                
        except Exception as e:
            print(f"NSE Equity API error: {e}")
        
        return {"gainers": [], "losers": []}
    
    def _fetch_from_bse_api(self, count: int = 5) -> Dict[str, List[StockMover]]:
        """Fetch from BSE API - ALL listed BSE stocks."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.bseindia.com/',
        }
        
        gainers = []
        losers = []
        
        try:
            # BSE Top Gainers
            gainers_url = 'https://api.bseindia.com/BseIndiaAPI/api/MktRGainerLoser/w?GLession=Session1&IndxGrp=&orderby=asc'
            resp = requests.get(gainers_url, headers=headers, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                for stock in data.get('Table', [])[:count]:
                    gainers.append(StockMover(
                        symbol=stock.get('scrip_cd', stock.get('scripcode', '')),
                        name=stock.get('scripname', stock.get('short_name', '')),
                        price=round(float(stock.get('ltradert', stock.get('LTP', 0))), 2),
                        change=round(float(stock.get('change', 0)), 2),
                        change_percentage=round(float(stock.get('pchange', 0)), 2),
                        volume=int(float(stock.get('volume', 0)))
                    ))
                print(f"✅ BSE API: Fetched {len(gainers)} gainers")
            
            # BSE Top Losers
            losers_url = 'https://api.bseindia.com/BseIndiaAPI/api/MktRGainerLoser/w?GLession=Session1&IndxGrp=&orderby=desc'
            resp = requests.get(losers_url, headers=headers, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                for stock in data.get('Table', [])[:count]:
                    losers.append(StockMover(
                        symbol=stock.get('scrip_cd', stock.get('scripcode', '')),
                        name=stock.get('scripname', stock.get('short_name', '')),
                        price=round(float(stock.get('ltradert', stock.get('LTP', 0))), 2),
                        change=round(float(stock.get('change', 0)), 2),
                        change_percentage=round(float(stock.get('pchange', 0)), 2),
                        volume=int(float(stock.get('volume', 0)))
                    ))
                print(f"✅ BSE API: Fetched {len(losers)} losers")
            
            if gainers or losers:
                return {"gainers": gainers, "losers": losers}
                
        except Exception as e:
            print(f"BSE API error: {e}")
        
        return {"gainers": [], "losers": []}

    def _fetch_from_groww(self, count: int = 5) -> Dict[str, List[StockMover]]:
        """Fetch top movers from Groww API (free, reliable)."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        
        gainers = []
        losers = []
        
        try:
            # Groww Top Gainers API
            gainers_resp = requests.get(
                'https://groww.in/v1/api/stocks_data/v1/top_gainers',
                headers=headers, timeout=10
            )
            
            if gainers_resp.status_code == 200:
                data = gainers_resp.json()
                for stock in data.get('topGainers', [])[:count]:
                    gainers.append(StockMover(
                        symbol=stock.get('nseScriptCode', stock.get('bseScriptCode', '')),
                        name=stock.get('companyName', ''),
                        price=round(float(stock.get('ltp', 0)), 2),
                        change=round(float(stock.get('dayChange', 0)), 2),
                        change_percentage=round(float(stock.get('dayChangePerc', 0)), 2),
                        volume=int(stock.get('volume', 0))
                    ))
                print(f"✅ Groww API: Fetched {len(gainers)} gainers")
            
            # Groww Top Losers API
            losers_resp = requests.get(
                'https://groww.in/v1/api/stocks_data/v1/top_losers',
                headers=headers, timeout=10
            )
            
            if losers_resp.status_code == 200:
                data = losers_resp.json()
                for stock in data.get('topLosers', [])[:count]:
                    losers.append(StockMover(
                        symbol=stock.get('nseScriptCode', stock.get('bseScriptCode', '')),
                        name=stock.get('companyName', ''),
                        price=round(float(stock.get('ltp', 0)), 2),
                        change=round(float(stock.get('dayChange', 0)), 2),
                        change_percentage=round(float(stock.get('dayChangePerc', 0)), 2),
                        volume=int(stock.get('volume', 0))
                    ))
                print(f"✅ Groww API: Fetched {len(losers)} losers")
            
            if gainers or losers:
                return {"gainers": gainers, "losers": losers}
                
        except Exception as e:
            print(f"Groww API error: {e}")
        
        return {"gainers": [], "losers": []}
    
    def _get_top_movers_fallback(self, count: int = 5) -> Dict[str, List[StockMover]]:
        """Fallback method using all 150+ Indian stocks for comprehensive market coverage."""
        movers_data = []
        
        def fetch_stock_data(symbol: str) -> Optional[Dict]:
            try:
                info = self.get_stock_info(symbol)
                current = info.get("current_price", 0)
                previous = info.get("previous_close", current)
                
                if current and previous and previous != 0:
                    change = current - previous
                    change_pct = (change / previous * 100)
                    
                    return {
                        "symbol": symbol,
                        "name": info.get("name", symbol),
                        "price": round(current, 2),
                        "change": round(change, 2),
                        "change_percentage": round(change_pct, 2),
                        "volume": info.get("volume", 0) or 0
                    }
            except Exception as e:
                pass  # Silent fail for individual stocks
            return None
        
        # Use 20 workers for faster parallel fetching of 150+ stocks
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(fetch_stock_data, symbol): symbol 
                      for symbol in ALL_INDIAN_STOCKS.keys()}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    movers_data.append(result)
        
        print(f"📊 Fetched data for {len(movers_data)} stocks from NSE/BSE")
        
        movers_data.sort(key=lambda x: x["change_percentage"], reverse=True)
        
        gainers = [StockMover(**m) for m in movers_data[:count] if m["change_percentage"] > 0]
        losers = [StockMover(**m) for m in movers_data[-count:] if m["change_percentage"] < 0]
        losers.reverse()
        
        return {"gainers": gainers, "losers": losers}
    
    def _fetch_sector_performance(self) -> Dict[str, float]:
        """Fetch real sector index performance from NSE sectoral indices."""
        sector_perf = {
            "IT": 0.0,
            "Banking": 0.0,
            "FMCG": 0.0,
            "Pharma": 0.0,
            "Energy": 0.0,
            "Metals": 0.0,
            "Automobile": 0.0,
            "Realty": 0.0,
        }
        
        # Map NSE index names to sector keys
        index_to_sector = {
            "NIFTY IT": "IT",
            "NIFTY BANK": "Banking",
            "NIFTY FMCG": "FMCG",
            "NIFTY PHARMA": "Pharma",
            "NIFTY ENERGY": "Energy",
            "NIFTY METAL": "Metals",
            "NIFTY AUTO": "Automobile",
            "NIFTY REALTY": "Realty",
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/',
        }
        
        try:
            # First get cookies from main page
            session = requests.Session()
            session.get('https://www.nseindia.com', headers=headers, timeout=10)
            
            for index_name, sector_key in index_to_sector.items():
                try:
                    url = f"https://www.nseindia.com/api/equity-stockIndices?index={index_name.replace(' ', '%20')}"
                    response = session.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # The first item in data is usually the index itself
                        index_data = data.get('data', [{}])[0] if data.get('data') else {}
                        change_pct = index_data.get('pChange', 0) or 0
                        sector_perf[sector_key] = round(float(change_pct), 2)
                        print(f"✅ {sector_key}: {change_pct:+.2f}%")
                except Exception as e:
                    print(f"⚠️ Failed to fetch {sector_key} index: {e}")
                    continue
            
            print(f"📈 Fetched sector performance for {sum(1 for v in sector_perf.values() if v != 0)} sectors")
            
        except Exception as e:
            print(f"❌ Error fetching sector performance: {e}")
            # Fall back to calculating from top movers
            try:
                movers = self.get_top_movers()
                sector_counts = {k: 0 for k in sector_perf}
                sector_totals = {k: 0.0 for k in sector_perf}
                
                for mover in movers.get("gainers", []) + movers.get("losers", []):
                    sector = STOCK_SECTORS.get(mover.symbol, "Other")
                    if sector in sector_perf:
                        sector_totals[sector] += mover.change_percentage
                        sector_counts[sector] += 1
                
                for sector in sector_perf:
                    if sector_counts[sector] > 0:
                        sector_perf[sector] = round(sector_totals[sector] / sector_counts[sector], 2)
            except Exception as fallback_error:
                print(f"❌ Fallback sector calculation failed: {fallback_error}")
        
        return sector_perf
    
    def get_market_overview(self) -> MarketOverview:
        """Get comprehensive market overview using NSE API data only."""
        indices = self.get_market_indices()
        movers = self.get_top_movers()
        
        # Fetch real sector index performance from NSE
        sector_perf = self._fetch_sector_performance()
        
        # Calculate market breadth from movers data
        advancing = len(movers.get("gainers", []))
        declining = len(movers.get("losers", []))
        
        # Determine overall sentiment
        nifty_change = next(
            (i.change_percentage for i in indices if "NIFTY" in i.name),
            0
        )
        
        if nifty_change > 1:
            sentiment = SentimentType.POSITIVE
        elif nifty_change < -1:
            sentiment = SentimentType.NEGATIVE
        else:
            sentiment = SentimentType.NEUTRAL
        
        return MarketOverview(
            indices=indices,
            top_gainers=movers["gainers"],
            top_losers=movers["losers"],
            market_breadth={"advancing": advancing, "declining": declining},
            sector_performance=sector_perf,
            market_sentiment=sentiment
        )
    
    def get_sector(self, symbol: str) -> str:
        """Get sector for a stock."""
        return STOCK_SECTORS.get(symbol.upper(), "Other")


# Global instance
market_data_service = MarketDataService()
