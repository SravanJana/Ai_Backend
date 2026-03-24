"""
AI Trading Copilot - AI Chatbot Service
Uses LangChain and OpenAI for intelligent trading conversations.
"""
import json
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional
from datetime import datetime

# Try to import langchain - make it optional for development
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Note: langchain-groq not installed. Using fallback responses.")

from config import settings
from models import ChatMessage, ChatResponse, Portfolio, RiskMetrics
from services.portfolio_service import portfolio_service
from services.market_data import market_data_service, INDIAN_STOCKS
from services.news_service import news_service
from services.risk_engine import risk_engine

# Stock name to symbol mapping (for matching user queries)
STOCK_NAME_MAPPING = {
    "infosys": "INFY",
    "infy": "INFY",
    "reliance": "RELIANCE",
    "tcs": "TCS",
    "tata consultancy": "TCS",
    "hdfc bank": "HDFCBANK",
    "hdfc": "HDFCBANK",
    "icici bank": "ICICIBANK",
    "icici": "ICICIBANK",
    "hindustan unilever": "HINDUNILVR",
    "hul": "HINDUNILVR",
    "itc": "ITC",
    "bharti airtel": "BHARTIARTL",
    "airtel": "BHARTIARTL",
    "sbi": "SBIN",
    "state bank": "SBIN",
    "bajaj finance": "BAJFINANCE",
    "wipro": "WIPRO",
    "hcl tech": "HCLTECH",
    "hcltech": "HCLTECH",
    "maruti": "MARUTI",
    "maruti suzuki": "MARUTI",
    "asian paints": "ASIANPAINT",
    "axis bank": "AXISBANK",
    "axis": "AXISBANK",
    "kotak": "KOTAKBANK",
    "kotak bank": "KOTAKBANK",
    "larsen": "LT",
    "l&t": "LT",
    "sun pharma": "SUNPHARMA",
    "titan": "TITAN",
    "ultratech": "ULTRACEMCO",
}




import re

class TradingCopilot:
    """AI-powered trading assistant chatbot."""

    SYSTEM_PROMPT = """You are an expert AI trading assistant and financial advisor called "Trading Copilot". 
You help users understand their investment portfolios, analyze stocks, and make informed trading decisions.

Your capabilities:
1. Analyze user portfolios and provide insights
2. Give stock analysis with technical indicators
3. Explain market trends and sentiment
4. Provide buy/sell/hold recommendations with reasoning
5. Calculate and explain risk metrics
6. Suggest portfolio rebalancing strategies

Guidelines:
- Always be professional and provide data-driven insights
- Explain financial concepts in simple terms when needed
- Never guarantee returns or make definitive predictions
- Always mention that your analysis is for informational purposes only
- Be concise but thorough in your responses
- Use specific numbers and data when available
- Format responses with clear sections when providing detailed analysis

Remember to:
- Reference specific holdings when discussing the user's portfolio
- Use actual stock prices and metrics from the data provided
- Give actionable insights based on the analysis data
"""

    def _is_trading_question(self, user_message: str) -> bool:
        """Return True if the question is related to trading/finance/markets."""
        trading_keywords = [
            'stock', 'portfolio', 'market', 'nifty', 'sensex', 'share', 'equity', 'mutual fund',
            'ipo', 'dividend', 'option', 'futures', 'derivative', 'buy', 'sell', 'hold', 'price',
            'gainers', 'losers', 'volume', 'delivery', 'fiidii', 'fii', 'dii', 'sector', 'allocation',
            'risk', 'volatility', 'returns', 'investment', 'invest', 'rebalance', 'broker', 'exchange',
            'bse', 'nse', 'analysis', 'trend', 'technical', 'fundamental', 'news', 'sentiment', 'insider',
            'bulk deal', 'peer', 'comparison', 'calendar', 'economic', 'watchlist', 'alert', 'performance',
            'finance', 'financial', 'bank', 'bond', 'yield', 'index', 'indices', 'capital', 'asset', 'liquidity'
        ]
        msg = user_message.lower()
        for kw in trading_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', msg):
                return True
        return False

    def __init__(self):
        self.llm = None
        self.conversations: Dict[int, List] = {}  # Simple list-based memory
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the language model."""
        if not LANGCHAIN_AVAILABLE:
            print("LangChain not available. Using fallback responses.", flush=True)
            self.llm = None
            return
            
        if settings.groq_api_key:
            try:
                self.llm = ChatGroq(
                    model=settings.model_name,
                    groq_api_key=settings.groq_api_key,
                    temperature=settings.temperature
                )
                # Verify by checking LLM is not None
                if self.llm is not None:
                    print(f"✅ Groq LLM initialized successfully (model: {settings.model_name})", flush=True)
                else:
                    print("❌ Groq init returned None!", flush=True)
            except Exception as e:
                print(f"❌ Error initializing Groq: {e}", flush=True)
                import traceback
                traceback.print_exc()
                self.llm = None
        else:
            print("❌ Groq API key not configured. Chatbot will use fallback responses.", flush=True)
            self.llm = None
    
    def _get_memory(self, user_id: int) -> List:
        """Get or create conversation memory for a user."""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        return self.conversations[user_id]
    
    def _add_to_memory(self, user_id: int, role: str, content: str):
        """Add message to conversation memory."""
        memory = self._get_memory(user_id)
        memory.append({"role": role, "content": content})
        # Keep only last 20 messages
        if len(memory) > 20:
            self.conversations[user_id] = memory[-20:]
    
    def _build_context(self, user_id: int, user_message: str) -> str:
        """Build context about user's portfolio and relevant market data."""
        context_parts = []
        
        # Get portfolio data
        try:
            portfolio = portfolio_service.get_portfolio(user_id)
            context_parts.append(f"""
USER PORTFOLIO:
Total Value: ₹{portfolio.total_value:,.2f}
Total Investment: ₹{portfolio.total_investment:,.2f}
Total P&L: ₹{portfolio.total_pnl:,.2f} ({portfolio.total_pnl_percentage:+.2f}%)

Holdings:""")
            
            for h in portfolio.holdings:
                pnl_str = f"{h.pnl_percentage:+.2f}%" if h.pnl_percentage else "N/A"
                context_parts.append(
                    f"- {h.symbol}: {h.quantity} shares @ ₹{h.average_price:.2f} "
                    f"(Current: ₹{h.current_price:.2f if h.current_price else 'N/A'}, P&L: {pnl_str})"
                )
        except Exception as e:
            context_parts.append(f"Unable to fetch portfolio: {e}")
        
        # Get risk metrics
        try:
            holdings = portfolio_service.get_holdings(user_id)
            risk_metrics = risk_engine.analyze_portfolio(holdings, user_id)
            
            context_parts.append(f"""

RISK ANALYSIS:
Risk Score: {risk_metrics.risk_score:.2f}/1.00
Risk Level: {risk_metrics.risk_level.value}
Volatility: {risk_metrics.volatility*100:.1f}%
Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}
Portfolio Beta: {risk_metrics.beta:.2f}

Sector Allocation:""")
            
            for se in risk_metrics.sector_exposure:
                context_parts.append(f"- {se.sector}: {se.percentage:.1f}%")
        except Exception as e:
            context_parts.append(f"Unable to fetch risk metrics: {e}")
        
        # If asking about specific stock, get its analysis
        message_lower = user_message.lower()
        stock_symbols = list(INDIAN_STOCKS.keys())
        
        mentioned_stocks = [s for s in stock_symbols if s.lower() in message_lower]
        
        for symbol in mentioned_stocks[:2]:  # Limit to 2 stocks
            try:
                analysis = market_data_service.analyze_stock(symbol)
                if analysis:
                    context_parts.append(f"""

STOCK ANALYSIS - {symbol}:
Current Price: ₹{analysis.current_price:.2f}
Trend: {analysis.trend.value}
Signal: {analysis.signal.value}
Confidence: {analysis.confidence*100:.0f}%
RSI: {analysis.technical_indicators.rsi:.1f}
Support: ₹{analysis.support_level:.2f}
Resistance: ₹{analysis.resistance_level:.2f}
Summary: {analysis.analysis_summary}""")
                    
                # Get sentiment
                sentiment = news_service.analyze_news_sentiment(symbol)
                context_parts.append(f"""
News Sentiment: {sentiment.overall_sentiment.value} (Score: {sentiment.sentiment_score:.2f})
Recent Headlines:""")
                
                for news in sentiment.news_items[:3]:
                    context_parts.append(f"- [{news.sentiment.value}] {news.headline}")
                    
            except Exception as e:
                context_parts.append(f"Unable to fetch analysis for {symbol}: {e}")
        
        # Get market overview if asking general market questions
        market_keywords = ['market', 'nifty', 'sensex', 'overall', 'today', 'sector']
        if any(kw in message_lower for kw in market_keywords):
            try:
                market = market_data_service.get_market_overview()
                context_parts.append(f"""
                    MARKET OVERVIEW:
                    Market Sentiment: {market.market_sentiment.value}
                    Sector Performance:""")
                
                for sector, perf in market.sector_performance.items():
                    context_parts.append(f"- {sector}: {perf:+.2f}%")
                
                context_parts.append("\nTop Gainers:")
                for g in market.top_gainers[:3]:
                    context_parts.append(f"- {g.symbol}: {g.change_percentage:+.2f}%")
                
                context_parts.append("\nTop Losers:")
                for l in market.top_losers[:3]:
                    context_parts.append(f"- {l.symbol}: {l.change_percentage:+.2f}%")
                    
            except Exception as e:
                context_parts.append(f"Unable to fetch market overview: {e}")
        
        return "\n".join(context_parts)
    
    async def chat(self, message: ChatMessage) -> ChatResponse:
        """Process user message and generate response."""
        user_id = message.user_id
        user_message = message.message

        print(f"[CHAT] Received message: {user_message[:50]}...", flush=True)

        # Block non-trading/finance questions
        disclaimer = "\n\n---\n*Disclaimer: This response is for informational purposes only and does not constitute financial advice. Please consult a qualified financial advisor before making investment decisions.*"
        if not self._is_trading_question(user_message):
            return ChatResponse(
                response="❌ Sorry, your question is not related to trading or finance. Please ask about stocks, markets, portfolios, or investments." + disclaimer,
                suggestions=[
                    "Analyze my portfolio",
                    "How is the market today?",
                    "Should I buy HDFC Bank?",
                    "Show sector performance"
                ]
            )

        # Check if LLM is available
        if self.llm is None:
            print("[CHAT] LLM not available, using fallback", flush=True)
            return self._fallback_response(user_message, "", user_id)

        try:
            # Build context (synchronous - no need for asyncio)
            context = ""
            try:
                context = self._build_context(user_id, user_message)
                print(f"[CHAT] Context built: {len(context)} chars", flush=True)
            except Exception as e:
                print(f"[CHAT] Context error: {e}", flush=True)
                context = "Market data temporarily unavailable."

            # Build prompt and call LLM
            prompt = f"""You are an AI Trading Copilot assistant helping users with stock market analysis.

Current Market Context:
{context}

User Question: {user_message}

Provide a helpful, accurate response. Use emojis for visual clarity. Format with markdown for better readability."""

            # Call Groq LLM - use invoke() synchronously wrapped in thread
            print("[CHAT] Calling Groq LLM...", flush=True)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.llm.invoke(prompt))
            response_text = response.content if hasattr(response, 'content') else str(response)
            print(f"[CHAT] LLM response received: {len(response_text)} chars", flush=True)

            # Store in memory
            self._add_to_memory(user_id, "user", user_message)
            self._add_to_memory(user_id, "assistant", response_text)

            return ChatResponse(
                response=response_text + disclaimer,
                suggestions=self._get_suggested_prompts(user_message)
            )

        except Exception as e:
            print(f"[CHAT] LLM error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return self._fallback_response(user_message, "", user_id)
    
    def _fallback_response(
        self, 
        user_message: str, 
        context: str, 
        user_id: int
    ) -> ChatResponse:
        """Generate response without LLM using rule-based logic."""
        message_lower = user_message.lower()
        
        # Portfolio questions
        disclaimer = "\n\n---\n*Disclaimer: This response is for informational purposes only and does not constitute financial advice. Please consult a qualified financial advisor before making investment decisions.*"
        if any(w in message_lower for w in ['portfolio', 'holdings', 'position']):
            response = """📊 **Your Portfolio Summary**

**Sample Portfolio (Demo):**

💰 **Total Estimated Value:** ₹5,00,000

**Holdings:**
🟢 **INFY**: 50 shares - IT Sector
🟢 **HDFCBANK**: 30 shares - Banking Sector  
🔴 **RELIANCE**: 20 shares - Energy Sector
🟢 **TCS**: 40 shares - IT Sector
🟡 **ITC**: 100 shares - FMCG Sector

**Sector Allocation:**
• IT: 40%
• Banking: 25%
• Energy: 20%
• FMCG: 15%

💡 **Tip:** This is a demo portfolio. Connect your broker account for real data.""" + disclaimer
            return ChatResponse(
                response=response,
                suggestions=["Analyze my risk", "Best performing stock?", "Should I rebalance?"]
            )
        
        # Risk questions
        if any(w in message_lower for w in ['risk', 'volatile', 'safe', 'dangerous']):
            response = """📉 **Risk Analysis**

**Portfolio Risk Assessment (Demo):**

🎯 **Risk Score:** 0.45/1.00 (Moderate)
📊 **Risk Level:** Medium
📈 **Expected Volatility:** 15-20%

**Risk Factors:**
• Market Risk: Medium - Diversified across sectors
• Sector Risk: Low - Good sector mix
• Concentration Risk: Low - No single stock > 30%

**Sector Exposure:**
• IT: 40% (Medium volatility)
• Banking: 25% (Higher volatility)
• Energy: 20% (Higher volatility)
• FMCG: 15% (Lower volatility, defensive)

💡 **Recommendations:**
• Your portfolio has moderate risk exposure
• Consider adding more defensive stocks (FMCG/Pharma)
• Maintain emergency fund before investing more""" + disclaimer
            return ChatResponse(
                response=response,
                suggestions=["How to reduce risk?", "Analyze my portfolio", "Best defensive stocks"]
            )
        
        # Stock analysis - match by symbol OR company name
        mentioned_symbol = None
        
        # First check direct symbol match
        for symbol in INDIAN_STOCKS.keys():
            if symbol.lower() in message_lower:
                mentioned_symbol = symbol
                break
        
        # If no symbol found, check company name mapping
        if not mentioned_symbol:
            for name, symbol in STOCK_NAME_MAPPING.items():
                if name in message_lower:
                    mentioned_symbol = symbol
                    break
        
        if mentioned_symbol:
            symbol = mentioned_symbol
            # Get stock name from INDIAN_STOCKS or mapping
            stock_names = {
                "INFY": "Infosys", "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services",
                "HDFCBANK": "HDFC Bank", "ICICIBANK": "ICICI Bank", "HINDUNILVR": "Hindustan Unilever",
                "ITC": "ITC Limited", "BHARTIARTL": "Bharti Airtel", "SBIN": "State Bank of India",
                "BAJFINANCE": "Bajaj Finance", "WIPRO": "Wipro", "HCLTECH": "HCL Technologies",
                "MARUTI": "Maruti Suzuki", "ASIANPAINT": "Asian Paints", "AXISBANK": "Axis Bank",
                "KOTAKBANK": "Kotak Mahindra Bank", "LT": "Larsen & Toubro", "SUNPHARMA": "Sun Pharma",
                "TITAN": "Titan Company", "ULTRACEMCO": "UltraTech Cement"
            }
            stock_name = stock_names.get(symbol, symbol)
            # Provide helpful guidance without calling external APIs
            response = f"""📈 **{stock_name} ({symbol}) Analysis**

**Stock Information:**
• Sector: {"IT" if symbol in ["INFY", "TCS", "WIPRO", "HCLTECH"] else "Banking" if "BANK" in symbol or symbol == "SBIN" else "FMCG" if symbol in ["ITC", "HINDUNILVR"] else "Diversified"}
• Exchange: NSE

**Investment Considerations:**
🟢 **Positive Factors:**
• Large-cap company with strong market position
• Part of major indices (NIFTY 50)
• Established track record

🔴 **Risk Factors:**
• Market volatility affects all stocks
• Sector-specific risks apply
• Global economic conditions

💡 **Recommendation:**
For detailed analysis, consider:
1. Check current price on NSE website
2. Review recent quarterly results
3. Compare with sector peers
4. Analyze technical charts""" + disclaimer
            return ChatResponse(
                response=response,
                suggestions=[f"Tell me about {symbol} sector", "Compare IT stocks", "Portfolio diversification tips"]
            )
        
        # Market overview
        if any(w in message_lower for w in ['market', 'nifty', 'sensex', 'today']):
            response = """🏛️ **Market Overview**

**Indian Stock Market Summary:**

📈 **Key Indices:**
• NIFTY 50 - Major benchmark for NSE
• SENSEX - BSE's flagship index
• Bank Nifty - Banking sector performance

📊 **Market Hours:**
• Pre-open: 9:00 AM - 9:15 AM
• Trading: 9:15 AM - 3:30 PM (Monday to Friday)

💡 **Top Sectors:**
• IT - TCS, Infosys, Wipro, HCL Tech
• Banking - HDFC Bank, ICICI Bank, SBI, Axis Bank
• FMCG - HUL, ITC, Nestle
• Energy - Reliance, ONGC

📱 **Tips:**
• Check NSE/BSE websites for real-time prices
• Monitor FII/DII activity for market trends
• Follow RBI announcements for banking stocks""" + disclaimer
            return ChatResponse(
                response=response,
                suggestions=["Tell me about IT stocks", "Banking sector analysis", "Best stocks for beginners"]
            )
        
        # Default response
        return ChatResponse(
            response="""👋 Hello! I'm your AI Trading Copilot. I can help you with:

📊 **Portfolio Analysis** - "Analyze my portfolio", "Show my holdings"
📉 **Risk Assessment** - "What's my risk level?", "Is my portfolio safe?"
📈 **Stock Analysis** - "Should I buy INFY?", "Analyze RELIANCE"
🏛️ **Market Overview** - "How is the market today?", "Top gainers"
💡 **Recommendations** - "Best stocks to buy", "Should I rebalance?"

What would you like to know?""" + disclaimer,
            suggestions=[
                "Analyze my portfolio",
                "What's my risk level?",
                "How is the market today?",
                "Should I buy HDFC Bank?"
            ]
        )
    
    def _get_suggested_prompts(self, last_message: str) -> List[str]:
        """Generate contextual follow-up suggestions."""
        message_lower = last_message.lower()
        
        if 'portfolio' in message_lower:
            return [
                "What's my risk level?",
                "Suggest rebalancing",
                "Best performing stock?"
            ]
        elif 'risk' in message_lower:
            return [
                "How to reduce risk?",
                "Add defensive stocks",
                "Sector diversification"
            ]
        elif 'market' in message_lower:
            return [
                "Top sectors today",
                "Should I invest now?",
                "Market sentiment"
            ]
        else:
            return [
                "Analyze my portfolio",
                "Market overview",
                "Stock recommendations"
            ]
    
    def clear_conversation(self, user_id: int):
        """Clear conversation history for a user."""
        if user_id in self.conversations:
            del self.conversations[user_id]


# Global instance
trading_copilot = TradingCopilot()
# trigger reload
