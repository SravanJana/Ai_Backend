"""
AI Trading Copilot - News and Sentiment Analysis Service
Uses FinBERT for financial sentiment analysis.
"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import feedparser

# Try to import transformers (optional - for FinBERT sentiment)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Note: transformers not installed. Using rule-based sentiment analysis.")

from models import NewsSentiment, OverallSentiment, SentimentType


class SentimentAnalyzer:
    """FinBERT-based sentiment analyzer for financial text."""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            self._initialized = False
            return
        
        try:
            # Use FinBERT for financial sentiment analysis
            model_name = "ProsusAI/finbert"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                device=-1  # CPU
            )
            self._initialized = True
            print("FinBERT sentiment model loaded successfully")
        except Exception as e:
            print(f"Error loading FinBERT model: {e}")
            print("Falling back to rule-based sentiment analysis")
            self._initialized = False
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of given text."""
        if not text:
            return {"sentiment": SentimentType.NEUTRAL, "confidence": 0.5}
        
        self._initialize()
        
        if self._pipeline:
            try:
                # Truncate text if too long
                text = text[:512]
                result = self._pipeline(text)[0]
                
                label = result['label'].lower()
                score = result['score']
                
                if label == 'positive':
                    sentiment = SentimentType.POSITIVE
                elif label == 'negative':
                    sentiment = SentimentType.NEGATIVE
                else:
                    sentiment = SentimentType.NEUTRAL
                
                return {
                    "sentiment": sentiment,
                    "confidence": round(score, 3)
                }
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
        
        # Fallback to rule-based analysis
        return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis as fallback."""
        text_lower = text.lower()
        
        positive_words = [
            'surge', 'gain', 'rise', 'up', 'bullish', 'growth', 'profit',
            'rally', 'soar', 'jump', 'strong', 'outperform', 'beat', 'exceed',
            'positive', 'upgrade', 'buy', 'recommend', 'success', 'record',
            'high', 'boost', 'improve', 'advance', 'recover'
        ]
        
        negative_words = [
            'fall', 'drop', 'decline', 'down', 'bearish', 'loss', 'crash',
            'plunge', 'sink', 'weak', 'underperform', 'miss', 'negative',
            'downgrade', 'sell', 'concern', 'worry', 'risk', 'low', 'cut',
            'reduce', 'struggle', 'fail', 'slump'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        
        if total == 0:
            return {"sentiment": SentimentType.NEUTRAL, "confidence": 0.5}
        
        if pos_count > neg_count:
            confidence = pos_count / total
            return {"sentiment": SentimentType.POSITIVE, "confidence": round(confidence, 3)}
        elif neg_count > pos_count:
            confidence = neg_count / total
            return {"sentiment": SentimentType.NEGATIVE, "confidence": round(confidence, 3)}
        else:
            return {"sentiment": SentimentType.NEUTRAL, "confidence": 0.5}


class NewsService:
    """Service for fetching and analyzing financial news."""
    
    # RSS feeds for Indian financial news - prioritize Google News for freshness
    RSS_FEEDS = {
        "economictimes": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "livemint": "https://www.livemint.com/rss/markets",
    }
    
    # Google News RSS for fresh market news
    GOOGLE_NEWS_MARKET_RSS = "https://news.google.com/rss/search?q=indian+stock+market+NSE+BSE&hl=en-IN&gl=IN&ceid=IN:en"
    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}+stock+india&hl=en-IN&gl=IN&ceid=IN:en"
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=10)  # Shorter cache for fresher news
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_time:
            return False
        return datetime.now() - self._cache_time[key] < self._cache_duration
    
    def _parse_published_date(self, published_str: str) -> Optional[datetime]:
        """Parse published date string to datetime."""
        if not published_str:
            return None
        try:
            from dateutil import parser
            return parser.parse(published_str)
        except:
            return None
    
    def _is_recent_news(self, published_str: str, max_days: int = 3) -> bool:
        """Check if news is within the last N days."""
        pub_date = self._parse_published_date(published_str)
        if not pub_date:
            return True  # Include if we can't parse (might be recent)
        
        now = datetime.now(pub_date.tzinfo) if pub_date.tzinfo else datetime.now()
        age = now - pub_date
        return age.days <= max_days
    
    def fetch_news(self, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch news articles from RSS feeds."""
        cache_key = f"news_{query}_{limit}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        news_items = []
        
        if query:
            # Fetch stock-specific news from Google News
            feed_url = self.GOOGLE_NEWS_RSS.format(query=query)
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:limit * 2]:  # Fetch extra to filter
                    published = getattr(entry, 'published', '')
                    if not self._is_recent_news(published):
                        continue
                    source_info = getattr(entry, 'source', None)
                    source_title = source_info.get('title', 'Google News') if isinstance(source_info, dict) else 'Google News'
                    summary = getattr(entry, 'summary', '') or ''
                    news_items.append({
                        "headline": getattr(entry, 'title', ''),
                        "source": source_title,
                        "url": getattr(entry, 'link', ''),
                        "published": published,
                        "summary": summary[:300] if summary else ''
                    })
            except Exception as e:
                print(f"Error fetching news for {query}: {e}")
        else:
            # Fetch general market news - prioritize Google News for freshness
            try:
                feed = feedparser.parse(self.GOOGLE_NEWS_MARKET_RSS)
                for entry in feed.entries[:limit]:
                    published = getattr(entry, 'published', '')
                    if not self._is_recent_news(published):
                        continue
                    source_info = getattr(entry, 'source', None)
                    source_title = source_info.get('title', 'News') if isinstance(source_info, dict) else 'Market News'
                    summary = getattr(entry, 'summary', '') or ''
                    news_items.append({
                        "headline": getattr(entry, 'title', ''),
                        "source": source_title,
                        "url": getattr(entry, 'link', ''),
                        "published": published,
                        "summary": summary[:300] if summary else ''
                    })
            except Exception as e:
                print(f"Error fetching Google News: {e}")
            
            # Supplement with other RSS feeds if needed
            if len(news_items) < limit:
                for source_name, feed_url in self.RSS_FEEDS.items():
                    try:
                        feed = feedparser.parse(feed_url)
                        for entry in feed.entries[:3]:
                            published = getattr(entry, 'published', '')
                            if not self._is_recent_news(published):
                                continue
                            summary = getattr(entry, 'summary', '') or ''
                            news_items.append({
                                "headline": getattr(entry, 'title', ''),
                                "source": source_name.replace("_", " ").title(),
                                "url": getattr(entry, 'link', ''),
                                "published": published,
                                "summary": summary[:300] if summary else ''
                            })
                    except Exception as e:
                        print(f"Error fetching from {source_name}: {e}")
        
        # Limit results
        news_items = news_items[:limit]
        
        self._cache[cache_key] = news_items
        self._cache_time[cache_key] = datetime.now()
        
        return news_items
    
    def analyze_news_sentiment(self, symbol: str) -> OverallSentiment:
        """Analyze sentiment for a stock based on news."""
        news_items = self.fetch_news(query=symbol, limit=10)
        
        analyzed_news = []
        sentiment_scores = []
        
        for item in news_items:
            # Combine headline and summary for analysis
            text = f"{item['headline']} {item.get('summary', '')}"
            result = self.sentiment_analyzer.analyze(text)
            
            sentiment = result["sentiment"]
            confidence = result["confidence"]
            
            # Convert sentiment to numeric score
            if sentiment == SentimentType.POSITIVE:
                score = confidence
            elif sentiment == SentimentType.NEGATIVE:
                score = -confidence
            else:
                score = 0
            
            sentiment_scores.append(score)
            
            analyzed_news.append(NewsSentiment(
                headline=item["headline"],
                source=item["source"],
                sentiment=sentiment,
                confidence=confidence,
                url=item.get("url"),
                published_at=None  # Would need proper date parsing
            ))
        
        # Calculate overall sentiment
        if sentiment_scores:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
            
            if avg_score > 0.2:
                overall_sentiment = SentimentType.POSITIVE
            elif avg_score < -0.2:
                overall_sentiment = SentimentType.NEGATIVE
            else:
                overall_sentiment = SentimentType.NEUTRAL
            
            confidence = min(abs(avg_score) + 0.3, 1.0)
        else:
            overall_sentiment = SentimentType.NEUTRAL
            avg_score = 0
            confidence = 0.5
        
        positive_count = sum(1 for n in analyzed_news if n.sentiment == SentimentType.POSITIVE)
        negative_count = sum(1 for n in analyzed_news if n.sentiment == SentimentType.NEGATIVE)
        neutral_count = sum(1 for n in analyzed_news if n.sentiment == SentimentType.NEUTRAL)
        
        return OverallSentiment(
            symbol=symbol.upper(),
            overall_sentiment=overall_sentiment,
            sentiment_score=round(avg_score, 3),
            confidence=round(confidence, 3),
            news_count=len(analyzed_news),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            news_items=analyzed_news
        )
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment from news."""
        news_items = self.fetch_news(limit=20)
        
        if not news_items:
            return {
                "sentiment": SentimentType.NEUTRAL,
                "confidence": 0.5,
                "summary": "Unable to fetch market news"
            }
        
        scores = []
        for item in news_items:
            result = self.sentiment_analyzer.analyze(item["headline"])
            sentiment = result["sentiment"]
            confidence = result["confidence"]
            
            if sentiment == SentimentType.POSITIVE:
                scores.append(confidence)
            elif sentiment == SentimentType.NEGATIVE:
                scores.append(-confidence)
            else:
                scores.append(0)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score > 0.15:
            sentiment = SentimentType.POSITIVE
            summary = "Market sentiment is positive with bullish news flow."
        elif avg_score < -0.15:
            sentiment = SentimentType.NEGATIVE
            summary = "Market sentiment is negative with bearish news dominating."
        else:
            sentiment = SentimentType.NEUTRAL
            summary = "Market sentiment is mixed with no clear direction."
        
        return {
            "sentiment": sentiment,
            "sentiment_score": round(avg_score, 3),
            "confidence": round(min(abs(avg_score) + 0.3, 1.0), 3),
            "summary": summary,
            "news_analyzed": len(news_items)
        }


# Global instances
sentiment_analyzer = SentimentAnalyzer()
news_service = NewsService()
