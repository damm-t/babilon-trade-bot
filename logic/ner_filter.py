"""
NER-based Relevance Filter for News-to-Ticker Mapping
Implements entity recognition to map news articles to relevant tickers
"""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional
import pandas as pd
from collections import defaultdict, Counter

try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")

logger = logging.getLogger(__name__)


class TickerNERFilter:
    """NER-based filter for mapping news to relevant tickers"""
    
    def __init__(self):
        self.ticker_entities = self._load_ticker_entities()
        self.company_mappings = self._load_company_mappings()
        self.nlp = None
        self.ner_pipeline = None
        
        # Initialize NLP models
        self._initialize_nlp()
    
    def _load_ticker_entities(self) -> Dict[str, Set[str]]:
        """Load ticker symbols and their associated entities"""
        return {
            'AAPL': {'Apple', 'Apple Inc', 'iPhone', 'iPad', 'Mac', 'iOS', 'macOS', 'Tim Cook', 'Cupertino'},
            'NVDA': {'NVIDIA', 'Nvidia', 'GeForce', 'RTX', 'CUDA', 'Jensen Huang', 'GPU', 'AI chips', 'gaming'},
            'TSLA': {'Tesla', 'Elon Musk', 'Model S', 'Model 3', 'Model X', 'Model Y', 'Cybertruck', 'Autopilot', 'Gigafactory'},
            'MSFT': {'Microsoft', 'Windows', 'Office', 'Azure', 'Xbox', 'Surface', 'Satya Nadella', 'Teams', 'LinkedIn'},
            'GOOGL': {'Google', 'Alphabet', 'YouTube', 'Android', 'Chrome', 'Gmail', 'Sundar Pichai', 'Waymo', 'Pixel'},
            'AMZN': {'Amazon', 'AWS', 'Prime', 'Alexa', 'Jeff Bezos', 'Andy Jassy', 'Kindle', 'Echo', 'Whole Foods'},
            'META': {'Meta', 'Facebook', 'Instagram', 'WhatsApp', 'Mark Zuckerberg', 'Reality Labs', 'Oculus', 'VR'},
            'NFLX': {'Netflix', 'streaming', 'Reed Hastings', 'Ted Sarandos', 'Stranger Things', 'The Crown'},
            'AMD': {'AMD', 'Advanced Micro Devices', 'Lisa Su', 'Ryzen', 'Radeon', 'EPYC', 'semiconductors'},
            'INTC': {'Intel', 'Pat Gelsinger', 'Core', 'Xeon', 'processors', 'chips', 'semiconductors'},
            'XLK': {'Technology', 'tech sector', 'technology stocks', 'tech ETF', 'SPDR Technology'},
            'SPY': {'S&P 500', 'SPDR S&P 500', 'index fund', 'market index', 'broad market'},
            'QQQ': {'NASDAQ', 'QQQ', 'Invesco QQQ', 'tech-heavy', 'growth stocks'},
            'IWM': {'Russell 2000', 'small cap', 'small-cap stocks', 'small companies'},
            'GLD': {'Gold', 'gold ETF', 'precious metals', 'SPDR Gold', 'gold prices'},
            'TLT': {'Treasury', 'bonds', 'Treasury bonds', 'iShares 20+ Year Treasury', 'government bonds'},
            'VTI': {'Total Stock Market', 'Vanguard Total Stock Market', 'broad market', 'total market'},
            'VEA': {'Developed Markets', 'international stocks', 'developed countries', 'ex-US'},
            'VWO': {'Emerging Markets', 'emerging market stocks', 'developing countries'},
            'BND': {'Total Bond Market', 'Vanguard Total Bond Market', 'bond index', 'fixed income'}
        }
    
    def _load_company_mappings(self) -> Dict[str, str]:
        """Load company name to ticker mappings"""
        return {
            'Apple Inc': 'AAPL',
            'Apple Computer': 'AAPL',
            'NVIDIA Corporation': 'NVDA',
            'Tesla Inc': 'TSLA',
            'Microsoft Corporation': 'MSFT',
            'Alphabet Inc': 'GOOGL',
            'Google': 'GOOGL',
            'Amazon.com Inc': 'AMZN',
            'Meta Platforms Inc': 'META',
            'Facebook': 'META',
            'Netflix Inc': 'NFLX',
            'Advanced Micro Devices': 'AMD',
            'Intel Corporation': 'INTC',
            'Technology Select Sector SPDR Fund': 'XLK',
            'SPDR S&P 500 ETF Trust': 'SPY',
            'Invesco QQQ Trust': 'QQQ',
            'iShares Russell 2000 ETF': 'IWM',
            'SPDR Gold Trust': 'GLD',
            'iShares 20+ Year Treasury Bond ETF': 'TLT',
            'Vanguard Total Stock Market ETF': 'VTI',
            'Vanguard FTSE Developed Markets ETF': 'VEA',
            'Vanguard FTSE Emerging Markets ETF': 'VWO',
            'Vanguard Total Bond Market ETF': 'BND'
        }
    
    def _initialize_nlp(self):
        """Initialize NLP models"""
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.ner_pipeline = pipeline("ner", aggregation_strategy="simple")
                logger.info("Transformers NER pipeline loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load transformers NER pipeline: {e}")
                self.ner_pipeline = None
    
    def extract_entities_spacy(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities using spaCy"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                entities.append((ent.text, ent.label_))
            return entities
        except Exception as e:
            logger.warning(f"spaCy entity extraction failed: {e}")
            return []
    
    def extract_entities_transformers(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities using transformers"""
        if not self.ner_pipeline:
            return []
        
        try:
            results = self.ner_pipeline(text)
            entities = []
            for result in results:
                entities.append((result['word'], result['entity_group']))
            return entities
        except Exception as e:
            logger.warning(f"Transformers entity extraction failed: {e}")
            return []
    
    def extract_entities_regex(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities using regex patterns"""
        entities = []
        
        # Company names (capitalized words)
        company_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        companies = re.findall(company_pattern, text)
        for company in companies:
            if len(company) > 2:  # Filter out short words
                entities.append((company, 'ORG'))
        
        # Ticker symbols (3-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{3,5}\b'
        tickers = re.findall(ticker_pattern, text)
        for ticker in tickers:
            entities.append((ticker, 'TICKER'))
        
        # Product names (quoted or capitalized)
        product_pattern = r'"[^"]*"|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        products = re.findall(product_pattern, text)
        for product in products:
            if len(product) > 2:
                entities.append((product.strip('"'), 'PRODUCT'))
        
        return entities
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities using all available methods"""
        entities = []
        
        # Try spaCy first
        if self.nlp:
            entities.extend(self.extract_entities_spacy(text))
        
        # Try transformers
        if self.ner_pipeline:
            entities.extend(self.extract_entities_transformers(text))
        
        # Always use regex as fallback
        entities.extend(self.extract_entities_regex(text))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
    
    def calculate_ticker_relevance(self, text: str, ticker: str) -> float:
        """Calculate relevance score for a ticker given text"""
        if ticker not in self.ticker_entities:
            return 0.0
        
        text_lower = text.lower()
        relevance_score = 0.0
        
        # Direct ticker mention
        if ticker.lower() in text_lower:
            relevance_score += 0.3
        
        # Company name mentions
        for entity in self.ticker_entities[ticker]:
            if entity.lower() in text_lower:
                relevance_score += 0.2
        
        # Entity extraction and matching
        entities = self.extract_entities(text)
        for entity_text, entity_type in entities:
            entity_lower = entity_text.lower()
            
            # Check against ticker entities
            for ticker_entity in self.ticker_entities[ticker]:
                if ticker_entity.lower() in entity_lower or entity_lower in ticker_entity.lower():
                    relevance_score += 0.15
            
            # Check company mappings
            if entity_text in self.company_mappings and self.company_mappings[entity_text] == ticker:
                relevance_score += 0.25
        
        # Context-based scoring
        context_keywords = {
            'earnings': 0.1,
            'revenue': 0.1,
            'profit': 0.1,
            'stock': 0.1,
            'shares': 0.1,
            'market': 0.05,
            'trading': 0.05,
            'price': 0.05,
            'analyst': 0.05,
            'upgrade': 0.1,
            'downgrade': 0.1,
            'target': 0.05,
            'forecast': 0.05
        }
        
        for keyword, weight in context_keywords.items():
            if keyword in text_lower:
                relevance_score += weight
        
        # Normalize score to 0-1 range
        return min(1.0, relevance_score)
    
    def filter_relevant_tickers(self, text: str, min_relevance: float = 0.1) -> List[Tuple[str, float]]:
        """Filter tickers by relevance to text"""
        relevant_tickers = []
        
        for ticker in self.ticker_entities.keys():
            relevance = self.calculate_ticker_relevance(text, ticker)
            if relevance >= min_relevance:
                relevant_tickers.append((ticker, relevance))
        
        # Sort by relevance score (descending)
        relevant_tickers.sort(key=lambda x: x[1], reverse=True)
        
        return relevant_tickers
    
    def process_news_batch(self, news_data: List[Dict]) -> List[Dict]:
        """Process a batch of news articles and assign relevant tickers"""
        processed_news = []
        
        for article in news_data:
            try:
                text = article.get('text', '') or article.get('content', '') or article.get('description', '')
                if not text:
                    continue
                
                # Find relevant tickers
                relevant_tickers = self.filter_relevant_tickers(text)
                
                # Add ticker information to article
                article_copy = article.copy()
                article_copy['relevant_tickers'] = relevant_tickers
                article_copy['primary_ticker'] = relevant_tickers[0][0] if relevant_tickers else None
                article_copy['max_relevance'] = relevant_tickers[0][1] if relevant_tickers else 0.0
                
                processed_news.append(article_copy)
                
            except Exception as e:
                logger.error(f"Error processing article: {e}")
                continue
        
        return processed_news
    
    def create_ticker_news_mapping(self, news_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Create mapping from tickers to relevant news articles"""
        ticker_news_map = defaultdict(list)
        
        for article in news_data:
            relevant_tickers = article.get('relevant_tickers', [])
            for ticker, relevance in relevant_tickers:
                ticker_news_map[ticker].append({
                    'article': article,
                    'relevance': relevance
                })
        
        # Sort articles by relevance for each ticker
        for ticker in ticker_news_map:
            ticker_news_map[ticker].sort(key=lambda x: x['relevance'], reverse=True)
        
        return dict(ticker_news_map)
    
    def get_ticker_sentiment_summary(self, ticker: str, news_data: List[Dict]) -> Dict:
        """Get sentiment summary for a specific ticker"""
        ticker_articles = []
        
        for article in news_data:
            relevant_tickers = article.get('relevant_tickers', [])
            for rel_ticker, relevance in relevant_tickers:
                if rel_ticker == ticker:
                    ticker_articles.append({
                        'article': article,
                        'relevance': relevance
                    })
        
        if not ticker_articles:
            return {
                'ticker': ticker,
                'article_count': 0,
                'avg_relevance': 0.0,
                'sentiment_scores': [],
                'weighted_sentiment': 0.0
            }
        
        # Extract sentiment scores (assuming they exist in articles)
        sentiment_scores = []
        weighted_scores = []
        
        for item in ticker_articles:
            article = item['article']
            relevance = item['relevance']
            
            # Try to extract sentiment score from article
            sentiment_score = article.get('sentiment_score', 0.0)
            if isinstance(sentiment_score, (int, float)):
                sentiment_scores.append(sentiment_score)
                weighted_scores.append(sentiment_score * relevance)
        
        avg_relevance = sum(item['relevance'] for item in ticker_articles) / len(ticker_articles)
        weighted_sentiment = sum(weighted_scores) / sum(item['relevance'] for item in ticker_articles) if ticker_articles else 0.0
        
        return {
            'ticker': ticker,
            'article_count': len(ticker_articles),
            'avg_relevance': avg_relevance,
            'sentiment_scores': sentiment_scores,
            'weighted_sentiment': weighted_sentiment,
            'articles': ticker_articles
        }


def create_news_ticker_dataset(news_data: List[Dict], output_file: str = "data/news_ticker_mapping.parquet") -> None:
    """Create a dataset mapping news articles to relevant tickers"""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    ner_filter = TickerNERFilter()
    
    # Process news data
    processed_news = ner_filter.process_news_batch(news_data)
    
    # Create ticker-news mapping
    ticker_news_map = ner_filter.create_ticker_news_mapping(processed_news)
    
    # Convert to DataFrame
    rows = []
    for ticker, articles in ticker_news_map.items():
        for item in articles:
            article = item['article']
            rows.append({
                'ticker': ticker,
                'article_id': article.get('id', ''),
                'title': article.get('title', ''),
                'text': article.get('text', '')[:500],  # Truncate for storage
                'relevance': item['relevance'],
                'sentiment_score': article.get('sentiment_score', 0.0),
                'published_at': article.get('published_at', ''),
                'source': article.get('source', '')
            })
    
    df = pd.DataFrame(rows)
    df.to_parquet(output_file)
    
    logger.info(f"Saved {len(df)} news-ticker mappings to {output_file}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Sample news data
    sample_news = [
        {
            'id': '1',
            'title': 'Apple Reports Strong Q4 Earnings',
            'text': 'Apple Inc reported strong quarterly earnings with iPhone sales driving growth. The company beat analyst expectations.',
            'published_at': '2024-01-15T10:00:00Z',
            'source': 'Reuters'
        },
        {
            'id': '2',
            'title': 'NVIDIA AI Chips in High Demand',
            'text': 'NVIDIA Corporation sees increased demand for its AI chips as companies invest in artificial intelligence infrastructure.',
            'published_at': '2024-01-15T11:00:00Z',
            'source': 'Bloomberg'
        }
    ]
    
    # Create news-ticker mapping
    create_news_ticker_dataset(sample_news)
