"""
Feature Store Builder for Phase 3
Combines TA features with sentiment features to create comprehensive feature store
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import glob

from .ta_features import TAFeatures
from .ner_filter import TickerNERFilter
from model.sentiment_model import analyze_sentiment

logger = logging.getLogger(__name__)


class FeatureStoreBuilder:
    """Build comprehensive feature store combining TA and sentiment features"""
    
    def __init__(self, data_dir: str = "data", features_dir: str = "data/features"):
        self.data_dir = data_dir
        self.features_dir = features_dir
        self.ta_calculator = TAFeatures()
        self.ner_filter = TickerNERFilter()
        
        # Create directories if they don't exist
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(os.path.join(features_dir, "ta"), exist_ok=True)
        os.makedirs(os.path.join(features_dir, "sentiment"), exist_ok=True)
        os.makedirs(os.path.join(features_dir, "combined"), exist_ok=True)
    
    def load_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load ticker data from CSV file"""
        csv_file = os.path.join(self.data_dir, f"{ticker}.csv")
        
        if not os.path.exists(csv_file):
            logger.warning(f"Data file not found for ticker {ticker}: {csv_file}")
            return None
        
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(df)} rows for ticker {ticker}")
            return df
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None
    
    def compute_ta_features(self, ticker: str) -> Optional[pd.DataFrame]:
        """Compute TA features for a ticker"""
        df = self.load_ticker_data(ticker)
        if df is None:
            return None
        
        try:
            ta_features = self.ta_calculator.build_comprehensive_features(df)
            ta_features['ticker'] = ticker
            
            # Save TA features
            ta_file = os.path.join(self.features_dir, "ta", f"{ticker}_ta_features.parquet")
            ta_features.to_parquet(ta_file)
            logger.info(f"Saved TA features for {ticker} to {ta_file}")
            
            return ta_features
        except Exception as e:
            logger.error(f"Error computing TA features for {ticker}: {e}")
            return None
    
    def compute_sentiment_features(self, ticker: str, news_data: List[Dict]) -> Optional[pd.DataFrame]:
        """Compute sentiment features for a ticker"""
        try:
            # Filter news relevant to this ticker
            relevant_news = []
            for article in news_data:
                relevant_tickers = self.ner_filter.filter_relevant_tickers(
                    article.get('text', '') or article.get('content', '') or article.get('description', '')
                )
                for rel_ticker, relevance in relevant_tickers:
                    if rel_ticker == ticker and relevance > 0.1:
                        relevant_news.append({
                            'article': article,
                            'relevance': relevance
                        })
            
            if not relevant_news:
                logger.warning(f"No relevant news found for ticker {ticker}")
                return None
            
            # Analyze sentiment for each article
            sentiment_data = []
            for item in relevant_news:
                article = item['article']
                text = article.get('text', '') or article.get('content', '') or article.get('description', '')
                
                if text:
                    sentiment, score = analyze_sentiment(text)
                    sentiment_data.append({
                        'timestamp': pd.to_datetime(article.get('published_at', datetime.now())),
                        'sentiment': sentiment,
                        'score': score,
                        'relevance': item['relevance'],
                        'title': article.get('title', ''),
                        'source': article.get('source', '')
                    })
            
            if not sentiment_data:
                return None
            
            # Create sentiment DataFrame
            sentiment_df = pd.DataFrame(sentiment_data)
            sentiment_df.set_index('timestamp', inplace=True)
            sentiment_df.sort_index(inplace=True)
            
            # Aggregate sentiment features by time periods
            sentiment_features = self._aggregate_sentiment_features(sentiment_df)
            sentiment_features['ticker'] = ticker
            
            # Save sentiment features
            sentiment_file = os.path.join(self.features_dir, "sentiment", f"{ticker}_sentiment_features.parquet")
            sentiment_features.to_parquet(sentiment_file)
            logger.info(f"Saved sentiment features for {ticker} to {sentiment_file}")
            
            return sentiment_features
            
        except Exception as e:
            logger.error(f"Error computing sentiment features for {ticker}: {e}")
            return None
    
    def _aggregate_sentiment_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment features by time periods"""
        # Resample to hourly intervals
        hourly_sentiment = sentiment_df.resample('h').agg({
            'score': ['mean', 'std', 'count'],
            'relevance': ['mean', 'max'],
            'sentiment': lambda x: x.mode().iloc[0] if not x.empty else 'neutral'
        })
        
        # Flatten column names
        hourly_sentiment.columns = ['_'.join(col).strip() for col in hourly_sentiment.columns]
        hourly_sentiment.columns = [col.replace('<lambda>', 'mode') for col in hourly_sentiment.columns]
        
        # Calculate additional sentiment features
        hourly_sentiment['sentiment_positive_ratio'] = (sentiment_df['sentiment'] == 'positive').resample('h').mean()
        hourly_sentiment['sentiment_negative_ratio'] = (sentiment_df['sentiment'] == 'negative').resample('h').mean()
        hourly_sentiment['sentiment_neutral_ratio'] = (sentiment_df['sentiment'] == 'neutral').resample('h').mean()
        
        # Weighted sentiment score
        hourly_sentiment['weighted_sentiment'] = (
            sentiment_df['score'] * sentiment_df['relevance']
        ).resample('h').sum() / sentiment_df['relevance'].resample('h').sum()
        
        # Sentiment momentum (change in sentiment over time)
        hourly_sentiment['sentiment_momentum'] = hourly_sentiment['score_mean'].diff()
        
        # Sentiment volatility
        hourly_sentiment['sentiment_volatility'] = hourly_sentiment['score_std'].rolling(24).mean()
        
        # News volume features
        hourly_sentiment['news_volume'] = hourly_sentiment['count']
        hourly_sentiment['news_volume_ma'] = hourly_sentiment['news_volume'].rolling(24).mean()
        hourly_sentiment['news_volume_ratio'] = hourly_sentiment['news_volume'] / hourly_sentiment['news_volume_ma']
        
        # Fill NaN values
        hourly_sentiment = hourly_sentiment.ffill().fillna(0)
        
        return hourly_sentiment
    
    def combine_features(self, ticker: str) -> Optional[pd.DataFrame]:
        """Combine TA and sentiment features for a ticker"""
        try:
            # Load TA features
            ta_file = os.path.join(self.features_dir, "ta", f"{ticker}_ta_features.parquet")
            if not os.path.exists(ta_file):
                logger.warning(f"TA features not found for {ticker}")
                return None
            
            ta_features = pd.read_parquet(ta_file)
            
            # Load sentiment features
            sentiment_file = os.path.join(self.features_dir, "sentiment", f"{ticker}_sentiment_features.parquet")
            if not os.path.exists(sentiment_file):
                logger.warning(f"Sentiment features not found for {ticker}")
                # Use only TA features
                combined_features = ta_features.copy()
            else:
                sentiment_features = pd.read_parquet(sentiment_file)
                
                # Merge features on timestamp
                combined_features = pd.merge(
                    ta_features, 
                    sentiment_features, 
                    left_index=True, 
                    right_index=True, 
                    how='outer'
                )
            
            # Add derived features
            combined_features = self._add_derived_features(combined_features)
            
            # Clean up data
            combined_features = self._clean_features(combined_features)
            
            # Save combined features
            combined_file = os.path.join(self.features_dir, "combined", f"{ticker}_combined_features.parquet")
            combined_features.to_parquet(combined_file)
            logger.info(f"Saved combined features for {ticker} to {combined_file}")
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error combining features for {ticker}: {e}")
            return None
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features that combine TA and sentiment"""
        # Sentiment-TA interaction features
        if 'score_mean' in df.columns and 'rsi_14' in df.columns:
            df['sentiment_rsi_interaction'] = df['score_mean'] * df['rsi_14']
        
        if 'score_mean' in df.columns and 'macd' in df.columns:
            df['sentiment_macd_interaction'] = df['score_mean'] * df['macd']
        
        if 'weighted_sentiment' in df.columns and 'price_change' in df.columns:
            df['sentiment_price_momentum'] = df['weighted_sentiment'] * df['price_change']
        
        # News impact features
        if 'news_volume' in df.columns and 'price_change' in df.columns:
            df['news_volume_price_impact'] = df['news_volume'] * abs(df['price_change'])
        
        # Sentiment regime features
        if 'score_mean' in df.columns:
            df['sentiment_regime'] = pd.cut(
                df['score_mean'], 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['negative', 'neutral', 'positive']
            )
        
        # Volatility-sentiment interaction
        if 'historical_volatility' in df.columns and 'sentiment_volatility' in df.columns:
            df['volatility_sentiment_ratio'] = df['historical_volatility'] / (df['sentiment_volatility'] + 1e-8)
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with forward fill, then backward fill
        df = df.ffill().bfill()
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        return df
    
    def build_feature_store(self, news_data: List[Dict] = None) -> Dict[str, pd.DataFrame]:
        """Build complete feature store for all tickers"""
        # Get list of available tickers
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        tickers = [os.path.basename(f).replace('.csv', '') for f in csv_files]
        tickers = [t for t in tickers if t not in ['logs', 'portfolio']]
        
        logger.info(f"Building feature store for tickers: {tickers}")
        
        feature_store = {}
        
        for ticker in tickers:
            logger.info(f"Processing ticker: {ticker}")
            
            # Compute TA features
            ta_features = self.compute_ta_features(ticker)
            if ta_features is None:
                continue
            
            # Compute sentiment features if news data is available
            if news_data:
                sentiment_features = self.compute_sentiment_features(ticker, news_data)
            else:
                sentiment_features = None
            
            # Combine features
            combined_features = self.combine_features(ticker)
            if combined_features is not None:
                feature_store[ticker] = combined_features
        
        # Create summary statistics
        self._create_feature_summary(feature_store)
        
        logger.info(f"Feature store built successfully with {len(feature_store)} tickers")
        return feature_store
    
    def _create_feature_summary(self, feature_store: Dict[str, pd.DataFrame]) -> None:
        """Create summary statistics for the feature store"""
        summary_data = []
        
        for ticker, features in feature_store.items():
            summary_data.append({
                'ticker': ticker,
                'feature_count': len(features.columns),
                'row_count': len(features),
                'date_range_start': features.index.min(),
                'date_range_end': features.index.max(),
                'missing_data_ratio': features.isnull().sum().sum() / (len(features) * len(features.columns)),
                'ta_features': len([col for col in features.columns if col not in ['ticker', 'score_mean', 'sentiment', 'relevance']]),
                'sentiment_features': len([col for col in features.columns if 'sentiment' in col or 'score' in col or 'news' in col])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.features_dir, "feature_summary.parquet")
        summary_df.to_parquet(summary_file)
        
        logger.info(f"Feature summary saved to {summary_file}")
        logger.info(f"Summary:\n{summary_df.to_string()}")


def create_feature_store_from_data(data_dir: str = "data", features_dir: str = "data/features", 
                                  news_data: List[Dict] = None) -> Dict[str, pd.DataFrame]:
    """Main function to create feature store from existing data"""
    builder = FeatureStoreBuilder(data_dir, features_dir)
    return builder.build_feature_store(news_data)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create feature store
    feature_store = create_feature_store_from_data()
    
    print(f"Feature store created with {len(feature_store)} tickers")
    for ticker, features in feature_store.items():
        print(f"{ticker}: {len(features)} rows, {len(features.columns)} features")
