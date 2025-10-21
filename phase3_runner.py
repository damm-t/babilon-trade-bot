#!/usr/bin/env python3
"""
Phase 3 Runner: Feature Expansion
Executes Phase 3 of Babilon Trade Bot development
- Builds comprehensive TA features
- Implements NER-based relevance filter
- Creates feature store (parquet) for model training
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.ta_features import TAFeatures, create_feature_store as create_ta_feature_store
from logic.ner_filter import TickerNERFilter, create_news_ticker_dataset
from logic.feature_store import FeatureStoreBuilder, create_feature_store_from_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase3_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import pyarrow
    except ImportError:
        missing_deps.append('pyarrow')
    
    try:
        import spacy
    except ImportError:
        logger.warning("spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")
    
    try:
        import transformers
    except ImportError:
        logger.warning("transformers not available. Install with: pip install transformers")
    
    if missing_deps:
        logger.error(f"Missing required dependencies: {missing_deps}")
        logger.error("Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True


def create_sample_news_data() -> List[Dict]:
    """Create sample news data for testing"""
    return [
        {
            'id': '1',
            'title': 'Apple Reports Strong Q4 Earnings Beat',
            'text': 'Apple Inc reported strong quarterly earnings with iPhone sales driving growth. The company beat analyst expectations with revenue of $94.8 billion.',
            'published_at': '2024-01-15T10:00:00Z',
            'source': 'Reuters',
            'sentiment_score': 0.8
        },
        {
            'id': '2',
            'title': 'NVIDIA AI Chips See Surging Demand',
            'text': 'NVIDIA Corporation sees increased demand for its AI chips as companies invest heavily in artificial intelligence infrastructure. Data center revenue grew 409%.',
            'published_at': '2024-01-15T11:00:00Z',
            'source': 'Bloomberg',
            'sentiment_score': 0.9
        },
        {
            'id': '3',
            'title': 'Tesla Stock Falls on Production Concerns',
            'text': 'Tesla Inc shares dropped after the company reported production delays at its Gigafactory. Elon Musk cited supply chain issues affecting Model Y production.',
            'published_at': '2024-01-15T12:00:00Z',
            'source': 'CNBC',
            'sentiment_score': 0.2
        },
        {
            'id': '4',
            'title': 'Microsoft Azure Growth Accelerates',
            'text': 'Microsoft Corporation reported strong growth in its Azure cloud platform. The company saw 35% revenue growth in its intelligent cloud segment.',
            'published_at': '2024-01-15T13:00:00Z',
            'source': 'TechCrunch',
            'sentiment_score': 0.7
        },
        {
            'id': '5',
            'title': 'Technology Sector Shows Mixed Signals',
            'text': 'The technology sector showed mixed signals with some stocks rising while others fell. The XLK ETF was relatively flat for the day.',
            'published_at': '2024-01-15T14:00:00Z',
            'source': 'MarketWatch',
            'sentiment_score': 0.5
        }
    ]


def run_ta_features_phase():
    """Execute TA features computation phase"""
    logger.info("=" * 50)
    logger.info("PHASE 3.1: Technical Analysis Features")
    logger.info("=" * 50)
    
    try:
        # Create TA feature store
        logger.info("Creating TA feature store...")
        create_ta_feature_store("data", "data/features/ta")
        
        # Verify TA features were created
        ta_dir = "data/features/ta"
        if os.path.exists(ta_dir):
            ta_files = [f for f in os.listdir(ta_dir) if f.endswith('.parquet')]
            logger.info(f"Created {len(ta_files)} TA feature files: {ta_files}")
        else:
            logger.error("TA features directory not created")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in TA features phase: {e}")
        return False


def run_ner_filter_phase():
    """Execute NER filter phase"""
    logger.info("=" * 50)
    logger.info("PHASE 3.2: NER-based Relevance Filter")
    logger.info("=" * 50)
    
    try:
        # Create sample news data
        logger.info("Creating sample news data...")
        news_data = create_sample_news_data()
        
        # Test NER filter
        logger.info("Testing NER filter...")
        ner_filter = TickerNERFilter()
        
        # Test with sample news
        for article in news_data:
            text = article['text']
            relevant_tickers = ner_filter.filter_relevant_tickers(text)
            logger.info(f"Article: {article['title'][:50]}...")
            logger.info(f"Relevant tickers: {relevant_tickers}")
        
        # Create news-ticker mapping
        logger.info("Creating news-ticker mapping...")
        create_news_ticker_dataset(news_data, "data/features/news_ticker_mapping.parquet")
        
        # Verify NER output
        if os.path.exists("data/features/news_ticker_mapping.parquet"):
            df = pd.read_parquet("data/features/news_ticker_mapping.parquet")
            logger.info(f"Created news-ticker mapping with {len(df)} entries")
            logger.info(f"Unique tickers: {df['ticker'].unique()}")
        else:
            logger.error("News-ticker mapping not created")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in NER filter phase: {e}")
        return False


def run_feature_store_phase():
    """Execute feature store creation phase"""
    logger.info("=" * 50)
    logger.info("PHASE 3.3: Combined Feature Store")
    logger.info("=" * 50)
    
    try:
        # Create sample news data
        news_data = create_sample_news_data()
        
        # Build comprehensive feature store
        logger.info("Building comprehensive feature store...")
        feature_store = create_feature_store_from_data(
            data_dir="data",
            features_dir="data/features",
            news_data=news_data
        )
        
        # Verify feature store
        if feature_store:
            logger.info(f"Feature store created with {len(feature_store)} tickers")
            
            for ticker, features in feature_store.items():
                logger.info(f"{ticker}: {len(features)} rows, {len(features.columns)} features")
                
                # Show sample features
                logger.info(f"Sample features for {ticker}:")
                sample_features = features.head(3)
                for col in sample_features.columns[:5]:  # Show first 5 columns
                    logger.info(f"  {col}: {sample_features[col].iloc[0] if not sample_features[col].isna().all() else 'N/A'}")
        else:
            logger.error("Feature store not created")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in feature store phase: {e}")
        return False


def run_validation_phase():
    """Execute validation phase"""
    logger.info("=" * 50)
    logger.info("PHASE 3.4: Validation & Summary")
    logger.info("=" * 50)
    
    try:
        # Check feature store structure
        features_dir = "data/features"
        
        if not os.path.exists(features_dir):
            logger.error("Features directory not found")
            return False
        
        # Check subdirectories
        subdirs = ['ta', 'sentiment', 'combined']
        for subdir in subdirs:
            subdir_path = os.path.join(features_dir, subdir)
            if os.path.exists(subdir_path):
                files = [f for f in os.listdir(subdir_path) if f.endswith('.parquet')]
                logger.info(f"{subdir} directory: {len(files)} files")
            else:
                logger.warning(f"{subdir} directory not found")
        
        # Check combined features
        combined_dir = os.path.join(features_dir, "combined")
        if os.path.exists(combined_dir):
            combined_files = [f for f in os.listdir(combined_dir) if f.endswith('.parquet')]
            logger.info(f"Combined features: {len(combined_files)} files")
            
            # Analyze feature quality
            for file in combined_files[:3]:  # Check first 3 files
                file_path = os.path.join(combined_dir, file)
                df = pd.read_parquet(file_path)
                
                logger.info(f"File: {file}")
                logger.info(f"  Rows: {len(df)}")
                logger.info(f"  Columns: {len(df.columns)}")
                logger.info(f"  Missing data: {df.isnull().sum().sum()}")
                logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        # Create summary report
        summary_file = "data/features/phase3_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Phase 3 Execution Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Execution time: {datetime.now()}\n")
            f.write(f"Features directory: {features_dir}\n")
            
            # Count files in each subdirectory
            for subdir in subdirs:
                subdir_path = os.path.join(features_dir, subdir)
                if os.path.exists(subdir_path):
                    files = [f for f in os.listdir(subdir_path) if f.endswith('.parquet')]
                    f.write(f"{subdir}: {len(files)} files\n")
        
        logger.info(f"Summary report saved to {summary_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error in validation phase: {e}")
        return False


def main():
    """Main execution function for Phase 3"""
    logger.info("Starting Phase 3: Feature Expansion")
    logger.info(f"Execution started at: {datetime.now()}")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Please install required packages.")
        return False
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Execute phases
    phases = [
        ("TA Features", run_ta_features_phase),
        ("NER Filter", run_ner_filter_phase),
        ("Feature Store", run_feature_store_phase),
        ("Validation", run_validation_phase)
    ]
    
    results = {}
    
    for phase_name, phase_func in phases:
        logger.info(f"\nExecuting {phase_name}...")
        try:
            result = phase_func()
            results[phase_name] = result
            if result:
                logger.info(f"✅ {phase_name} completed successfully")
            else:
                logger.error(f"❌ {phase_name} failed")
        except Exception as e:
            logger.error(f"❌ {phase_name} failed with exception: {e}")
            results[phase_name] = False
    
    # Final summary
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3 EXECUTION SUMMARY")
    logger.info("=" * 50)
    
    success_count = sum(results.values())
    total_phases = len(results)
    
    for phase_name, result in results.items():
        status = "✅ SUCCESS" if result else "❌ FAILED"
        logger.info(f"{phase_name}: {status}")
    
    logger.info(f"\nOverall: {success_count}/{total_phases} phases completed successfully")
    
    if success_count == total_phases:
        logger.info("🎉 Phase 3 completed successfully!")
        logger.info("Feature store is ready for model training.")
        return True
    else:
        logger.error("⚠️ Phase 3 completed with errors.")
        logger.error("Please check the logs and fix any issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
