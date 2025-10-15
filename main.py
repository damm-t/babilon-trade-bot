import logging
import sys
from typing import Optional

from broker.alpaca_client import AlpacaClient, AlpacaClientError
from config import STOCK_SYMBOL, TRADE_THRESHOLD, LOG_LEVEL
from logic.trade_decision import generate_trade_signal, TradeSignal
from model.sentiment_model import analyze_sentiment

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BabilonTradeBot:
    """Enhanced Babilon Trade Bot with comprehensive error handling and risk management"""
    
    def __init__(self, symbol: str = STOCK_SYMBOL):
        self.symbol = symbol
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Alpaca client with error handling"""
        try:
            self.client = AlpacaClient()
            logger.info(f"Successfully initialized Babilon Trade Bot for {self.symbol}")
        except AlpacaClientError as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error initializing client: {e}")
            sys.exit(1)
    
    def get_current_price(self) -> Optional[float]:
        """Get current market price for the symbol"""
        try:
            bars = self.client.client.get_bars(self.symbol, "1Min", limit=1)
            if bars and self.symbol in bars:
                return float(bars[self.symbol][0].c)
            return None
        except Exception as e:
            logger.error(f"Failed to get current price for {self.symbol}: {e}")
            return None
    
    def get_price_history(self, limit: int = 20) -> list:
        """Get price history for technical analysis"""
        try:
            bars = self.client.client.get_bars(self.symbol, "1Day", limit=limit)
            if bars and self.symbol in bars:
                return [float(bar.c) for bar in bars[self.symbol]]
            return []
        except Exception as e:
            logger.error(f"Failed to get price history for {self.symbol}: {e}")
            return []
    
    def analyze_and_trade(self, news_text: str) -> dict:
        """
        Main trading function that analyzes sentiment and executes trades
        
        Args:
            news_text: News text to analyze for sentiment
            
        Returns:
            Dictionary with analysis results and trade outcome
        """
        result = {
            'success': False,
            'symbol': self.symbol,
            'news_text': news_text,
            'sentiment': None,
            'sentiment_score': None,
            'trade_signal': None,
            'trade_executed': False,
            'error': None
        }
        
        try:
            # Step 1: Analyze sentiment
            logger.info(f"Analyzing sentiment for: {news_text[:100]}...")
            sentiment, score = analyze_sentiment(news_text)
            result['sentiment'] = sentiment
            result['sentiment_score'] = score
            
            logger.info(f"[ANALYSIS] Sentiment: {sentiment} (Score: {score:.2f})")
            
            # Step 2: Get market data
            current_price = self.get_current_price()
            price_history = self.get_price_history()
            
            if current_price is None:
                logger.warning(f"Could not get current price for {self.symbol}, skipping trade")
                result['error'] = "Could not get current price"
                return result
            
            # Step 3: Generate trade signal
            reference_price = price_history[-2] if len(price_history) >= 2 else current_price
            trade_signal = generate_trade_signal(
                sentiment=sentiment,
                score=score,
                positive_threshold=TRADE_THRESHOLD,
                current_price=current_price,
                reference_price=reference_price,
                price_history=price_history
            )
            
            result['trade_signal'] = trade_signal.to_dict()
            logger.info(f"[SIGNAL] {trade_signal.action} (Confidence: {trade_signal.confidence:.2f})")
            logger.info(f"[REASONING] {trade_signal.reasoning}")
            
            # Step 4: Execute trade if signal is strong enough
            if trade_signal.action in ["BUY", "SELL"] and trade_signal.confidence > 0.6:
                try:
                    order_result = self.client.place_order(
                        symbol=self.symbol,
                        side=trade_signal.action,
                        current_price=current_price
                    )
                    
                    if order_result:
                        result['trade_executed'] = True
                        result['order_id'] = order_result.get('id')
                        logger.info(f"[TRADE] {trade_signal.action} order placed for {self.symbol}")
                        logger.info(f"[ORDER] ID: {order_result.get('id')}")
                    else:
                        logger.info(f"[TRADE] Order not placed (position constraints)")
                        
                except AlpacaClientError as e:
                    logger.error(f"[TRADE] Failed to place order: {e}")
                    result['error'] = f"Order failed: {e}"
                except Exception as e:
                    logger.error(f"[TRADE] Unexpected error placing order: {e}")
                    result['error'] = f"Unexpected order error: {e}"
            else:
                logger.info(f"[TRADE] No trade executed (signal: {trade_signal.action}, confidence: {trade_signal.confidence:.2f})")
            
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Unexpected error in analyze_and_trade: {e}")
            result['error'] = f"Analysis failed: {e}"
        
        return result
    
    def get_portfolio_status(self) -> dict:
        """Get current portfolio status"""
        try:
            account_info = self.client.get_account_info()
            position = self.client.get_position(self.symbol)
            
            return {
                'account': account_info,
                'position': position,
                'symbol': self.symbol
            }
        except Exception as e:
            logger.error(f"Failed to get portfolio status: {e}")
            return {'error': str(e)}


def run_bot(news_text: str, symbol: str = STOCK_SYMBOL) -> dict:
    """
    Legacy function for backward compatibility
    
    Args:
        news_text: News text to analyze
        symbol: Stock symbol to trade
        
    Returns:
        Analysis and trade result
    """
    bot = BabilonTradeBot(symbol)
    return bot.analyze_and_trade(news_text)


def main():
    """Main function with enhanced error handling"""
    try:
        # Example news text
        example_news = "Nvidia shares soar after smashing Wall Street estimates with record quarterly revenue."
        
        logger.info("Starting Babilon Trade Bot...")
        
        # Run the bot
        result = run_bot(example_news)
        
        # Print results
        print("\n" + "="*50)
        print("BABILON TRADE BOT - ANALYSIS RESULTS")
        print("="*50)
        print(f"Symbol: {result['symbol']}")
        print(f"News: {result['news_text'][:100]}...")
        print(f"Sentiment: {result['sentiment']} (Score: {result['sentiment_score']:.2f})")
        
        if result['trade_signal']:
            signal = result['trade_signal']
            print(f"Signal: {signal['action']} (Confidence: {signal['confidence']:.2f})")
            print(f"Reasoning: {signal['reasoning']}")
            if signal['stop_loss']:
                print(f"Stop Loss: ${signal['stop_loss']:.2f}")
            if signal['take_profit']:
                print(f"Take Profit: ${signal['take_profit']:.2f}")
        
        print(f"Trade Executed: {result['trade_executed']}")
        if result.get('order_id'):
            print(f"Order ID: {result['order_id']}")
        
        if result['error']:
            print(f"Error: {result['error']}")
        
        print("="*50)
        
        # Show portfolio status
        bot = BabilonTradeBot()
        portfolio = bot.get_portfolio_status()
        if 'error' not in portfolio:
            print("\nPORTFOLIO STATUS:")
            print(f"Cash: ${portfolio['account']['cash']:.2f}")
            print(f"Portfolio Value: ${portfolio['account']['portfolio_value']:.2f}")
            if portfolio['position']:
                pos = portfolio['position']
                print(f"Position in {pos['symbol']}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f}")
                print(f"Unrealized P&L: ${pos['unrealized_pl']:.2f} ({pos['unrealized_plpc']:.2%})")
            else:
                print(f"No position in {STOCK_SYMBOL}")
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()