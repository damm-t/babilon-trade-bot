import logging
import os
import time
from typing import Optional, Dict, Any
from decimal import Decimal

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError, RestAPIError
except ImportError:
    # alpaca-trade-api not installed, create dummy classes
    class APIError(Exception):
        def __init__(self, *args, **kwargs):
            super().__init__(*args)
            self.status_code = kwargs.get('status_code', 500)
    
    class RestAPIError(Exception):
        def __init__(self, *args, **kwargs):
            super().__init__(*args)
            self.status_code = kwargs.get('status_code', 500)
    
    class tradeapi:
        class REST:
            def __init__(self, *args, **kwargs):
                raise ImportError("alpaca-trade-api not installed. Please install it with: pip install alpaca-trade-api")
from config import (
    ALPACA_API_KEY, 
    ALPACA_SECRET_KEY, 
    ALPACA_BASE_URL,
    MAX_RETRIES,
    RETRY_DELAY,
    REQUEST_TIMEOUT,
    MAX_POSITION_SIZE
)

logger = logging.getLogger(__name__)


class AlpacaClientError(Exception):
    """Custom exception for Alpaca client errors"""
    pass


class AlpacaClient:
    """Enhanced Alpaca client with error handling and retry logic"""
    
    def __init__(self):
        self.client = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            base_url=ALPACA_BASE_URL,
            api_version='v2'
        )
        self._validate_connection()
    
    def _validate_connection(self) -> None:
        """Validate API connection on initialization"""
        try:
            account = self.client.get_account()
            logger.info(f"Connected to Alpaca account: {account.account_number}")
        except Exception as e:
            raise AlpacaClientError(f"Failed to connect to Alpaca API: {e}")
    
    def _retry_operation(self, operation, *args, **kwargs):
        """Retry an operation with exponential backoff"""
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                return operation(*args, **kwargs)
            except (APIError, RestAPIError) as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"API error on attempt {attempt + 1}: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {MAX_RETRIES} attempts failed. Last error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise AlpacaClientError(f"Unexpected error: {e}")
        
        raise AlpacaClientError(f"Operation failed after {MAX_RETRIES} attempts: {last_exception}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.client.get_account()
            return {
                'account_number': account.account_number,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'daytrade_count': account.daytrade_count,
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise AlpacaClientError(f"Failed to get account info: {e}")
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position information for a symbol"""
        try:
            position = self.client.get_position(symbol)
            return {
                'symbol': position.symbol,
                'qty': float(position.qty),
                'market_value': float(position.market_value),
                'cost_basis': float(position.cost_basis),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'avg_entry_price': float(position.avg_entry_price)
            }
        except APIError as e:
            if e.status_code == 404:
                return None  # No position found
            logger.error(f"Failed to get position for {symbol}: {e}")
            raise AlpacaClientError(f"Failed to get position for {symbol}: {e}")
    
    def calculate_position_size(self, symbol: str, current_price: float) -> int:
        """Calculate appropriate position size based on risk management rules"""
        try:
            account_info = self.get_account_info()
            portfolio_value = account_info['portfolio_value']
            
            # Calculate position size based on maximum position size and portfolio risk
            max_position_value = min(
                MAX_POSITION_SIZE,
                portfolio_value * MAX_PORTFOLIO_RISK
            )
            
            # Calculate number of shares
            shares = int(max_position_value / current_price)
            
            logger.info(f"Calculated position size for {symbol}: {shares} shares (${max_position_value:.2f})")
            return max(1, shares)  # At least 1 share
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return 1  # Default to 1 share on error
    
    def place_order(self, symbol: str, side: str, qty: Optional[int] = None, 
                   current_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Place an order with enhanced error handling and position management"""
        try:
            side = side.lower()
            
            # Check current position
            position = self.get_position(symbol)
            
            if side == 'buy':
                if position and float(position['qty']) > 0:
                    logger.warning(f"Already holding {symbol}, skipping BUY to avoid wash trade")
                    return None
                
                # Calculate position size if not provided
                if qty is None:
                    if current_price is None:
                        # Get current market price
                        bars = self.client.get_bars(symbol, "1Min", limit=1)
                        if bars:
                            current_price = float(bars[symbol][0].c)
                        else:
                            logger.error(f"Could not get current price for {symbol}")
                            return None
                    qty = self.calculate_position_size(symbol, current_price)
            
            elif side == 'sell':
                if not position or float(position['qty']) <= 0:
                    logger.warning(f"No holdings of {symbol}, skipping SELL")
                    return None
                qty = int(position['qty'])  # Sell all shares
            
            # Place the order
            order = self.client.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'
            )
            
            logger.info(f"Order placed: {side.upper()} {qty} shares of {symbol}")
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'type': order.type,
                'time_in_force': order.time_in_force,
                'status': order.status,
                'submitted_at': order.submitted_at
            }
            
        except APIError as e:
            logger.error(f"Alpaca API error placing order for {symbol}: {e}")
            raise AlpacaClientError(f"Alpaca API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error placing order for {symbol}: {e}")
            raise AlpacaClientError(f"Unexpected error: {e}")
    
    def get_orders(self, symbol: Optional[str] = None, status: str = 'all') -> list:
        """Get order history"""
        try:
            orders = self.client.list_orders(status=status, symbols=symbol)
            return [{
                'id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'filled_qty': order.filled_qty,
                'filled_avg_price': order.filled_avg_price
            } for order in orders]
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            raise AlpacaClientError(f"Failed to get orders: {e}")


# Legacy functions for backward compatibility
def get_alpaca_client():
    """Legacy function - returns new AlpacaClient instance"""
    return AlpacaClient()


def place_order(client, symbol, side, qty=1, current_price=None):
    """Legacy function - places order using new client"""
    if isinstance(client, AlpacaClient):
        return client.place_order(symbol, side, qty, current_price)
    else:
        # Handle old client type
        logger.warning("Using legacy place_order function. Consider upgrading to AlpacaClient")
        try:
            position = client.get_position(symbol)
            if side.lower() == 'buy' and float(position.qty) > 0:
                logger.warning(f"Already holding {symbol}, skipping BUY to avoid wash trade.")
                return None
            if side.lower() == 'sell' and float(position.qty) == 0:
                logger.warning(f"No holdings of {symbol}, skipping SELL.")
                return None
        except Exception:
            if side.lower() == 'sell':
                logger.warning(f"No holdings of {symbol}, skipping SELL.")
                return None

        return client.submit_order(
            symbol=symbol,
            qty=qty,
            side=side.lower(),
            type='market',
            time_in_force='gtc'
        )
