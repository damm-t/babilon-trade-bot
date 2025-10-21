"""
Broker Executor for Phase 6
Implements broker wrapper with retries, order monitoring, and safety constraints
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import requests
import json

from .risk_manager import RiskManager, PositionSizing
from .hybrid_signal import HybridSignal

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """Order data structure"""
    id: str
    ticker: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    error_message: Optional[str] = None


@dataclass
class Position:
    """Position data structure"""
    ticker: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    cost_basis: float


@dataclass
class Account:
    """Account data structure"""
    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    equity: float
    positions: Dict[str, Position]
    orders: List[Order]


class BrokerExecutor:
    """Broker executor with retry logic and safety constraints"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.risk_manager = RiskManager()
        self.orders = {}
        self.positions = {}
        self.account = None
        self.retry_count = 0
        self.max_retries = self.config['max_retries']
        
    def _get_default_config(self) -> Dict:
        """Get default broker configuration"""
        return {
            'api_key': '',
            'secret_key': '',
            'base_url': 'https://paper-api.alpaca.markets',
            'data_url': 'https://data.alpaca.markets',
            'max_retries': 3,
            'retry_delay': 1.0,  # seconds
            'timeout': 30.0,     # seconds
            'max_order_size': 1000,  # shares
            'min_order_size': 1,     # shares
            'commission_rate': 0.001,  # 0.1%
            'slippage_rate': 0.0005,    # 0.05%
            'paper_trading': True,
            'rate_limit_delay': 0.1  # seconds between requests
        }
    
    def initialize(self, api_key: str, secret_key: str, paper_trading: bool = True):
        """Initialize broker connection"""
        self.config['api_key'] = api_key
        self.config['secret_key'] = secret_key
        self.config['paper_trading'] = paper_trading
        
        if paper_trading:
            self.config['base_url'] = 'https://paper-api.alpaca.markets'
        else:
            self.config['base_url'] = 'https://api.alpaca.markets'
        
        logger.info(f"Initialized broker executor (paper_trading: {paper_trading})")
        
        # Test connection
        try:
            self._get_account()
            logger.info("Broker connection successful")
            return True
        except Exception as e:
            logger.error(f"Broker connection failed: {e}")
            return False
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make authenticated request to broker API"""
        url = f"{self.config['base_url']}{endpoint}"
        headers = {
            'APCA-API-KEY-ID': self.config['api_key'],
            'APCA-API-SECRET-KEY': self.config['secret_key'],
            'Content-Type': 'application/json'
        }
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=self.config['timeout'])
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data, timeout=self.config['timeout'])
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=self.config['timeout'])
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise
    
    def _get_account(self) -> Account:
        """Get account information"""
        try:
            data = self._make_request('GET', '/v2/account')
            
            self.account = Account(
                account_id=data['id'],
                buying_power=float(data['buying_power']),
                cash=float(data['cash']),
                portfolio_value=float(data['portfolio_value']),
                equity=float(data['equity']),
                positions={},
                orders=[]
            )
            
            return self.account
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        try:
            data = self._make_request('GET', '/v2/positions')
            
            positions = {}
            for pos_data in data:
                ticker = pos_data['symbol']
                positions[ticker] = Position(
                    ticker=ticker,
                    quantity=float(pos_data['qty']),
                    avg_price=float(pos_data['avg_entry_price']),
                    market_value=float(pos_data['market_value']),
                    unrealized_pnl=float(pos_data['unrealized_pl']),
                    realized_pnl=float(pos_data['realized_pl']),
                    cost_basis=float(pos_data['cost_basis'])
                )
            
            self.positions = positions
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}
    
    def get_orders(self, status: Optional[str] = None) -> List[Order]:
        """Get orders with optional status filter"""
        try:
            endpoint = '/v2/orders'
            if status:
                endpoint += f'?status={status}'
            
            data = self._make_request('GET', endpoint)
            
            orders = []
            for order_data in data:
                order = Order(
                    id=order_data['id'],
                    ticker=order_data['symbol'],
                    side=order_data['side'],
                    quantity=float(order_data['qty']),
                    order_type=OrderType(order_data['order_type']),
                    price=float(order_data['limit_price']) if order_data.get('limit_price') else None,
                    stop_price=float(order_data['stop_price']) if order_data.get('stop_price') else None,
                    time_in_force=order_data['time_in_force'],
                    status=OrderStatus(order_data['status']),
                    filled_quantity=float(order_data['filled_qty']),
                    filled_price=float(order_data['filled_avg_price']) if order_data.get('filled_avg_price') else None,
                    commission=float(order_data.get('commission', 0)),
                    created_at=datetime.fromisoformat(order_data['created_at'].replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(order_data['updated_at'].replace('Z', '+00:00'))
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def submit_order(self, 
                    ticker: str,
                    side: str,
                    quantity: float,
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    time_in_force: str = "day") -> Order:
        """Submit order with safety checks and retry logic"""
        
        # Validate order parameters
        validation_result = self._validate_order(ticker, side, quantity, order_type, price)
        if not validation_result[0]:
            raise ValueError(f"Order validation failed: {validation_result[1]}")
        
        # Apply risk management
        risk_check = self._check_order_risk(ticker, side, quantity)
        if not risk_check[0]:
            raise ValueError(f"Risk check failed: {risk_check[1]}")
        
        # Prepare order data
        order_data = {
            'symbol': ticker,
            'qty': str(int(quantity)),
            'side': side,
            'type': order_type.value,
            'time_in_force': time_in_force
        }
        
        if order_type == OrderType.LIMIT and price:
            order_data['limit_price'] = str(price)
        elif order_type == OrderType.STOP and stop_price:
            order_data['stop_price'] = str(stop_price)
        elif order_type == OrderType.STOP_LIMIT and price and stop_price:
            order_data['limit_price'] = str(price)
            order_data['stop_price'] = str(stop_price)
        
        # Submit order with retry logic
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Submitting order: {ticker} {side} {quantity} (attempt {attempt + 1})")
                
                response = self._make_request('POST', '/v2/orders', order_data)
                
                # Create order object
                order = Order(
                    id=response['id'],
                    ticker=ticker,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    price=price,
                    stop_price=stop_price,
                    time_in_force=time_in_force,
                    status=OrderStatus.SUBMITTED,
                    created_at=datetime.now()
                )
                
                self.orders[order.id] = order
                logger.info(f"Order submitted successfully: {order.id}")
                
                return order
                
            except Exception as e:
                logger.warning(f"Order submission attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.config['retry_delay'] * (attempt + 1))
                else:
                    raise Exception(f"Order submission failed after {self.max_retries} attempts: {e}")
    
    def _validate_order(self, ticker: str, side: str, quantity: float, 
                       order_type: OrderType, price: Optional[float]) -> Tuple[bool, str]:
        """Validate order parameters"""
        
        # Check ticker format
        if not ticker or len(ticker) < 1:
            return False, "Invalid ticker symbol"
        
        # Check side
        if side not in ['buy', 'sell']:
            return False, "Invalid side (must be 'buy' or 'sell')"
        
        # Check quantity
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if quantity < self.config['min_order_size']:
            return False, f"Quantity below minimum ({self.config['min_order_size']})"
        
        if quantity > self.config['max_order_size']:
            return False, f"Quantity above maximum ({self.config['max_order_size']})"
        
        # Check order type and price
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            return False, "Price required for limit orders"
        
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and price is None:
            return False, "Stop price required for stop orders"
        
        return True, "Valid order"
    
    def _check_order_risk(self, ticker: str, side: str, quantity: float) -> Tuple[bool, str]:
        """Check order against risk management rules"""
        
        # Get current account info
        try:
            account = self._get_account()
        except Exception as e:
            return False, f"Failed to get account info: {e}"
        
        # Check buying power for buy orders
        if side == 'buy':
            estimated_cost = quantity * 100  # Rough estimate
            if estimated_cost > account.buying_power:
                return False, f"Insufficient buying power: {estimated_cost} > {account.buying_power}"
        
        # Check position limits
        current_positions = self.get_positions()
        if len(current_positions) >= 5:  # Max 5 positions
            return False, "Maximum number of positions reached"
        
        # Check individual position size
        position_value = quantity * 100  # Rough estimate
        portfolio_value = account.portfolio_value
        position_ratio = position_value / portfolio_value if portfolio_value > 0 else 0
        
        if position_ratio > 0.2:  # Max 20% per position
            return False, f"Position size too large: {position_ratio:.2%} > 20%"
        
        return True, "Risk check passed"
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self._make_request('DELETE', f'/v2/orders/{order_id}')
            logger.info(f"Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders"""
        try:
            self._make_request('DELETE', '/v2/orders')
            logger.info("All orders cancelled successfully")
            return 0  # Success
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 1  # Failure
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status"""
        try:
            data = self._make_request('GET', f'/v2/orders/{order_id}')
            return OrderStatus(data['status'])
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None
    
    def wait_for_fill(self, order_id: str, timeout: int = 300) -> bool:
        """Wait for order to fill with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_order_status(order_id)
            
            if status == OrderStatus.FILLED:
                logger.info(f"Order {order_id} filled successfully")
                return True
            elif status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                logger.warning(f"Order {order_id} ended with status: {status}")
                return False
            
            time.sleep(1)  # Check every second
        
        logger.warning(f"Order {order_id} timed out after {timeout} seconds")
        return False
    
    def execute_signal(self, signal: HybridSignal, current_price: float) -> Optional[Order]:
        """Execute a trading signal"""
        
        if signal.action == "HOLD":
            logger.info("Signal is HOLD - no action taken")
            return None
        
        # Determine order parameters
        side = "buy" if signal.action == "BUY" else "sell"
        
        # Calculate quantity based on signal confidence and risk management
        position_sizing = self.risk_manager.calculate_position_size(
            signal_confidence=signal.confidence,
            volatility=signal.risk_metrics.get('volatility_risk', 0.2),
            portfolio_value=10000,  # Default portfolio value
            current_positions=self.positions,
            ticker="AAPL"  # Default ticker - should be passed from signal
        )
        
        quantity = position_sizing.risk_adjusted_size
        
        if quantity <= 0:
            logger.warning("Position size is zero or negative - skipping order")
            return None
        
        # Submit order
        try:
            order = self.submit_order(
                ticker="AAPL",  # Should be extracted from signal
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            
            logger.info(f"Signal executed: {signal.action} {quantity} shares")
            return order
            
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            return None
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        try:
            account = self._get_account()
            positions = self.get_positions()
            orders = self.get_orders()
            
            return {
                'account_id': account.account_id,
                'buying_power': account.buying_power,
                'cash': account.cash,
                'portfolio_value': account.portfolio_value,
                'equity': account.equity,
                'num_positions': len(positions),
                'num_orders': len(orders),
                'positions': {ticker: {
                    'quantity': pos.quantity,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl
                } for ticker, pos in positions.items()}
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            return {}


def create_broker_executor(config: Optional[Dict] = None) -> BrokerExecutor:
    """Create a broker executor with optional configuration"""
    return BrokerExecutor(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create broker executor
    executor = create_broker_executor()
    
    print("Broker executor ready for paper trading")
