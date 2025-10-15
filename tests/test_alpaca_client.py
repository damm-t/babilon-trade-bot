import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from broker.alpaca_client import AlpacaClient, AlpacaClientError


class TestAlpacaClient(unittest.TestCase):
    """Test cases for AlpacaClient with comprehensive error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_api_key = "test_api_key"
        self.mock_secret_key = "test_secret_key"
        self.mock_base_url = "https://paper-api.alpaca.markets"
        
        # Mock environment variables
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': self.mock_api_key,
            'ALPACA_SECRET_KEY': self.mock_secret_key,
            'ALPACA_BASE_URL': self.mock_base_url
        }):
            self.client = None
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_client_initialization_success(self, mock_rest):
        """Test successful client initialization"""
        # Mock successful account retrieval
        mock_account = Mock()
        mock_account.account_number = "123456"
        mock_rest.return_value.get_account.return_value = mock_account
        
        client = AlpacaClient()
        
        # Verify initialization
        mock_rest.assert_called_once_with(
            self.mock_api_key,
            self.mock_secret_key,
            base_url=self.mock_base_url,
            api_version='v2'
        )
        self.assertIsNotNone(client.client)
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_client_initialization_failure(self, mock_rest):
        """Test client initialization failure"""
        # Mock API connection failure
        mock_rest.return_value.get_account.side_effect = Exception("Connection failed")
        
        with self.assertRaises(AlpacaClientError):
            AlpacaClient()
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_get_account_info_success(self, mock_rest):
        """Test successful account info retrieval"""
        # Mock account data
        mock_account = Mock()
        mock_account.account_number = "123456"
        mock_account.buying_power = "10000.00"
        mock_account.cash = "5000.00"
        mock_account.portfolio_value = "15000.00"
        mock_account.equity = "15000.00"
        mock_account.last_equity = "14000.00"
        mock_account.daytrade_count = 5
        mock_account.pattern_day_trader = False
        mock_account.trading_blocked = False
        mock_account.account_blocked = False
        
        mock_rest.return_value.get_account.return_value = mock_account
        
        client = AlpacaClient()
        account_info = client.get_account_info()
        
        # Verify account info structure
        self.assertEqual(account_info['account_number'], "123456")
        self.assertEqual(account_info['buying_power'], 10000.0)
        self.assertEqual(account_info['cash'], 5000.0)
        self.assertEqual(account_info['portfolio_value'], 15000.0)
        self.assertFalse(account_info['trading_blocked'])
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_get_account_info_failure(self, mock_rest):
        """Test account info retrieval failure"""
        mock_rest.return_value.get_account.side_effect = Exception("API Error")
        
        client = AlpacaClient()
        
        with self.assertRaises(AlpacaClientError):
            client.get_account_info()
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_get_position_success(self, mock_rest):
        """Test successful position retrieval"""
        # Mock position data
        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "10.0"
        mock_position.market_value = "1500.00"
        mock_position.cost_basis = "1400.00"
        mock_position.unrealized_pl = "100.00"
        mock_position.unrealized_plpc = "0.0714"
        mock_position.avg_entry_price = "140.00"
        
        mock_rest.return_value.get_account.return_value = Mock()
        mock_rest.return_value.get_position.return_value = mock_position
        
        client = AlpacaClient()
        position = client.get_position("AAPL")
        
        # Verify position data
        self.assertEqual(position['symbol'], "AAPL")
        self.assertEqual(position['qty'], 10.0)
        self.assertEqual(position['market_value'], 1500.0)
        self.assertEqual(position['unrealized_pl'], 100.0)
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_get_position_not_found(self, mock_rest):
        """Test position not found (404 error)"""
        from alpaca_trade_api.rest import APIError
        
        # Mock 404 error
        mock_error = APIError()
        mock_error.status_code = 404
        mock_rest.return_value.get_account.return_value = Mock()
        mock_rest.return_value.get_position.side_effect = mock_error
        
        client = AlpacaClient()
        position = client.get_position("NONEXISTENT")
        
        # Should return None for non-existent position
        self.assertIsNone(position)
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_calculate_position_size(self, mock_rest):
        """Test position size calculation"""
        # Mock account with portfolio value
        mock_account = Mock()
        mock_account.portfolio_value = "10000.00"
        mock_rest.return_value.get_account.return_value = mock_account
        
        client = AlpacaClient()
        
        # Test position size calculation
        shares = client.calculate_position_size("AAPL", 150.0)
        
        # Should calculate based on MAX_POSITION_SIZE and portfolio risk
        expected_max_value = min(1000.0, 10000.0 * 0.02)  # 200
        expected_shares = int(expected_max_value / 150.0)  # 1 share
        self.assertEqual(shares, max(1, expected_shares))
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_place_order_buy_success(self, mock_rest):
        """Test successful buy order placement"""
        # Mock account and order data
        mock_account = Mock()
        mock_account.account_number = "123456"
        mock_rest.return_value.get_account.return_value = mock_account
        
        mock_order = Mock()
        mock_order.id = "order_123"
        mock_order.symbol = "AAPL"
        mock_order.qty = 1
        mock_order.side = "buy"
        mock_order.type = "market"
        mock_order.time_in_force = "gtc"
        mock_order.status = "submitted"
        mock_order.submitted_at = "2023-01-01T10:00:00Z"
        
        mock_rest.return_value.get_position.side_effect = APIError()  # No position
        mock_rest.return_value.submit_order.return_value = mock_order
        
        client = AlpacaClient()
        result = client.place_order("AAPL", "buy", current_price=150.0)
        
        # Verify order was placed
        self.assertIsNotNone(result)
        self.assertEqual(result['id'], "order_123")
        self.assertEqual(result['side'], "buy")
        self.assertEqual(result['symbol'], "AAPL")
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_place_order_already_holding(self, mock_rest):
        """Test buy order when already holding position"""
        # Mock account
        mock_account = Mock()
        mock_account.account_number = "123456"
        mock_rest.return_value.get_account.return_value = mock_account
        
        # Mock existing position
        mock_position = Mock()
        mock_position.qty = "5.0"
        mock_rest.return_value.get_position.return_value = mock_position
        
        client = AlpacaClient()
        result = client.place_order("AAPL", "buy", current_price=150.0)
        
        # Should return None (no order placed)
        self.assertIsNone(result)
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_place_order_sell_success(self, mock_rest):
        """Test successful sell order placement"""
        # Mock account
        mock_account = Mock()
        mock_account.account_number = "123456"
        mock_rest.return_value.get_account.return_value = mock_account
        
        # Mock existing position
        mock_position = Mock()
        mock_position.qty = "5.0"
        mock_position.symbol = "AAPL"
        mock_position.market_value = "750.00"
        mock_position.cost_basis = "700.00"
        mock_position.unrealized_pl = "50.00"
        mock_position.unrealized_plpc = "0.0714"
        mock_position.avg_entry_price = "140.00"
        mock_rest.return_value.get_position.return_value = mock_position
        
        # Mock order
        mock_order = Mock()
        mock_order.id = "order_456"
        mock_order.symbol = "AAPL"
        mock_order.qty = 5
        mock_order.side = "sell"
        mock_order.type = "market"
        mock_order.time_in_force = "gtc"
        mock_order.status = "submitted"
        mock_order.submitted_at = "2023-01-01T10:00:00Z"
        mock_rest.return_value.submit_order.return_value = mock_order
        
        client = AlpacaClient()
        result = client.place_order("AAPL", "sell", current_price=150.0)
        
        # Verify sell order
        self.assertIsNotNone(result)
        self.assertEqual(result['side'], "sell")
        self.assertEqual(result['qty'], 5)
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_place_order_no_position_to_sell(self, mock_rest):
        """Test sell order when no position exists"""
        # Mock account
        mock_account = Mock()
        mock_account.account_number = "123456"
        mock_rest.return_value.get_account.return_value = mock_account
        
        # Mock no position (404 error)
        from alpaca_trade_api.rest import APIError
        mock_error = APIError()
        mock_error.status_code = 404
        mock_rest.return_value.get_position.side_effect = mock_error
        
        client = AlpacaClient()
        result = client.place_order("AAPL", "sell", current_price=150.0)
        
        # Should return None (no order placed)
        self.assertIsNone(result)
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_place_order_api_error(self, mock_rest):
        """Test order placement with API error"""
        # Mock account
        mock_account = Mock()
        mock_account.account_number = "123456"
        mock_rest.return_value.get_account.return_value = mock_account
        
        # Mock API error
        from alpaca_trade_api.rest import APIError
        mock_error = APIError()
        mock_error.status_code = 500
        mock_rest.return_value.get_position.side_effect = mock_error
        
        client = AlpacaClient()
        
        with self.assertRaises(AlpacaClientError):
            client.place_order("AAPL", "buy", current_price=150.0)
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_get_orders_success(self, mock_rest):
        """Test successful order retrieval"""
        # Mock account
        mock_account = Mock()
        mock_account.account_number = "123456"
        mock_rest.return_value.get_account.return_value = mock_account
        
        # Mock orders
        mock_order1 = Mock()
        mock_order1.id = "order_1"
        mock_order1.symbol = "AAPL"
        mock_order1.qty = 1
        mock_order1.side = "buy"
        mock_order1.type = "market"
        mock_order1.status = "filled"
        mock_order1.submitted_at = "2023-01-01T10:00:00Z"
        mock_order1.filled_at = "2023-01-01T10:01:00Z"
        mock_order1.filled_qty = 1
        mock_order1.filled_avg_price = "150.00"
        
        mock_rest.return_value.list_orders.return_value = [mock_order1]
        
        client = AlpacaClient()
        orders = client.get_orders("AAPL")
        
        # Verify orders
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['id'], "order_1")
        self.assertEqual(orders[0]['symbol'], "AAPL")
        self.assertEqual(orders[0]['status'], "filled")
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_get_orders_failure(self, mock_rest):
        """Test order retrieval failure"""
        # Mock account
        mock_account = Mock()
        mock_account.account_number = "123456"
        mock_rest.return_value.get_account.return_value = mock_account
        
        # Mock API error
        mock_rest.return_value.list_orders.side_effect = Exception("API Error")
        
        client = AlpacaClient()
        
        with self.assertRaises(AlpacaClientError):
            client.get_orders()


class TestAlpacaClientEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret',
            'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets'
        }):
            pass
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_calculate_position_size_edge_cases(self, mock_rest):
        """Test position size calculation edge cases"""
        mock_account = Mock()
        mock_account.portfolio_value = "100.00"  # Very small portfolio
        mock_rest.return_value.get_account.return_value = mock_account
        
        client = AlpacaClient()
        
        # Test with very high price
        shares = client.calculate_position_size("AAPL", 10000.0)
        self.assertEqual(shares, 1)  # Should be at least 1
        
        # Test with zero price (should handle gracefully)
        shares = client.calculate_position_size("AAPL", 0.0)
        self.assertEqual(shares, 1)  # Should default to 1
    
    @patch('broker.alpaca_client.tradeapi.REST')
    def test_retry_logic(self, mock_rest):
        """Test retry logic with exponential backoff"""
        mock_account = Mock()
        mock_account.account_number = "123456"
        mock_rest.return_value.get_account.return_value = mock_account
        
        # Mock API error that eventually succeeds
        from alpaca_trade_api.rest import APIError
        mock_error = APIError()
        mock_error.status_code = 500
        
        mock_rest.return_value.get_position.side_effect = [
            mock_error,  # First attempt fails
            mock_error,  # Second attempt fails
            Mock(qty="10.0")  # Third attempt succeeds
        ]
        
        client = AlpacaClient()
        
        # Should eventually succeed after retries
        position = client.get_position("AAPL")
        self.assertIsNotNone(position)


if __name__ == '__main__':
    unittest.main()

