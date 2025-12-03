"""
trading/execution_system.py
High-performance execution system for Python 3.14
"""

from enum import Enum
from alpaca_trade_api.rest import APIError
import alpaca_trade_api as tradeapi
from dataclasses import dataclass, field
import asyncio
import threading
from typing import Any, Optional
from collections import deque
import uuid
from datetime import datetime

from settings import FREE_THREADED

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Order container"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class ExecutionEngine:
    """Python 3.14 execution engine with threading support"""
    
    def __init__(self, use_threading: bool = FREE_THREADED) -> None:
        self.use_threading: bool = use_threading
        self.orders: dict[str, Order] = {}
        self.order_queue: deque[Order] = deque()
        self.execution_stats: dict[str, Any] = {
            'orders_executed': 0,
            'total_volume': 0.0,
            'avg_execution_time': 0.0,
            'fill_rate': 0.0
        }
        self.is_running: bool = False
    
    def start(self) -> None:
        """Start execution engine"""
        self.is_running = True
        
        if self.use_threading:
            # Python 3.14: Use multiple interpreters for isolation
            self.execution_thread = threading.Thread(target=self._execution_loop)
            self.execution_thread.start()
    
    def stop(self) -> None:
        """Stop execution engine"""
        self.is_running = False
        if hasattr(self, 'execution_thread'):
            self.execution_thread.join()
    
    def submit_order(self, order: Order) -> str:
        """Submit order for execution"""
        self.orders[order.id] = order
        self.order_queue.append(order)
        return order.id
    
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = 'market',
        price: Optional[float] = None
    ) -> Optional[str]:
        """Execute trade asynchronously"""
        
        order = Order(
            symbol=symbol,
            side=OrderSide(side.lower()),
            order_type=OrderType(order_type.lower()),
            quantity=quantity,
            price=price
        )
        
        order_id = self.submit_order(order)
        
        self.execution_stats['orders_executed'] += 1
        self.execution_stats['total_volume'] += quantity
        
        return order_id
    
    def _execution_loop(self) -> None:
        """Main execution loop"""
        while self.is_running:
            try:
                if len(self.order_queue) > 0:
                    order = self.order_queue.popleft()
                    self._process_order(order)
                
                threading.Event().wait(0.1)
            except Exception as e:
                print(f"Error in execution loop: {e}")
    
    def _process_order(self, order: Order) -> None:
        """Process individual order"""
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity

class AlpacaExecutor:
    """Execute trades via Alpaca"""
    
    def __init__(self, api: tradeapi.REST):
        self.api = api
    
    async def place_order(self, symbol: str, qty: int, side: str, type: str = 'market'):
        """Place an order"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force='gtc'
            )
            return order
        except APIError as e:
            print(f"Order failed: {e}")
            return None