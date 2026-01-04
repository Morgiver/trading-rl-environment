"""Base class and implementations for action execution strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from gymnasium import spaces

from trading_simulator import TradingSimulator, OrderType, OrderSide


class ActionStrategy(ABC):
    """
    Base class for action execution strategies.

    Subclass this to create custom action spaces and execution logic.
    This allows flexible mapping between RL agent actions and trading operations.
    """

    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        """
        Get the Gymnasium action space.

        Returns:
            Gymnasium Space object defining valid actions
        """
        pass

    @abstractmethod
    def execute_action(
        self, action: Any, simulator: TradingSimulator, env_state: Dict[str, Any]
    ) -> None:
        """
        Execute the given action using the simulator.

        Args:
            action: Action from the agent (format depends on action space)
            simulator: TradingSimulator instance to execute orders on
            env_state: Current environment state dict containing:
                - simulator_state: State from simulator.get_state()
                - current_candle: Current market candle
                - trade_quantity: Configured trade quantity
                etc.

        Raises:
            ValueError: If action is invalid
            RuntimeError: If execution fails (e.g., insufficient balance)
        """
        pass

    @abstractmethod
    def get_action_description(self, action: Any) -> str:
        """
        Get human-readable description of an action.

        Args:
            action: Action to describe

        Returns:
            String description of the action
        """
        pass


class SimpleActionStrategy(ActionStrategy):
    """
    Simple 3-action strategy: HOLD, BUY, SELL.

    Action space: Discrete(3)
    - 0: HOLD (do nothing)
    - 1: BUY (market order)
    - 2: SELL (market order or close position)

    This is the default strategy, compatible with basic RL algorithms.
    """

    def __init__(self, trade_quantity: float = 1.0):
        """
        Initialize SimpleActionStrategy.

        Args:
            trade_quantity: Fixed quantity to trade per action
        """
        self.trade_quantity = trade_quantity

    def get_action_space(self) -> spaces.Discrete:
        """Return discrete action space with 3 actions."""
        return spaces.Discrete(3)

    def execute_action(
        self, action: int, simulator: TradingSimulator, env_state: Dict[str, Any]
    ) -> None:
        """Execute simple BUY/SELL/HOLD action."""
        if action == 0:
            # HOLD - do nothing
            return

        elif action == 1:
            # BUY - market order
            try:
                simulator.place_order(
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=self.trade_quantity,
                )
            except RuntimeError:
                # Insufficient balance or other error - skip
                pass

        elif action == 2:
            # SELL - close position or open short
            position = simulator.get_position()

            if position.is_long:
                # Close long position
                try:
                    simulator.place_order(
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        quantity=min(position.quantity, self.trade_quantity),
                    )
                except RuntimeError:
                    pass
            elif position.is_flat:
                # Open short position
                try:
                    simulator.place_order(
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        quantity=self.trade_quantity,
                    )
                except RuntimeError:
                    pass

        else:
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, or 2")

    def get_action_description(self, action: int) -> str:
        """Get description of action."""
        descriptions = {
            0: "HOLD (do nothing)",
            1: f"BUY {self.trade_quantity} units (market)",
            2: f"SELL {self.trade_quantity} units (market)",
        }
        return descriptions.get(action, f"Unknown action: {action}")


class ExtendedActionStrategy(ActionStrategy):
    """
    Extended action strategy with 9 discrete actions.

    Action space: Discrete(9)
    - 0: HOLD
    - 1: BUY_MARKET (immediate buy at market price)
    - 2: SELL_MARKET (immediate sell at market price)
    - 3: BUY_LIMIT (limit buy 1% below current price)
    - 4: SELL_LIMIT (limit sell 1% above current price)
    - 5: CLOSE_POSITION (close entire position)
    - 6: CANCEL_ALL_ORDERS (cancel all pending orders)
    - 7: SET_STOP_LOSS (place stop loss at -2% from current price)
    - 8: SET_TAKE_PROFIT (place take profit at +2% from current price)
    """

    def __init__(
        self,
        trade_quantity: float = 1.0,
        limit_offset_pct: float = 0.01,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.02,
    ):
        """
        Initialize ExtendedActionStrategy.

        Args:
            trade_quantity: Fixed quantity to trade per action
            limit_offset_pct: Price offset for limit orders (e.g., 0.01 = 1%)
            stop_loss_pct: Stop loss distance from current price (e.g., 0.02 = 2%)
            take_profit_pct: Take profit distance from current price (e.g., 0.02 = 2%)
        """
        self.trade_quantity = trade_quantity
        self.limit_offset_pct = limit_offset_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def get_action_space(self) -> spaces.Discrete:
        """Return discrete action space with 9 actions."""
        return spaces.Discrete(9)

    def execute_action(
        self, action: int, simulator: TradingSimulator, env_state: Dict[str, Any]
    ) -> None:
        """Execute extended action."""
        current_price = simulator.last_price
        position = simulator.get_position()

        if action == 0:
            # HOLD
            return

        elif action == 1:
            # BUY_MARKET
            try:
                simulator.place_order(
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=self.trade_quantity,
                )
            except RuntimeError:
                pass

        elif action == 2:
            # SELL_MARKET
            try:
                if position.is_long:
                    qty = min(position.quantity, self.trade_quantity)
                else:
                    qty = self.trade_quantity

                simulator.place_order(
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    quantity=qty,
                )
            except RuntimeError:
                pass

        elif action == 3:
            # BUY_LIMIT (1% below current price)
            limit_price = current_price * (1 - self.limit_offset_pct)
            try:
                simulator.place_order(
                    order_type=OrderType.LIMIT,
                    side=OrderSide.BUY,
                    quantity=self.trade_quantity,
                    price=limit_price,
                )
            except RuntimeError:
                pass

        elif action == 4:
            # SELL_LIMIT (1% above current price)
            limit_price = current_price * (1 + self.limit_offset_pct)
            try:
                simulator.place_order(
                    order_type=OrderType.LIMIT,
                    side=OrderSide.SELL,
                    quantity=self.trade_quantity,
                    price=limit_price,
                )
            except RuntimeError:
                pass

        elif action == 5:
            # CLOSE_POSITION (close entire position)
            if not position.is_flat:
                try:
                    side = OrderSide.SELL if position.is_long else OrderSide.BUY
                    simulator.place_order(
                        order_type=OrderType.MARKET,
                        side=side,
                        quantity=abs(position.quantity),
                    )
                except RuntimeError:
                    pass

        elif action == 6:
            # CANCEL_ALL_ORDERS
            pending_orders = simulator.get_pending_orders()
            for order in pending_orders:
                try:
                    simulator.cancel_order(order.order_id)
                except RuntimeError:
                    pass

        elif action == 7:
            # SET_STOP_LOSS (only if position exists)
            if not position.is_flat:
                try:
                    if position.is_long:
                        stop_price = current_price * (1 - self.stop_loss_pct)
                        simulator.place_order(
                            order_type=OrderType.STOP_LOSS,
                            side=OrderSide.SELL,
                            quantity=abs(position.quantity),
                            price=stop_price,
                        )
                    else:  # is_short
                        stop_price = current_price * (1 + self.stop_loss_pct)
                        simulator.place_order(
                            order_type=OrderType.STOP_LOSS,
                            side=OrderSide.BUY,
                            quantity=abs(position.quantity),
                            price=stop_price,
                        )
                except RuntimeError:
                    pass

        elif action == 8:
            # SET_TAKE_PROFIT (only if position exists)
            if not position.is_flat:
                try:
                    if position.is_long:
                        tp_price = current_price * (1 + self.take_profit_pct)
                        simulator.place_order(
                            order_type=OrderType.TAKE_PROFIT,
                            side=OrderSide.SELL,
                            quantity=abs(position.quantity),
                            price=tp_price,
                        )
                    else:  # is_short
                        tp_price = current_price * (1 - self.take_profit_pct)
                        simulator.place_order(
                            order_type=OrderType.TAKE_PROFIT,
                            side=OrderSide.BUY,
                            quantity=abs(position.quantity),
                            price=tp_price,
                        )
                except RuntimeError:
                    pass

        else:
            raise ValueError(f"Invalid action: {action}. Must be 0-8")

    def get_action_description(self, action: int) -> str:
        """Get description of action."""
        descriptions = {
            0: "HOLD",
            1: f"BUY {self.trade_quantity} units (market)",
            2: f"SELL {self.trade_quantity} units (market)",
            3: f"BUY {self.trade_quantity} units (limit -{self.limit_offset_pct*100}%)",
            4: f"SELL {self.trade_quantity} units (limit +{self.limit_offset_pct*100}%)",
            5: "CLOSE entire position",
            6: "CANCEL all pending orders",
            7: f"SET STOP LOSS (-{self.stop_loss_pct*100}%)",
            8: f"SET TAKE PROFIT (+{self.take_profit_pct*100}%)",
        }
        return descriptions.get(action, f"Unknown action: {action}")


class ContinuousActionStrategy(ActionStrategy):
    """
    Continuous action strategy using Box space.

    Action space: Box(low=-1, high=1, shape=(3,))
    - action[0]: Order type [-1, 1] mapped to [MARKET, LIMIT, STOP_LOSS, TAKE_PROFIT]
    - action[1]: Side [-1, 1] mapped to [BUY, SELL] (includes position close)
    - action[2]: Quantity ratio [-1, 1] mapped to [0, 2*trade_quantity]

    This allows fine-grained control over trading parameters.
    """

    def __init__(self, trade_quantity: float = 1.0, max_price_offset_pct: float = 0.05):
        """
        Initialize ContinuousActionStrategy.

        Args:
            trade_quantity: Base trade quantity
            max_price_offset_pct: Maximum price offset for limit/stop orders (e.g., 0.05 = 5%)
        """
        self.trade_quantity = trade_quantity
        self.max_price_offset_pct = max_price_offset_pct

    def get_action_space(self) -> spaces.Box:
        """Return continuous action space."""
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def execute_action(
        self, action: np.ndarray, simulator: TradingSimulator, env_state: Dict[str, Any]
    ) -> None:
        """Execute continuous action."""
        # Decode action
        order_type_val, side_val, quantity_val = action

        # Map order type: [-1, 1] → [0, 3]
        order_type_idx = int((order_type_val + 1) * 1.5)  # Maps to 0, 1, 2, 3
        order_type_idx = np.clip(order_type_idx, 0, 3)
        order_types = [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]
        order_type = order_types[order_type_idx]

        # Map side: [-1, 1] → [BUY, SELL]
        side = OrderSide.BUY if side_val < 0 else OrderSide.SELL

        # Map quantity: [-1, 1] → [0, 2*trade_quantity]
        quantity = self.trade_quantity * (1 + quantity_val)
        quantity = np.clip(quantity, 0.01, 2 * self.trade_quantity)

        # Calculate price for non-market orders
        current_price = simulator.last_price
        price = None

        if order_type != OrderType.MARKET:
            # Use quantity_val as price offset indicator
            price_offset_pct = quantity_val * self.max_price_offset_pct
            if side == OrderSide.BUY:
                price = current_price * (1 - abs(price_offset_pct))
            else:
                price = current_price * (1 + abs(price_offset_pct))

        # Execute order
        try:
            simulator.place_order(
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
            )
        except (ValueError, RuntimeError):
            # Invalid order or insufficient balance - skip
            pass

    def get_action_description(self, action: np.ndarray) -> str:
        """Get description of continuous action."""
        order_type_val, side_val, quantity_val = action

        order_type_idx = int((order_type_val + 1) * 1.5)
        order_type_idx = np.clip(order_type_idx, 0, 3)
        order_types_str = ["MARKET", "LIMIT", "STOP_LOSS", "TAKE_PROFIT"]
        order_type_str = order_types_str[order_type_idx]

        side_str = "BUY" if side_val < 0 else "SELL"
        quantity = self.trade_quantity * (1 + quantity_val)

        return f"{side_str} {quantity:.2f} units ({order_type_str})"
