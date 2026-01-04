"""Base class and implementations for reward strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class RewardStrategy(ABC):
    """
    Base class for reward calculation strategies.

    Subclass this to create custom reward functions for your RL agent.
    """

    @abstractmethod
    def calculate(
        self, env_state: Dict[str, Any], action: int, next_env_state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for a state transition.

        Args:
            env_state: Current environment state before action
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            next_env_state: Environment state after action

        Returns:
            Reward value (float)

        State dictionary structure:
            {
                'simulator_state': {
                    'balance': float,
                    'position': {
                        'quantity': float,
                        'average_price': float,
                        'side': str or None,
                        'is_flat': bool
                    },
                    'pnl': {
                        'realized': float,
                        'unrealized': float,
                        'total': float,
                        'fees': float,
                        'net': float
                    },
                    'last_price': float,
                    'pending_orders_count': int,
                    'equity': float
                },
                'asset_view_stats': {
                    'symbol': str,
                    'total_periods': dict,
                    'price_range': dict,
                    'volume_stats': dict,
                    'latest_prices': dict,
                    ...
                },
                'current_step': int
            }
        """
        pass

    def reset(self) -> None:
        """
        Reset strategy state (if any) at episode start.

        Override this if your strategy maintains state between steps.
        """
        pass


class SimplePnLReward(RewardStrategy):
    """
    Simple reward based on change in unrealized PnL.

    Rewards the agent for increasing unrealized profits and penalizes
    for increasing unrealized losses.
    """

    def __init__(self, scale: float = 0.01):
        """
        Initialize SimplePnLReward.

        Args:
            scale: Scaling factor for reward normalization (default: 0.01)
        """
        self.scale = scale

    def calculate(
        self, env_state: Dict[str, Any], action: int, next_env_state: Dict[str, Any]
    ) -> float:
        """Calculate reward as scaled PnL change."""
        prev_pnl = env_state["simulator_state"]["pnl"]["unrealized"]
        next_pnl = next_env_state["simulator_state"]["pnl"]["unrealized"]

        pnl_change = next_pnl - prev_pnl
        return pnl_change * self.scale


class RealizedPnLReward(RewardStrategy):
    """
    Reward based on realized PnL only.

    Only gives rewards when positions are closed and PnL is realized.
    """

    def __init__(self, scale: float = 0.01):
        """
        Initialize RealizedPnLReward.

        Args:
            scale: Scaling factor for reward normalization
        """
        self.scale = scale
        self.last_realized_pnl = 0.0

    def calculate(
        self, env_state: Dict[str, Any], action: int, next_env_state: Dict[str, Any]
    ) -> float:
        """Calculate reward as change in realized PnL."""
        next_realized = next_env_state["simulator_state"]["pnl"]["realized"]
        realized_change = next_realized - self.last_realized_pnl
        self.last_realized_pnl = next_realized

        return realized_change * self.scale

    def reset(self) -> None:
        """Reset realized PnL tracker."""
        self.last_realized_pnl = 0.0


class SharpeRatioReward(RewardStrategy):
    """
    Reward based on approximate Sharpe ratio.

    Encourages consistent returns with low volatility.
    """

    def __init__(self, window_size: int = 20, scale: float = 1.0):
        """
        Initialize SharpeRatioReward.

        Args:
            window_size: Number of steps to calculate Sharpe ratio over
            scale: Scaling factor for reward
        """
        self.window_size = window_size
        self.scale = scale
        self.returns_history = []

    def calculate(
        self, env_state: Dict[str, Any], action: int, next_env_state: Dict[str, Any]
    ) -> float:
        """Calculate reward based on Sharpe ratio approximation."""
        # Calculate return
        prev_equity = env_state["simulator_state"]["equity"]
        next_equity = next_env_state["simulator_state"]["equity"]

        if prev_equity == 0:
            return_pct = 0.0
        else:
            return_pct = (next_equity - prev_equity) / prev_equity

        # Track returns
        self.returns_history.append(return_pct)
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        # Not enough data yet
        if len(self.returns_history) < 2:
            return return_pct * self.scale

        # Calculate Sharpe approximation
        import numpy as np

        mean_return = np.mean(self.returns_history)
        std_return = np.std(self.returns_history)

        if std_return == 0:
            sharpe = 0.0
        else:
            sharpe = mean_return / std_return

        return sharpe * self.scale

    def reset(self) -> None:
        """Reset returns history."""
        self.returns_history = []


class RiskAdjustedReward(RewardStrategy):
    """
    Reward that balances profit with risk management.

    Penalizes large drawdowns and rewards profitable trades.
    """

    def __init__(
        self,
        profit_weight: float = 1.0,
        drawdown_penalty: float = 2.0,
        position_penalty: float = 0.1,
        scale: float = 0.01,
    ):
        """
        Initialize RiskAdjustedReward.

        Args:
            profit_weight: Weight for profit component
            drawdown_penalty: Penalty multiplier for drawdowns
            position_penalty: Small penalty for holding positions (encourages turnover)
            scale: Overall scaling factor
        """
        self.profit_weight = profit_weight
        self.drawdown_penalty = drawdown_penalty
        self.position_penalty = position_penalty
        self.scale = scale
        self.max_equity = 0.0

    def calculate(
        self, env_state: Dict[str, Any], action: int, next_env_state: Dict[str, Any]
    ) -> float:
        """Calculate risk-adjusted reward."""
        # Profit component
        prev_pnl = env_state["simulator_state"]["pnl"]["net"]
        next_pnl = next_env_state["simulator_state"]["pnl"]["net"]
        pnl_change = next_pnl - prev_pnl

        # Drawdown component
        next_equity = next_env_state["simulator_state"]["equity"]
        self.max_equity = max(self.max_equity, next_equity)

        if self.max_equity > 0:
            drawdown = (self.max_equity - next_equity) / self.max_equity
        else:
            drawdown = 0.0

        # Position holding penalty
        position_qty = abs(next_env_state["simulator_state"]["position"]["quantity"])
        position_cost = position_qty * self.position_penalty

        # Combine components
        reward = (
            pnl_change * self.profit_weight - drawdown * self.drawdown_penalty - position_cost
        )

        return reward * self.scale

    def reset(self) -> None:
        """Reset max equity tracker."""
        self.max_equity = 0.0
