"""Observation space builder for RL environment."""

from typing import Dict, List, Tuple
import numpy as np
from gymnasium import spaces


class ObservationBuilder:
    """
    Builds observation space and observations for the RL environment.

    Combines normalized multi-timeframe data with position/account information.
    """

    def __init__(self, timeframes: List[str], max_periods: int = 100, initial_balance: float = 10000.0, max_position_size: float = 10.0):
        """
        Initialize observation builder.

        Args:
            timeframes: List of timeframe strings (e.g., ["1T", "5T", "1H"])
            max_periods: Maximum periods per timeframe
            initial_balance: Initial account balance for normalization
            max_position_size: Maximum expected position size for normalization (in units)
        """
        self.timeframes = timeframes
        self.max_periods = max_periods
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size

        # Will be set after first observation build
        self._observation_space = None
        self._n_features_per_timeframe = None

    def build_observation_space(self, n_features_per_timeframe: int) -> spaces.Box:
        """
        Build the Gymnasium observation space.

        Args:
            n_features_per_timeframe: Number of features (OHLCV + indicators) per timeframe

        Returns:
            Gymnasium Box space for observations
        """
        self._n_features_per_timeframe = n_features_per_timeframe

        # Calculate total observation size
        # Each timeframe contributes: n_features * max_periods
        timeframe_features = len(self.timeframes) * n_features_per_timeframe * self.max_periods

        # Position features: quantity, avg_price, unrealized_pnl, is_long, is_short, is_flat
        position_features = 6

        # Account features: balance, equity, realized_pnl
        account_features = 3

        total_features = timeframe_features + position_features + account_features

        # Build feature-specific bounds
        # Most features normalized to [-1, 1], but balance/equity use log scale [0, ~3]
        low_bounds = np.full(total_features, -1.0, dtype=np.float32)
        high_bounds = np.full(total_features, 1.0, dtype=np.float32)

        # Adjust bounds for balance/equity features (last 3 account features)
        # Position: timeframe_features (0 to N-1)
        # Account features: N to N+2 (balance, equity, realized_pnl)
        balance_idx = timeframe_features + position_features  # balance
        equity_idx = balance_idx + 1  # equity

        # log1p ranges: 0 to ~3 for reasonable growth (up to 20x initial balance)
        low_bounds[balance_idx] = 0.0
        high_bounds[balance_idx] = 5.0  # log1p(148) ≈ 5, allows 148x growth
        low_bounds[equity_idx] = 0.0
        high_bounds[equity_idx] = 5.0

        self._observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, shape=(total_features,), dtype=np.float32
        )

        return self._observation_space

    @property
    def observation_space(self) -> spaces.Box:
        """Get the observation space."""
        if self._observation_space is None:
            raise RuntimeError(
                "Observation space not built yet. Call build_observation_space() first."
            )
        return self._observation_space

    def build_observation(
        self, normalized_data: Dict[str, np.ndarray], simulator_state: Dict
    ) -> np.ndarray:
        """
        Build observation from normalized timeframe data and simulator state.

        Args:
            normalized_data: Dict of normalized arrays from asset_view.to_normalize_all()
                            Format: {'1T': array(...), '5T': array(...), ...}
            simulator_state: State dict from simulator.get_state()

        Returns:
            Flat numpy array representing the observation
        """
        observation_parts = []

        # 1. Normalized multi-timeframe data
        for tf in self.timeframes:
            if tf in normalized_data and len(normalized_data[tf]) > 0:
                data = normalized_data[tf]

                # Pad or truncate to max_periods
                if len(data) < self.max_periods:
                    # Pad with zeros at the beginning (no data yet)
                    padding = np.zeros(
                        (self.max_periods - len(data), data.shape[1]), dtype=np.float32
                    )
                    data = np.vstack([padding, data])
                elif len(data) > self.max_periods:
                    # Take last max_periods
                    data = data[-self.max_periods :]

                # Flatten timeframe data
                observation_parts.append(data.flatten())
            else:
                # No data for this timeframe yet - use zeros
                n_features = (
                    self._n_features_per_timeframe
                    if self._n_features_per_timeframe
                    else 5  # OHLCV default
                )
                observation_parts.append(
                    np.zeros(self.max_periods * n_features, dtype=np.float32)
                )

        # 2. Position features (normalized)
        position = simulator_state["position"]

        # Normalize quantity to [-1, 1] range based on max_position_size
        normalized_quantity = np.clip(position["quantity"] / self.max_position_size, -1.0, 1.0)

        # Normalize average price (use last price as reference)
        last_price = simulator_state.get("last_price")
        if last_price is not None and last_price > 0 and position["average_price"] > 0:
            normalized_avg_price = (position["average_price"] - last_price) / last_price
            normalized_avg_price = np.clip(normalized_avg_price, -1.0, 1.0)
        else:
            normalized_avg_price = 0.0

        # Unrealized PnL normalized by balance
        balance = simulator_state["balance"]
        if balance > 0:
            pnl_ratio = simulator_state["pnl"]["unrealized"] / balance
            normalized_unrealized_pnl = np.clip(pnl_ratio, -1.0, 1.0)
        else:
            normalized_unrealized_pnl = 0.0

        # Binary position indicators
        is_long = 1.0 if position["quantity"] > 0 else 0.0
        is_short = 1.0 if position["quantity"] < 0 else 0.0
        is_flat = 1.0 if position["is_flat"] else 0.0

        position_features = np.array(
            [
                normalized_quantity,
                normalized_avg_price,
                normalized_unrealized_pnl,
                is_long,
                is_short,
                is_flat,
            ],
            dtype=np.float32,
        )
        observation_parts.append(position_features)

        # 3. Account features
        balance_ratio = balance / self.initial_balance if self.initial_balance > 0 else 0.0
        equity_ratio = simulator_state["equity"] / self.initial_balance if self.initial_balance > 0 else 0.0

        realized_pnl = simulator_state["pnl"]["realized"]
        realized_pnl_ratio = realized_pnl / self.initial_balance if self.initial_balance > 0 else 0.0

        # Use log1p for balance/equity to handle unlimited growth without saturation
        # log1p(x) = log(1 + x) allows smooth scaling for both losses and unlimited gains
        # Range: log1p(0) = 0, log1p(1) ≈ 0.69, log1p(4) ≈ 1.6, log1p(9) ≈ 2.3
        balance_feature = np.log1p(max(0.0, balance_ratio))
        equity_feature = np.log1p(max(0.0, equity_ratio))

        account_features = np.array(
            [
                balance_feature,  # Unbounded growth via log scale
                equity_feature,   # Unbounded growth via log scale
                np.clip(realized_pnl_ratio, -1.0, 1.0),  # Realized PnL still clipped
            ],
            dtype=np.float32,
        )
        observation_parts.append(account_features)

        # Concatenate all parts
        observation = np.concatenate(observation_parts)

        return observation.astype(np.float32)

    def get_observation_info(self) -> Dict[str, any]:
        """
        Get information about the observation structure.

        Returns:
            Dictionary with observation structure details
        """
        if self._observation_space is None:
            return {"error": "Observation space not built yet"}

        info = {
            "total_features": self._observation_space.shape[0],
            "timeframes": self.timeframes,
            "max_periods": self.max_periods,
            "features_per_timeframe": self._n_features_per_timeframe,
            "timeframe_features_total": len(self.timeframes)
            * self._n_features_per_timeframe
            * self.max_periods,
            "position_features": 6,
            "account_features": 3,
        }

        return info
