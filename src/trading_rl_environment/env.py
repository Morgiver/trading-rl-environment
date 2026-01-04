"""Main Gymnasium environment for trading RL."""

from typing import Dict, List, Optional, Tuple, Any
from datetime import time as Time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from trading_frame import Candle, InsufficientDataError
from trading_asset_view import AssetView
from trading_simulator import TradingSimulator, OrderType, OrderSide, PnLMode

from .reward_strategy import RewardStrategy, SimplePnLReward
from .observation_builder import ObservationBuilder
from .enums import DataMode


class TradingEnv(gym.Env):
    """
    Gymnasium environment for trading with RL agents.

    Uses trading-frame, trading-asset-view, and trading-simulator for
    realistic multi-timeframe trading simulation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        symbol: str,
        timeframes: List[str],
        initial_balance: float = 10000.0,
        reward_strategy: Optional[RewardStrategy] = None,
        max_periods: int = 100,
        max_steps: int = 1000,
        trade_quantity: float = 1.0,
        pnl_mode: PnLMode = PnLMode.FIAT,
        fee_rate: float = 0.001,
        data_mode: DataMode = DataMode.LINEAR,
        episode_start_time: Optional[Time] = None,
        episode_end_time: Optional[Time] = None,
        **simulator_kwargs,
    ):
        """
        Initialize TradingEnv.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframes: List of timeframe strings (e.g., ["1T", "5T", "1H"])
            initial_balance: Starting balance for simulator
            reward_strategy: Custom reward strategy (defaults to SimplePnLReward)
            max_periods: Maximum periods to keep per timeframe
            max_steps: Maximum steps per episode
            trade_quantity: Fixed quantity to trade per action
            pnl_mode: PnL calculation mode
            fee_rate: Trading fee rate (e.g., 0.001 = 0.1%)
            data_mode: LINEAR (continuous traversal) or SHUFFLE (random days)
            episode_start_time: Time of day to start episodes (e.g., time(9, 30))
            episode_end_time: Time of day to end episodes (e.g., time(16, 0))
            **simulator_kwargs: Additional kwargs for TradingSimulator
        """
        super().__init__()

        self.symbol = symbol
        self.timeframes = timeframes
        self.initial_balance = initial_balance
        self.max_periods = max_periods
        self.max_steps = max_steps
        self.trade_quantity = trade_quantity
        self.data_mode = data_mode
        self.episode_start_time = episode_start_time
        self.episode_end_time = episode_end_time

        # Initialize reward strategy
        self.reward_strategy = reward_strategy or SimplePnLReward()

        # Initialize AssetView for multi-timeframe data
        self.asset_view = AssetView(
            symbol=symbol, timeframes=timeframes, max_periods=max_periods
        )

        # Initialize TradingSimulator
        self.simulator = TradingSimulator(
            initial_balance=initial_balance,
            pnl_mode=pnl_mode,
            fee_rate=fee_rate,
            **simulator_kwargs,
        )

        # Historical data
        self.historical_candles: List[Candle] = []
        self.current_candle_idx = 0  # Current position in dataset
        self.episode_start_idx = 0   # Start index for current episode
        self.episode_end_idx = 0     # End index for current episode

        # Observation builder
        # Use trade_quantity as max_position_size for normalization (agent can accumulate positions)
        # For more conservative normalization, multiply by expected max accumulation factor
        self.observation_builder = ObservationBuilder(
            timeframes=timeframes,
            max_periods=max_periods,
            initial_balance=initial_balance,
            max_position_size=trade_quantity * 10  # Allow up to 10x accumulation
        )

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation space (will be built after first data load)
        self.observation_space = None

        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0

        # State for reward calculation
        self._last_env_state = None

    def load_historical_data(self, candles: List[Candle]) -> None:
        """
        Load historical candle data for training/backtesting.

        Args:
            candles: List of Candle instances (should be sorted by date)
        """
        self.historical_candles = candles

        # Build observation space based on data dimensions
        if candles:
            # Feed a few candles to determine feature count
            temp_view = AssetView(
                symbol=self.symbol, timeframes=self.timeframes, max_periods=self.max_periods
            )
            for i, candle in enumerate(candles[: min(10, len(candles))]):
                temp_view.feed(candle)

            # Get normalized data to determine feature count
            normalized = temp_view.to_normalize_all()
            if normalized and self.timeframes[0] in normalized:
                n_features = normalized[self.timeframes[0]].shape[1]
                self.observation_space = self.observation_builder.build_observation_space(
                    n_features
                )

    def load_from_dataframe(self, df, **kwargs) -> None:
        """
        Load historical data from pandas DataFrame.

        Args:
            df: pandas DataFrame with OHLCV data
            **kwargs: Additional arguments for AssetView.load_from_dataframe()
        """
        # Create temporary AssetView to convert DataFrame to Candles
        temp_view = AssetView(
            symbol=self.symbol, timeframes=self.timeframes, max_periods=self.max_periods
        )
        temp_view.load_from_dataframe(df, **kwargs)

        # Extract candles from the shortest timeframe
        shortest_tf = self.timeframes[0]  # Assuming first is shortest
        frame = temp_view.get_frame(shortest_tf)

        # Reconstruct candles from periods
        candles = []
        for period in frame.periods:
            candle = Candle(
                date=period.open_date,
                open=period.open_price,
                high=period.high_price,
                low=period.low_price,
                close=period.close_price,
                volume=float(period.volume),
            )
            candles.append(candle)

        self.load_historical_data(candles)

    def add_indicator(self, timeframe: str, indicator, column_name) -> None:
        """
        Add indicator to a specific timeframe.

        Args:
            timeframe: Timeframe to add indicator to
            indicator: Indicator instance
            column_name: Column name(s) for the indicator
        """
        self.asset_view.add_indicator(timeframe, indicator, column_name)

        # Rebuild observation space
        if self.historical_candles:
            self.load_historical_data(self.historical_candles)

    def add_indicator_to_all(self, indicator, column_name) -> None:
        """
        Add indicator to all timeframes.

        Args:
            indicator: Indicator instance
            column_name: Column name(s) for the indicator
        """
        self.asset_view.add_indicator_to_all(indicator, column_name)

        # Rebuild observation space
        if self.historical_candles:
            self.load_historical_data(self.historical_candles)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Behavior depends on data_mode:
        - LINEAR: Continue from where last episode ended (preserves timeframes)
        - SHUFFLE: Start from random day (rebuilds timeframes)

        Validates that there is enough data to prefill timeframes before starting episode.
        If not, retries reset without incrementing episode count.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset simulator and reward strategy BEFORE prefill to ensure clean state
        self.simulator.reset()
        self.reward_strategy.reset()

        max_retries = 100  # Prevent infinite loop
        for attempt in range(max_retries):
            # Determine episode start and end indices
            if self.data_mode == DataMode.LINEAR:
                self._reset_linear_mode()
            else:  # SHUFFLE
                self._reset_shuffle_mode()

            # Validate episode has enough data
            if self._validate_episode():
                break  # Valid episode found
            else:
                # Not enough data, retry (don't increment episode_count yet)
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to find valid episode after {max_retries} attempts. "
                        f"Dataset might be too small or constraints too restrictive."
                    )

        # Feed the first candle of the episode to initialize market price
        # This ensures simulator has a valid last_price for observation
        if self.current_candle_idx < len(self.historical_candles):
            first_candle = self.historical_candles[self.current_candle_idx]
            self.asset_view.feed(first_candle)
            self.simulator.update_market(first_candle)
            # Increment so first step() uses next candle
            self.current_candle_idx += 1

        # Reset episode state (only increment episode_count on successful reset)
        self.episode_count += 1
        self._last_env_state = None

        # Get initial observation
        observation = self._get_observation()

        # Store initial state for reward calculation
        self._last_env_state = self._get_env_state()

        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=HOLD, 1=BUY, 2=SELL)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}. Must be 0 (HOLD), 1 (BUY), or 2 (SELL)")

        # Check if we reached episode end
        if self.current_candle_idx >= self.episode_end_idx:
            # Episode ends - reached end boundary
            observation = self._get_observation()
            reward = 0.0
            terminated = False
            truncated = True  # Episode truncated by time boundary
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        # Check if we have more data (safety check)
        if self.current_candle_idx >= len(self.historical_candles):
            observation = self._get_observation()
            reward = 0.0
            terminated = True
            truncated = False
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        # Get next candle
        candle = self.historical_candles[self.current_candle_idx]

        # Update market data FIRST (so simulator has valid price for trading)
        self.asset_view.feed(candle)
        self.simulator.update_market(candle)

        # Execute action AFTER market update
        # Note: Action is executed on the candle that was just fed to the observation
        # This creates realistic timing where the agent sees data then acts on it
        # (no look-ahead bias - agent cannot trade before seeing the price)
        self._execute_action(action)

        # Increment indices
        self.current_candle_idx += 1
        self.total_steps += 1

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        next_env_state = self._get_env_state()
        reward = self.reward_strategy.calculate(self._last_env_state, action, next_env_state)
        self._last_env_state = next_env_state

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = (self.current_candle_idx >= self.episode_end_idx)

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> None:
        """Execute the trading action."""
        if action == 0:
            # HOLD - do nothing
            return

        elif action == 1:
            # BUY
            try:
                self.simulator.place_order(
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=self.trade_quantity,
                )
            except RuntimeError:
                # Insufficient balance or other error - skip
                pass

        elif action == 2:
            # SELL
            position = self.simulator.get_position()

            if position.is_long:
                # Close long position
                try:
                    self.simulator.place_order(
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        quantity=min(position.quantity, self.trade_quantity),
                    )
                except RuntimeError:
                    pass
            elif position.is_flat:
                # Open short position
                try:
                    self.simulator.place_order(
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        quantity=self.trade_quantity,
                    )
                except RuntimeError:
                    pass

    def _calculate_required_prefill_candles(self) -> int:
        """
        Calculate minimum number of candles required to prefill timeframes.

        Returns the number of base timeframe candles needed to have at least
        one complete period in the longest timeframe.

        Returns:
            Minimum number of candles required for prefill
        """
        # Parse timeframe strings to get multipliers
        # Format: "1T" (1 minute), "5T" (5 minutes), "1H" (1 hour), "1D" (1 day)
        max_multiplier = 0

        for tf in self.timeframes:
            # Extract number and unit
            import re

            match = re.match(r"(\d+)([THDM])", tf)
            if match:
                num = int(match.group(1))
                unit = match.group(2)

                # Convert to minutes (assuming base is 1 minute)
                if unit == "T":  # Minutes
                    multiplier = num
                elif unit == "H":  # Hours
                    multiplier = num * 60
                elif unit == "D":  # Days
                    multiplier = num * 60 * 24
                else:
                    multiplier = num

                max_multiplier = max(max_multiplier, multiplier)

        # Need enough base candles to fill max_periods of the longest timeframe
        # For example: if longest timeframe is 1H (60 minutes) and max_periods=100
        # we need 60 * 100 = 6000 base (1-minute) candles
        required = max_multiplier * self.max_periods if max_multiplier > 0 else self.max_periods
        return required

    def _validate_episode(self) -> bool:
        """
        Validate that episode has enough data for prefilling timeframes.

        Returns:
            True if episode is valid, False otherwise
        """
        # Check that episode has at least one step
        if self.episode_start_idx >= self.episode_end_idx:
            # In LINEAR mode, if we reached end of data, wrap around
            if self.data_mode == DataMode.LINEAR:
                # Allow wrapping - reset will handle it
                return True
            return False

        # Calculate required prefill
        required_prefill = self._calculate_required_prefill_candles()

        # Check if we have enough data before episode_start_idx
        available_prefill = self.episode_start_idx

        # Need at least required_prefill candles OR all available if less
        if available_prefill < required_prefill:
            # Not enough prefill data - check if this is acceptable
            # If we're at the very beginning of the dataset, accept what we have
            if self.episode_start_idx == 0:
                # First candle of dataset - can't have prefill, but acceptable
                return True

            # With episode_start_time, we might start mid-day
            # Check if we have reasonable prefill (at least 10% of requirement)
            if self.episode_start_time is not None:
                # Accept if we have at least 10% of required prefill or 50 candles minimum
                # This allows starting mid-day but still ensures some timeframe data
                min_acceptable = max(required_prefill * 0.1, 50)
                if available_prefill >= min_acceptable:
                    return True
                # Not enough prefill even for time-constrained episodes
                return False

            # For small datasets, accept what we have (at least 10% of required or 10 candles)
            if available_prefill >= min(required_prefill * 0.1, 10):
                return True

            # Without time constraints and insufficient prefill, retry
            return False

        return True

    def _find_episode_boundaries(self, start_idx: int) -> Tuple[int, int]:
        """
        Find start and end indices for an episode based on time constraints.

        Args:
            start_idx: Candidate start index in historical_candles

        Returns:
            Tuple of (episode_start_idx, episode_end_idx)
        """
        # If no time constraints, use max_steps
        if self.episode_start_time is None and self.episode_end_time is None:
            end_idx = min(start_idx + self.max_steps, len(self.historical_candles) - 1)
            return start_idx, end_idx

        # Find first candle matching start_time
        actual_start_idx = start_idx
        if self.episode_start_time is not None:
            for i in range(start_idx, len(self.historical_candles)):
                candle_time = self.historical_candles[i].date.time()
                if candle_time >= self.episode_start_time:
                    actual_start_idx = i
                    break

        # Find last candle before end_time or max_steps
        actual_end_idx = min(actual_start_idx + self.max_steps, len(self.historical_candles) - 1)
        if self.episode_end_time is not None:
            for i in range(actual_start_idx, len(self.historical_candles)):
                candle_time = self.historical_candles[i].date.time()
                if candle_time >= self.episode_end_time:
                    actual_end_idx = i
                    break
                # Also respect max_steps
                if i - actual_start_idx >= self.max_steps:
                    actual_end_idx = i
                    break

        return actual_start_idx, actual_end_idx

    def _reset_linear_mode(self) -> None:
        """
        Reset in LINEAR mode: continue from where last episode ended.

        Process:
        1. First episode: Find first valid day with enough prefill data
        2. Subsequent episodes: Continue feeding until next episode_start_time
        3. Preserves AssetView state across episodes (no rebuild)
        """
        # First episode: find first valid day
        if self.episode_count == 0:
            # Calculate required prefill
            required_prefill = self._calculate_required_prefill_candles()

            # Find first day that has enough data for prefill + episode
            if self.episode_start_time is not None:
                # Find all days
                days_map = {}
                for i, candle in enumerate(self.historical_candles):
                    day = candle.date.date()
                    if day not in days_map:
                        days_map[day] = i

                # Find first valid day
                valid_day_found = False
                for day, day_start_idx in sorted(days_map.items()):
                    # Check if we have enough prefill before this day
                    if day_start_idx >= required_prefill:
                        # Find episode_start_time on this day
                        for i in range(day_start_idx, len(self.historical_candles)):
                            candle = self.historical_candles[i]
                            if candle.date.date() != day:
                                break
                            if candle.date.time() >= self.episode_start_time:
                                self.episode_start_idx = i
                                valid_day_found = True
                                break
                    if valid_day_found:
                        break

                if not valid_day_found:
                    # Fallback: start at first occurrence of episode_start_time
                    # (even without full prefill)
                    for i, candle in enumerate(self.historical_candles):
                        if candle.date.time() >= self.episode_start_time:
                            self.episode_start_idx = i
                            break
                    else:
                        # No matching time found - start at beginning
                        self.episode_start_idx = min(required_prefill, len(self.historical_candles) // 2)
            else:
                # No time constraint: start after prefill requirement
                # Use the full required_prefill to ensure all timeframes have max_periods candles
                if required_prefill >= len(self.historical_candles):
                    raise ValueError(
                        f"Insufficient data: need at least {required_prefill + 1} candles "
                        f"(prefill + 1 for episode) but only have {len(self.historical_candles)}. "
                        f"Increase dataset size or reduce max_periods."
                    )
                self.episode_start_idx = required_prefill

            # Find episode end
            _, self.episode_end_idx = self._find_episode_boundaries(self.episode_start_idx)

            # Rebuild timeframes and prefill from beginning
            self.asset_view = AssetView(
                symbol=self.symbol, timeframes=self.timeframes, max_periods=self.max_periods
            )

            # Prefill all timeframes up to episode start using prefill()
            # This ensures ALL timeframes have max_periods before starting episode
            target_candle = self.historical_candles[self.episode_start_idx]
            target_timestamp = target_candle.date.timestamp()

            try:
                for i in range(len(self.historical_candles)):
                    candle = self.historical_candles[i]
                    self.simulator.update_market(candle)

                    if self.asset_view.prefill(candle, target_timestamp=target_timestamp, require_full=True):
                        # All timeframes are full at target timestamp
                        break
            except InsufficientDataError as e:
                # Not enough data to fill all timeframes before episode start
                raise RuntimeError(
                    f"Insufficient historical data to prefill timeframes: {e}. "
                    f"Required {self._calculate_required_prefill_candles()} candles for prefill, "
                    f"but dataset only has {len(self.historical_candles)} candles total. "
                    f"Increase dataset size or reduce max_periods."
                )

            self.current_candle_idx = self.episode_start_idx

        else:
            # Subsequent episodes: continue feeding until next episode_start_time
            # AssetView already has state - just continue feeding

            # Start from where we left off
            fill_idx = self.current_candle_idx

            # If reached end of data, wrap around
            if fill_idx >= len(self.historical_candles):
                fill_idx = 0
                # Rebuild timeframes when wrapping
                self.asset_view = AssetView(
                    symbol=self.symbol, timeframes=self.timeframes, max_periods=self.max_periods
                )

            # Find next episode_start_time
            if self.episode_start_time is not None:
                # Feed candles until we reach next episode_start_time
                episode_found = False
                for i in range(fill_idx, len(self.historical_candles)):
                    candle = self.historical_candles[i]

                    # Check if this is episode start time
                    if candle.date.time() >= self.episode_start_time:
                        self.episode_start_idx = i
                        episode_found = True
                        break
                    else:
                        # Not episode start yet - feed to continue building timeframes
                        self.asset_view.feed(candle)
                        self.simulator.update_market(candle)

                if not episode_found:
                    # Reached end without finding start_time - wrap around
                    self.episode_start_idx = fill_idx
            else:
                # No time constraint: start immediately
                self.episode_start_idx = fill_idx

            # Find episode end
            _, self.episode_end_idx = self._find_episode_boundaries(self.episode_start_idx)
            self.current_candle_idx = self.episode_start_idx

    def _reset_shuffle_mode(self) -> None:
        """
        Reset in SHUFFLE mode: start from random day at episode_start_time.

        Process:
        1. Choose random day from dataset
        2. Find episode_start_time on that day (or start of day if not specified)
        3. Prefill timeframes with candles from start of day (00:00) to episode_start_time
        4. Start episode at episode_start_time
        """
        # Find all unique days in dataset
        days_map = {}  # date -> first_candle_idx of that day
        for i, candle in enumerate(self.historical_candles):
            day = candle.date.date()
            if day not in days_map:
                days_map[day] = i

        available_days = list(days_map.keys())

        if not available_days:
            raise ValueError("No days available in historical data")

        # Choose random day
        random_day = random.choice(available_days)
        day_start_idx = days_map[random_day]

        # Find episode start at episode_start_time on that day
        if self.episode_start_time is not None:
            # Find first candle >= episode_start_time on random_day
            episode_start_found = False
            for i in range(day_start_idx, len(self.historical_candles)):
                candle = self.historical_candles[i]

                # Still on same day?
                if candle.date.date() != random_day:
                    break

                # Found start time?
                if candle.date.time() >= self.episode_start_time:
                    self.episode_start_idx = i
                    episode_start_found = True
                    break

            if not episode_start_found:
                # Fallback: start at beginning of day
                self.episode_start_idx = day_start_idx
        else:
            # No start time specified: start at beginning of day (00:00)
            self.episode_start_idx = day_start_idx

        # Find episode end
        _, self.episode_end_idx = self._find_episode_boundaries(self.episode_start_idx)

        # Rebuild AssetView
        self.asset_view = AssetView(
            symbol=self.symbol, timeframes=self.timeframes, max_periods=self.max_periods
        )

        # Prefill using prefill() method to ensure ALL timeframes are full
        target_candle = self.historical_candles[self.episode_start_idx]
        target_timestamp = target_candle.date.timestamp()

        try:
            # Start from beginning of dataset (or earlier if needed for prefill)
            # The prefill() method will stop when all timeframes are full at target timestamp
            for i in range(len(self.historical_candles)):
                candle = self.historical_candles[i]
                self.simulator.update_market(candle)

                if self.asset_view.prefill(candle, target_timestamp=target_timestamp, require_full=True):
                    # All timeframes are full at target timestamp
                    break
        except InsufficientDataError as e:
            # Not enough data to prefill all timeframes before episode_start_time on this day
            # In SHUFFLE mode, this will be retried with a different random day by the outer retry loop
            raise RuntimeError(
                f"Insufficient data to prefill timeframes for day {random_day}: {e}. "
                f"Not enough historical data before episode start time on this day."
            )

        self.current_candle_idx = self.episode_start_idx

    def _get_observation(self) -> np.ndarray:
        """Build current observation."""
        # Get normalized multi-timeframe data
        normalized_data = self.asset_view.to_normalize_all()

        # Get simulator state
        simulator_state = self.simulator.get_state()

        # Build observation
        observation = self.observation_builder.build_observation(normalized_data, simulator_state)

        return observation

    def _get_env_state(self) -> Dict[str, Any]:
        """Get complete environment state for reward calculation."""
        return {
            "simulator_state": self.simulator.get_state(),
            "asset_view_stats": self.asset_view.get_statistics(),
            "current_candle_idx": self.current_candle_idx,
        }

    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if balance drops too low (e.g., 10% of initial)
        if self.simulator.balance < self.initial_balance * 0.1:
            return True

        # Terminate if equity drops too low
        equity = self.simulator.balance + self.simulator.get_pnl()["net"]
        if equity < self.initial_balance * 0.1:
            return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        pnl = self.simulator.get_pnl()
        position = self.simulator.get_position()

        return {
            "candle_idx": self.current_candle_idx,
            "episode": self.episode_count,
            "episode_start_idx": self.episode_start_idx,
            "episode_end_idx": self.episode_end_idx,
            "data_mode": self.data_mode.name,
            "balance": self.simulator.balance,
            "equity": self.simulator.balance + pnl["net"],
            "pnl": pnl,
            "position": {
                "quantity": position.quantity,
                "average_price": position.average_price,
                "is_flat": position.is_flat,
                "is_long": position.is_long,
                "is_short": position.is_short,
            },
            "last_price": self.simulator.last_price,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive environment statistics.

        Returns:
            Dictionary with episode stats, simulator state, and asset view stats
        """
        return {
            "episode_stats": {
                "episode": self.episode_count,
                "total_steps": self.total_steps,
                "current_candle_idx": self.current_candle_idx,
                "episode_start_idx": self.episode_start_idx,
                "episode_end_idx": self.episode_end_idx,
            },
            "simulator_state": self.simulator.get_state(),
            "asset_view_stats": self.asset_view.get_statistics(),
        }

    def render(self) -> None:
        """Render environment (not implemented)."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass
