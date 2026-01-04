"""Tests for ObservationBuilder."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from trading_rl_environment import TradingEnv
from trading_frame import Candle
from trading_frame.indicators import RSI, SMA

class TestObservationBuilder:
    """Test ObservationBuilder functionality."""

    def test_observation_shape(self, real_candles_prefill):
        """Test observation shape is correct."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T", "5T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Observation should be 1D array
        assert len(obs.shape) == 1
        assert obs.shape[0] > 0

    def test_observation_with_indicators(self, real_candles_prefill):
        """Test observation includes indicator data."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T", "5T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        # Add indicators
        env.add_indicator("1T", RSI(length=14), "RSI_14")
        env.add_indicator("5T", SMA(period=20), "SMA_20")

        obs, info = env.reset()

        # Observation size should increase with indicators
        assert obs.shape[0] > 0
        assert not np.any(np.isnan(obs))  # Should handle NaN values

    def test_observation_with_position(self, real_candles_prefill):
        """Test observation includes position information."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            trade_quantity=0.1,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        # Reset and take BUY action
        obs1, info1 = env.reset()

        # BUY to create position
        obs2, reward, terminated, truncated, info2 = env.step(1)

        # Observation should reflect position
        assert not np.array_equal(obs1, obs2)
        assert not info2["position"]["is_flat"]

    def test_observation_normalization(self, real_candles_prefill):
        """Test observation values are normalized."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T", "5T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Most values should be in reasonable range after normalization
        # (some may be outside [-1, 1] but not extreme)
        assert np.all(np.abs(obs) < 100)  # Reasonable bound

    def test_observation_without_position(self, real_candles_prefill):
        """Test observation when flat (no position)."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Should have valid observation even without position
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert not np.any(np.isnan(obs))

    def test_observation_multiple_timeframes(self, real_candles_prefill):
        """Test observation with multiple timeframes."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T", "5T", "15T", "1H"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Should have data from all timeframes
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] > 0

    def test_observation_with_empty_timeframe(self, real_candles_prefill):
        """Test observation when some timeframes might not have full data."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T", "1H"],  # 1H might not have much data initially
            initial_balance=10000,
            max_periods=5,  # Small max_periods
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Should handle partial data gracefully
        assert isinstance(obs, np.ndarray)
        # NaN should be replaced with 0
        assert not np.any(np.isnan(obs))

    def test_observation_consistency(self, real_candles_prefill):
        """Test observation shape stays consistent across steps."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T", "5T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()
        initial_shape = obs.shape

        # Take multiple steps
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)

            # Shape should remain consistent
            assert obs.shape == initial_shape

            if terminated or truncated:
                break

    def test_observation_with_high_position(self, real_candles_prefill):
        """Test observation with large position relative to balance."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            trade_quantity=1.0,  # Large quantity
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # BUY large position
        obs, reward, terminated, truncated, info = env.step(1)

        # Should handle large position values
        assert isinstance(obs, np.ndarray)
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_rebuild_observation_space(self, real_candles_prefill):
        """Test rebuilding observation space when indicators change."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs1, info1 = env.reset()
        shape1 = obs1.shape

        # Add indicator
        env.add_indicator("1T", RSI(length=14), "RSI_14")

        obs2, info2 = env.reset()
        shape2 = obs2.shape

        # Shape should change
        assert shape2[0] > shape1[0]

    def test_observation_all_zeros(self, real_candles_prefill):
        """Test observation handles edge case of all zero values."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=100,
        )

        # Generate flat price candles
        candles = []
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        for i in range(200):
            candle = Candle(
                date=base_time + timedelta(minutes=i),
                open=50000.0,
                high=50000.0,
                low=50000.0,
                close=50000.0,
                volume=100.0,
            )
            candles.append(candle)

        env.load_historical_data(candles)
        obs, info = env.reset()

        # Should handle flat prices gracefully
        assert isinstance(obs, np.ndarray)
        assert not np.any(np.isnan(obs))
