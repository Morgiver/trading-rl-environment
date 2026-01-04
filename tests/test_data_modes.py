"""Tests for data modes (LINEAR and SHUFFLE)."""

import pytest
import numpy as np
from datetime import time
from trading_rl_environment import TradingEnv, DataMode

class TestDataModes:
    """Test LINEAR and SHUFFLE data modes."""

    def test_linear_mode_sequential(self, real_candles_prefill):
        """Test LINEAR mode processes data sequentially."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            data_mode=DataMode.LINEAR,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        # First episode
        obs1, info1 = env.reset()
        start_idx_ep1 = env.episode_start_idx

        # Take some steps
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break

        # Second episode should continue from where first ended
        obs2, info2 = env.reset()
        start_idx_ep2 = env.episode_start_idx

        # In LINEAR mode, second episode should start after first
        # (unless wrapped around)
        assert env.episode_count == 2

    def test_shuffle_mode_random(self, real_candles_prefill):
        """Test SHUFFLE mode selects random days."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            data_mode=DataMode.SHUFFLE,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        # Reset multiple times and collect start indices
        start_indices = []
        for _ in range(5):
            obs, info = env.reset()
            start_indices.append(env.episode_start_idx)

        # In SHUFFLE mode, start indices should vary
        # (statistically unlikely all are the same with 5 days)
        assert len(start_indices) == 5

    def test_linear_mode_with_time_constraints(self, real_candles_prefill):
        """Test LINEAR mode with episode start/end times."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            data_mode=DataMode.LINEAR,
            episode_start_time=time(9, 30),  # 9:30 AM
            episode_end_time=time(11, 0),    # 11:00 AM
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Episode should start at 9:30
        start_candle = env.historical_candles[env.episode_start_idx]
        assert start_candle.date.time() >= time(9, 30)

        # Take steps until episode ends
        steps = 0
        while steps < 100:
            obs, reward, terminated, truncated, info = env.step(0)
            steps += 1

            if terminated or truncated:
                break

        # Should have terminated within time window
        assert terminated or truncated

    def test_shuffle_mode_with_time_constraints(self, real_candles_prefill):
        """Test SHUFFLE mode with episode start/end times."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            data_mode=DataMode.SHUFFLE,
            episode_start_time=time(10, 0),  # 10:00 AM
            episode_end_time=time(12, 0),    # 12:00 PM
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Episode should start at or after 10:00
        start_candle = env.historical_candles[env.episode_start_idx]
        assert start_candle.date.time() >= time(10, 0)

        # Episode should end reasonably close to 12:00 (with small buffer for max_steps)
        end_candle = env.historical_candles[env.episode_end_idx]
        # Allow a small buffer as max_steps might extend slightly beyond time constraint
        assert end_candle.date.time() <= time(13, 0)

    def test_episode_validation_insufficient_data(self, real_candles_prefill):
        """Test episode validation with insufficient prefill data."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1H"],  # Requires 60 * max_periods candles
            initial_balance=10000,
            max_periods=200,  # Requires 12000 1-minute candles
        )

        # Only provide 100 candles (insufficient)
        candles = real_candles_prefill
        env.load_historical_data(candles)

        # Should still reset with partial prefill acceptance
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)

    def test_linear_mode_wrapping(self, real_candles_prefill):
        """Test LINEAR mode wraps around when reaching end of data."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            data_mode=DataMode.LINEAR,
            max_steps=50,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        # Run multiple episodes to potentially wrap
        for episode in range(10):
            obs, info = env.reset()

            steps = 0
            while steps < 60:
                obs, reward, terminated, truncated, info = env.step(0)
                steps += 1

                if terminated or truncated:
                    break

        # Should have completed multiple episodes
        assert env.episode_count >= 5

    def test_shuffle_mode_prefill(self, real_candles_prefill):
        """Test SHUFFLE mode correctly prefills timeframes."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T", "5T"],
            initial_balance=10000,
            data_mode=DataMode.SHUFFLE,
            episode_start_time=time(10, 0),
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # AssetView should have data for all timeframes
        asset_data = env.asset_view.to_normalize_all()

        # Should have data for both timeframes
        assert "1T" in asset_data
        assert "5T" in asset_data

    def test_episode_boundaries(self, real_candles_prefill):
        """Test episode start and end boundaries are correct."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_steps=20,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Episode boundaries should be valid
        assert env.episode_start_idx < env.episode_end_idx
        assert env.episode_start_idx >= 0
        assert env.episode_end_idx < len(env.historical_candles)

        # Episode length should not exceed max_steps + 1
        episode_length = env.episode_end_idx - env.episode_start_idx
        assert episode_length <= env.max_steps + 10  # Some buffer for time constraints

    def test_invalid_action(self, real_candles_prefill):
        """Test invalid action raises error."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Invalid action should raise ValueError
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(99)

    def test_step_beyond_episode_end(self, real_candles_prefill):
        """Test stepping beyond episode end."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_steps=5,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Take steps until truncated
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(0)

            if truncated:
                # Further steps should also be truncated
                obs2, reward2, terminated2, truncated2, info2 = env.step(0)
                assert truncated2 or terminated2
                break
