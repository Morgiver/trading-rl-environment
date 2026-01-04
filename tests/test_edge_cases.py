"""Tests for edge cases and error handling."""

import pytest
import numpy as np
from trading_rl_environment import TradingEnv, DataMode

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_load_data_rebuilds_observation_space(self, real_candles_prefill):
        """Test loading data rebuilds observation space."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=100,
        )

        # Initially no observation space
        assert env.observation_space is None

        candles = real_candles_prefill
        env.load_historical_data(candles)

        # Should have observation space after loading
        assert env.observation_space is not None

    def test_episode_end_at_data_boundary(self, real_candles_prefill):
        """Test episode ending exactly at data boundary."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_steps=40,
            max_periods=100,
        )

        # Small dataset
        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Step until we hit the boundary
        for _ in range(60):
            obs, reward, terminated, truncated, info = env.step(0)

            if terminated or truncated:
                break

        # Should have terminated or truncated
        assert terminated or truncated

    def test_very_small_dataset(self, real_candles_prefill):
        """Test with very small dataset (edge case)."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=5,  # Very small
        )

        # Minimal dataset
        candles = real_candles_prefill
        env.load_historical_data(candles)

        # Should still reset successfully
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)

        # Should be able to take at least one step
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)

    def test_multiple_consecutive_buys(self, real_candles_prefill):
        """Test multiple consecutive buy actions."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            trade_quantity=0.1,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Multiple consecutive buys
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(1)  # BUY

            if terminated or truncated:
                break

        # Position should have accumulated
        assert info["position"]["quantity"] > 0

    def test_multiple_consecutive_sells(self, real_candles_prefill):
        """Test multiple consecutive sell actions."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            trade_quantity=0.1,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # BUY first
        obs, reward, _, _, info = env.step(1)

        # Multiple consecutive sells
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(2)  # SELL

            if terminated or truncated:
                break

        # Position might be negative (short) or flat
        assert isinstance(info["position"]["quantity"], (int, float))

    def test_episode_count_increments(self, real_candles_prefill):
        """Test episode count increments correctly."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        # First episode
        obs, info = env.reset()
        assert env.episode_count == 1

        # Second episode
        obs, info = env.reset()
        assert env.episode_count == 2

        # Third episode
        obs, info = env.reset()
        assert env.episode_count == 3

    def test_total_steps_increments(self, real_candles_prefill):
        """Test total steps counter increments."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()
        initial_steps = env.total_steps

        # Take steps
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(0)

            if terminated or truncated:
                break

        # Total steps should have increased
        assert env.total_steps > initial_steps

    def test_close_method(self, real_candles_prefill):
        """Test close method exists and works."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Close should work without error
        env.close()

    def test_render_method(self, real_candles_prefill):
        """Test render method exists."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Render should not raise error
        env.render()

    def test_statistics_format(self, real_candles_prefill):
        """Test statistics are in correct format."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        stats = env.get_statistics()

        # Check required keys
        assert "episode_stats" in stats
        assert "simulator_state" in stats
        assert "asset_view_stats" in stats

        # Check nested structure
        assert "episode" in stats["episode_stats"]
        assert "total_steps" in stats["episode_stats"]

    def test_get_env_state(self, real_candles_prefill):
        """Test _get_env_state returns complete state."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # _get_env_state should work
        state = env._get_env_state()

        assert "asset_view_stats" in state
        assert "simulator_state" in state
        assert isinstance(state["asset_view_stats"], dict)
        assert isinstance(state["simulator_state"], dict)

    def test_no_time_constraints(self, real_candles_prefill):
        """Test episode without time constraints uses full data."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_steps=50,
            episode_start_time=None,  # No constraints
            episode_end_time=None,
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        obs, info = env.reset()

        # Should be able to take max_steps
        steps = 0
        for _ in range(60):
            obs, reward, terminated, truncated, info = env.step(0)
            steps += 1

            if terminated or truncated:
                break

        # Should have taken close to max_steps
        assert steps > 0
