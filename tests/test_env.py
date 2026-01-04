"""Tests for TradingEnv."""

import pytest
import numpy as np
from trading_rl_environment import TradingEnv, SimplePnLReward
from trading_frame.indicators import RSI, SMA


class TestTradingEnv:
    """Test TradingEnv class."""

    def test_init(self):
        """Test environment initialization."""
        env = TradingEnv(
            symbol="BTC/USDT",
            timeframes=["1T", "5T"],
            initial_balance=10000,
        )

        assert env.symbol == "BTC/USDT"
        assert env.timeframes == ["1T", "5T"]
        assert env.initial_balance == 10000
        assert env.action_space.n == 3  # HOLD, BUY, SELL

    def test_load_historical_data(self, real_candles_small):
        """Test loading historical data."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
        )

        env.load_historical_data(real_candles_small)

        assert len(env.historical_candles) == len(real_candles_small)
        assert env.observation_space is not None

    def test_reset(self, real_candles_prefill):
        """Test environment reset."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_periods=100,  # Smaller for faster tests
        )

        env.load_historical_data(real_candles_prefill)

        observation, info = env.reset()

        assert isinstance(observation, np.ndarray)
        assert observation.shape == env.observation_space.shape
        assert "balance" in info
        assert info["balance"] == 10000
        assert info["position"]["is_flat"]

    def test_step(self, real_candles_prefill):
        """Test environment step."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_steps=50,
            max_periods=100,
        )

        env.load_historical_data(real_candles_prefill)

        observation, info = env.reset()

        # Take a step with HOLD action
        next_obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_buy_action(self, real_candles_prefill):
        """Test BUY action."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            trade_quantity=1.0,
            max_periods=100,
        )

        env.load_historical_data(real_candles_prefill)

        observation, info = env.reset()
        initial_balance = info["balance"]

        # Execute BUY
        next_obs, reward, terminated, truncated, info = env.step(1)  # BUY

        # Should have a position now
        assert not info["position"]["is_flat"]
        assert info["position"]["is_long"]
        assert info["balance"] < initial_balance  # Fees deducted

    def test_sell_action(self, real_candles_prefill):
        """Test SELL action after BUY."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            trade_quantity=1.0,
            max_periods=100,
        )

        env.load_historical_data(real_candles_prefill)

        observation, info = env.reset()

        # BUY first
        env.step(1)

        # Then SELL
        next_obs, reward, terminated, truncated, info = env.step(2)

        # Position should be closed or reduced

    def test_with_indicators(self, real_candles_prefill):
        """Test environment with technical indicators."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T", "5T"],
            initial_balance=10000,
            max_periods=100,
        )

        env.load_historical_data(real_candles_prefill)

        # Add indicators
        env.add_indicator("1T", RSI(length=14), "RSI_14")
        env.add_indicator_to_all(SMA(period=20), "SMA_20")

        # Should rebuild observation space
        observation, info = env.reset()

        assert isinstance(observation, np.ndarray)
        assert observation.shape == env.observation_space.shape

    def test_episode_termination(self, real_candles_prefill):
        """Test episode termination conditions."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            max_steps=10,
            max_periods=100,
        )

        env.load_historical_data(real_candles_prefill)

        observation, info = env.reset()
        done = False
        steps = 0

        while not done and steps < 20:  # Safety limit
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        # Should terminate within max_steps
        assert done
        assert steps <= env.max_steps

    def test_reward_calculation(self, real_candles_prefill):
        """Test reward calculation."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            reward_strategy=SimplePnLReward(scale=0.01),
            max_periods=100,
        )

        env.load_historical_data(real_candles_prefill)

        observation, info = env.reset()

        # Take some actions
        for _ in range(5):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            # Reward should be a float
            assert isinstance(reward, (int, float))

    def test_get_statistics(self, real_candles_prefill):
        """Test getting environment statistics."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T", "5T"],
            initial_balance=10000,
            max_periods=100,
        )

        env.load_historical_data(real_candles_prefill)

        env.reset()

        stats = env.get_statistics()

        assert "episode_stats" in stats
        assert "simulator_state" in stats
        assert "asset_view_stats" in stats
        assert stats["episode_stats"]["episode"] == 1
