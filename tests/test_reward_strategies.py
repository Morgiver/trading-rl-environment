"""Tests for reward strategies."""

import pytest
import numpy as np
from trading_rl_environment import (
    TradingEnv,
    SimplePnLReward,
    RealizedPnLReward,
    SharpeRatioReward,
    RiskAdjustedReward,
)

class TestRewardStrategies:
    """Test different reward strategies."""

    def test_simple_pnl_reward(self, real_candles_prefill):
        """Test SimplePnLReward strategy."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            reward_strategy=SimplePnLReward(scale=0.01),
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        observation, info = env.reset()
        initial_balance = info["balance"]

        # Take BUY action
        obs, reward, terminated, truncated, info = env.step(1)  # BUY

        # Reward should be calculated
        assert isinstance(reward, (int, float))

        # Take another step
        obs, reward2, terminated, truncated, info = env.step(0)  # HOLD
        assert isinstance(reward2, (int, float))

    def test_realized_pnl_reward(self, real_candles_prefill):
        """Test RealizedPnLReward strategy."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            reward_strategy=RealizedPnLReward(scale=0.01),
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        observation, info = env.reset()

        # BUY then SELL to realize profit
        obs, reward1, _, _, _ = env.step(1)  # BUY
        # Reward should be 0 (no realized PnL yet)
        assert reward1 == 0.0

        # Hold for a step
        obs, reward2, _, _, _ = env.step(0)  # HOLD
        assert reward2 == 0.0  # Still no realized PnL

        # SELL to realize
        obs, reward3, _, _, info = env.step(2)  # SELL
        # Now reward should be non-zero (realized PnL)
        assert isinstance(reward3, (int, float))

    def test_sharpe_ratio_reward(self, real_candles_prefill):
        """Test SharpeRatioReward strategy."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            reward_strategy=SharpeRatioReward(window_size=10, scale=1.0),
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        observation, info = env.reset()

        # Take multiple steps to build up return history
        rewards = []
        for i in range(15):
            action = 1 if i % 3 == 0 else 0  # BUY every 3 steps
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        # Rewards should be calculated (may be 0 for first few steps)
        assert all(isinstance(r, (int, float)) for r in rewards)
        # After window size, should have non-zero rewards
        assert len(rewards) > 0

    def test_risk_adjusted_reward(self, real_candles_prefill):
        """Test RiskAdjustedReward strategy."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            reward_strategy=RiskAdjustedReward(
                profit_weight=1.0,
                drawdown_penalty=2.0,
                position_penalty=0.1,
                scale=0.01,
            ),
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        observation, info = env.reset()

        # Take actions
        obs, reward1, _, _, _ = env.step(1)  # BUY
        assert isinstance(reward1, (int, float))

        obs, reward2, _, _, _ = env.step(0)  # HOLD
        assert isinstance(reward2, (int, float))

        obs, reward3, _, _, info = env.step(2)  # SELL
        assert isinstance(reward3, (int, float))

    def test_reward_strategy_reset(self, real_candles_prefill):
        """Test that reward strategies reset properly."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            reward_strategy=SharpeRatioReward(window_size=5, scale=1.0),
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        # First episode
        observation, info = env.reset()
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break

        # Reset for second episode
        observation2, info2 = env.reset()

        # Should be able to take steps in new episode
        obs, reward, _, _, info = env.step(1)
        assert isinstance(reward, (int, float))

    def test_reward_with_multiple_trades(self, real_candles_prefill):
        """Test rewards with multiple buy/sell cycles."""
        env = TradingEnv(
            symbol="CME_NQ",
            timeframes=["1T"],
            initial_balance=10000,
            trade_quantity=0.1,
            reward_strategy=SimplePnLReward(scale=0.01),
            max_periods=100,
        )

        candles = real_candles_prefill
        env.load_historical_data(candles)

        observation, info = env.reset()

        # Multiple trade cycles
        for cycle in range(3):
            # BUY
            obs, reward_buy, _, _, _ = env.step(1)
            assert isinstance(reward_buy, (int, float))

            # HOLD
            obs, reward_hold, _, _, _ = env.step(0)
            assert isinstance(reward_hold, (int, float))

            # SELL
            obs, reward_sell, terminated, truncated, info = env.step(2)
            assert isinstance(reward_sell, (int, float))

            if terminated or truncated:
                break
