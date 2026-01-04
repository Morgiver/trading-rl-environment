"""
Demonstration of different action strategies.

Shows how to use SimpleActionStrategy, ExtendedActionStrategy, and ContinuousActionStrategy
for different levels of trading control.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from load_candles import load_candles_from_db

from trading_rl_environment import (
    TradingEnv,
    SimpleActionStrategy,
    ExtendedActionStrategy,
    ContinuousActionStrategy,
)
import numpy as np


def demo_simple_action_strategy():
    """Demo with simple 3-action strategy (HOLD, BUY, SELL)."""
    print("\n" + "=" * 60)
    print("DEMO 1: SimpleActionStrategy (3 actions)")
    print("=" * 60)

    # Load real candles
    candles = load_candles_from_db(limit=5000)

    # Create environment with SimpleActionStrategy (default)
    env = TradingEnv(
        symbol="CME_NQ",
        timeframes=["1T"],
        initial_balance=10000,
        max_periods=50,
        trade_quantity=1.0,
    )

    env.load_historical_data(candles)
    observation, info = env.reset()

    print(f"\nAction space: {env.action_space}")
    print(f"Action space size: {env.action_space.n}")
    print("\nActions:")
    for i in range(env.action_space.n):
        print(f"  {i}: {env.action_strategy.get_action_description(i)}")

    # Run a few steps
    print("\nRunning 5 steps with random actions:")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step + 1}: action={action} ({env.action_strategy.get_action_description(action)}), "
              f"balance={info['balance']:.2f}, position={info['position']['quantity']}")

        if terminated or truncated:
            break

    env.close()


def demo_extended_action_strategy():
    """Demo with extended 9-action strategy (includes limit orders, stop loss, etc.)."""
    print("\n" + "=" * 60)
    print("DEMO 2: ExtendedActionStrategy (9 actions)")
    print("=" * 60)

    # Load real candles
    candles = load_candles_from_db(limit=5000)

    # Create environment with ExtendedActionStrategy
    action_strategy = ExtendedActionStrategy(
        trade_quantity=1.0,
        limit_offset_pct=0.01,  # 1% limit offset
        stop_loss_pct=0.02,     # 2% stop loss
        take_profit_pct=0.02,   # 2% take profit
    )

    env = TradingEnv(
        symbol="CME_NQ",
        timeframes=["1T"],
        initial_balance=10000,
        action_strategy=action_strategy,
        max_periods=50,
    )

    env.load_historical_data(candles)
    observation, info = env.reset()

    print(f"\nAction space: {env.action_space}")
    print(f"Action space size: {env.action_space.n}")
    print("\nActions:")
    for i in range(env.action_space.n):
        print(f"  {i}: {env.action_strategy.get_action_description(i)}")

    # Demonstrate specific actions
    print("\nDemonstrating extended actions:")

    # Step 1: Buy market
    obs, reward, terminated, truncated, info = env.step(1)
    print(f"  1. BUY_MARKET: balance={info['balance']:.2f}, position={info['position']['quantity']}")

    # Step 2: Set stop loss
    obs, reward, terminated, truncated, info = env.step(7)
    print(f"  2. SET_STOP_LOSS: pending orders={env.simulator.get_pending_orders()}")

    # Step 3: Set take profit
    obs, reward, terminated, truncated, info = env.step(8)
    print(f"  3. SET_TAKE_PROFIT: pending orders={len(env.simulator.get_pending_orders())}")

    # Step 4: Place limit buy
    obs, reward, terminated, truncated, info = env.step(3)
    print(f"  4. BUY_LIMIT: pending orders={len(env.simulator.get_pending_orders())}")

    # Step 5: Cancel all orders
    obs, reward, terminated, truncated, info = env.step(6)
    print(f"  5. CANCEL_ALL: pending orders={len(env.simulator.get_pending_orders())}")

    # Step 6: Close position
    obs, reward, terminated, truncated, info = env.step(5)
    print(f"  6. CLOSE_POSITION: position={info['position']['quantity']}, is_flat={info['position']['is_flat']}")

    env.close()


def demo_continuous_action_strategy():
    """Demo with continuous action strategy (fine-grained control)."""
    print("\n" + "=" * 60)
    print("DEMO 3: ContinuousActionStrategy (Box space)")
    print("=" * 60)

    # Load real candles
    candles = load_candles_from_db(limit=5000)

    # Create environment with ContinuousActionStrategy
    action_strategy = ContinuousActionStrategy(
        trade_quantity=1.0,
        max_price_offset_pct=0.05,  # 5% max price offset
    )

    env = TradingEnv(
        symbol="CME_NQ",
        timeframes=["1T"],
        initial_balance=10000,
        action_strategy=action_strategy,
        max_periods=50,
    )

    env.load_historical_data(candles)
    observation, info = env.reset()

    print(f"\nAction space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action space bounds: low={env.action_space.low}, high={env.action_space.high}")
    print("\nAction format: [order_type, side, quantity]")
    print("  order_type: [-1, 1] -> [MARKET, LIMIT, STOP_LOSS, TAKE_PROFIT]")
    print("  side: [-1, 1] -> [BUY, SELL]")
    print("  quantity: [-1, 1] -> [0, 2*trade_quantity]")

    # Demonstrate specific continuous actions
    print("\nDemonstrating continuous actions:")

    # Action 1: Market buy with full quantity
    action1 = np.array([-1.0, -1.0, 1.0])  # MARKET, BUY, 2x quantity
    obs, reward, terminated, truncated, info = env.step(action1)
    print(f"  1. Action {action1}: {env.action_strategy.get_action_description(action1)}")
    print(f"     balance={info['balance']:.2f}, position={info['position']['quantity']}")

    # Action 2: Limit sell with half quantity
    action2 = np.array([0.0, 1.0, 0.0])  # LIMIT, SELL, 1x quantity
    obs, reward, terminated, truncated, info = env.step(action2)
    print(f"  2. Action {action2}: {env.action_strategy.get_action_description(action2)}")
    print(f"     pending orders={len(env.simulator.get_pending_orders())}")

    # Action 3: Random continuous action
    action3 = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action3)
    print(f"  3. Action {action3}: {env.action_strategy.get_action_description(action3)}")

    env.close()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("ACTION STRATEGIES DEMONSTRATION")
    print("=" * 60)

    demo_simple_action_strategy()
    demo_extended_action_strategy()
    demo_continuous_action_strategy()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
