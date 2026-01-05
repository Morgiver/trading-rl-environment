"""
Demonstration of render() in debug mode.

Shows how to visualize the environment state during training/testing.
"""

import sys
from pathlib import Path

# Add src and utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from trading_rl_environment import TradingEnv, SimpleActionStrategy, ExtendedActionStrategy
from load_candles import load_candles_from_db


def demo_render_basic():
    """Demo render with basic SimpleActionStrategy."""
    print("\n" + "="*80)
    print("DEMO: render() with SimpleActionStrategy")
    print("="*80)

    # Load real NASDAQ data
    candles = load_candles_from_db(
        db_path="candle_CME_NQ.db",
        limit=10000  # 10k candles for quick demo
    )

    # Create environment
    env = TradingEnv(
        symbol="NQ",
        timeframes=["1T", "5T", "15T"],
        initial_balance=10000.0,
        max_steps=20,  # Short episode for demo
        action_strategy=SimpleActionStrategy(trade_quantity=1.0),
    )

    env.load_historical_data(candles)

    # Run one episode with render
    obs, info = env.reset()
    env.render()  # Render initial state

    for step in range(5):
        # Take random action
        action = env.action_space.sample()

        print(f"\n>>> Taking action: {action} ({env.action_strategy.get_action_description(action)})")

        obs, reward, terminated, truncated, info = env.step(action)

        # Render after action
        env.render()

        print(f"Reward: {reward:.4f}")

        if terminated or truncated:
            print(f"\nEpisode ended: terminated={terminated}, truncated={truncated}")
            break

    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80 + "\n")


def demo_render_extended():
    """Demo render with ExtendedActionStrategy."""
    print("\n" + "="*80)
    print("DEMO: render() with ExtendedActionStrategy (9 actions)")
    print("="*80)

    # Load real NASDAQ data
    candles = load_candles_from_db(
        db_path="candle_CME_NQ.db",
        limit=10000
    )

    # Create environment with extended actions
    env = TradingEnv(
        symbol="NQ",
        timeframes=["1T", "5T"],
        initial_balance=10000.0,
        max_steps=20,
        action_strategy=ExtendedActionStrategy(
            trade_quantity=1.0,
            limit_offset_pct=0.01,
            stop_loss_pct=0.02,
            take_profit_pct=0.02
        ),
    )

    env.load_historical_data(candles)

    # Run one episode with specific actions
    obs, info = env.reset()
    env.render()

    # Execute specific sequence of actions
    action_sequence = [
        (1, "BUY_MARKET - Open long position"),
        (7, "SET_STOP_LOSS - Add stop loss"),
        (8, "SET_TAKE_PROFIT - Add take profit"),
        (0, "HOLD - Wait"),
        (3, "BUY_LIMIT - Add limit order"),
    ]

    for action, description in action_sequence:
        print(f"\n>>> Taking action: {action} - {description}")

        obs, reward, terminated, truncated, info = env.step(action)

        # Render after action
        env.render()

        print(f"Reward: {reward:.4f}")

        if terminated or truncated:
            print(f"\nEpisode ended: terminated={terminated}, truncated={truncated}")
            break

    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run basic demo
    demo_render_basic()

    # Run extended demo
    demo_render_extended()
