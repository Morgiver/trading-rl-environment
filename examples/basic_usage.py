"""Basic usage example of TradingEnv."""

from datetime import datetime, timedelta
from trading_frame import Candle
from trading_rl_environment import TradingEnv, SimplePnLReward


def generate_sample_data(n_candles: int = 1000, start_price: float = 50000.0):
    """Generate sample candle data for testing."""
    import numpy as np

    candles = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    price = start_price

    for i in range(n_candles):
        # Random walk with slight upward bias
        price_change = np.random.randn() * 100 + 5
        price += price_change

        open_price = price
        close_price = price + np.random.randn() * 30

        # Ensure high/low contain open/close
        high_price = max(open_price, close_price) + abs(np.random.randn() * 50)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 50)
        volume = 100 + abs(np.random.randn() * 50)

        candle = Candle(
            date=base_time + timedelta(minutes=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )
        candles.append(candle)

        # Update price for next iteration
        price = close_price

    return candles


def main():
    """Run basic training example."""
    print("=" * 60)
    print("Trading RL Environment - Basic Usage Example")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample candle data...")
    candles = generate_sample_data(n_candles=500)
    print(f"   Generated {len(candles)} candles")

    # Create environment
    print("\n2. Creating environment...")
    env = TradingEnv(
        symbol="BTC/USDT",
        timeframes=["1T", "5T", "15T"],  # 1min, 5min, 15min
        initial_balance=10000,
        reward_strategy=SimplePnLReward(scale=0.01),
        max_periods=50,
        max_steps=300,
        trade_quantity=0.1,
        fee_rate=0.001,  # 0.1% fee
    )
    print(f"   Symbol: {env.symbol}")
    print(f"   Timeframes: {env.timeframes}")
    print(f"   Initial balance: ${env.initial_balance:,.2f}")

    # Load historical data
    print("\n3. Loading historical data...")
    env.load_historical_data(candles)
    print(f"   Loaded {len(candles)} candles")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Run a simple episode with random actions
    print("\n4. Running episode with random actions...")
    observation, info = env.reset()

    print(f"\n   Initial state:")
    print(f"   - Balance: ${info['balance']:,.2f}")
    print(f"   - Equity: ${info['equity']:,.2f}")
    print(f"   - Position: {'FLAT' if info['position']['is_flat'] else 'ACTIVE'}")

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 100:  # Limit to 100 steps for demo
        # Random action (in practice, use your RL agent here)
        action = env.action_space.sample()

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        # Print progress every 20 steps
        if steps % 20 == 0:
            action_name = ["HOLD", "BUY", "SELL"][action]
            print(f"\n   Step {steps}:")
            print(f"   - Action: {action_name}")
            print(f"   - Reward: {reward:.4f}")
            print(f"   - Balance: ${info['balance']:,.2f}")
            print(f"   - Equity: ${info['equity']:,.2f}")
            print(f"   - Unrealized PnL: ${info['pnl']['unrealized']:,.2f}")
            print(f"   - Realized PnL: ${info['pnl']['realized']:,.2f}")

    # Final statistics
    print(f"\n5. Episode complete!")
    print(f"   Total steps: {steps}")
    print(f"   Total reward: {total_reward:.4f}")
    print(f"\n   Final state:")
    print(f"   - Balance: ${info['balance']:,.2f}")
    print(f"   - Equity: ${info['equity']:,.2f}")
    print(f"   - Net PnL: ${info['pnl']['net']:,.2f}")
    print(f"   - Realized PnL: ${info['pnl']['realized']:,.2f}")
    print(f"   - Unrealized PnL: ${info['pnl']['unrealized']:,.2f}")
    print(f"   - Total Fees: ${info['pnl']['fees']:,.2f}")

    # Get comprehensive statistics
    stats = env.get_statistics()
    print(f"\n6. Environment statistics:")
    print(f"   - Episodes: {stats['episode_stats']['episode']}")
    print(f"   - Total steps: {stats['episode_stats']['total_steps']}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
