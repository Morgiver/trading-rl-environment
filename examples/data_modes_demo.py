"""Demonstration of LINEAR and SHUFFLE data modes."""

from datetime import datetime, timedelta, time
import numpy as np
from trading_frame import Candle
from trading_rl_environment import TradingEnv, DataMode, SimplePnLReward


def generate_multi_day_data(n_days=10, candles_per_day=100):
    """Generate multiple days of candle data with timestamps."""
    candles = []
    base_date = datetime(2024, 1, 1, 9, 0, 0)  # Start at 9am
    price = 50000.0

    for day in range(n_days):
        for minute in range(candles_per_day):
            # Each day starts at 9am and runs for candles_per_day minutes
            candle_time = base_date + timedelta(days=day, minutes=minute)

            open_price = price
            close_price = price + np.random.randn() * 10
            high_price = max(open_price, close_price) + abs(np.random.randn() * 20)
            low_price = min(open_price, close_price) - abs(np.random.randn() * 20)
            volume = 100 + abs(np.random.randn() * 20)

            candle = Candle(
                date=candle_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
            )
            candles.append(candle)
            price = close_price

    return candles


def test_linear_mode():
    """Test LINEAR mode: continuous traversal of dataset."""
    print("=" * 70)
    print("LINEAR MODE DEMO")
    print("=" * 70)
    print("\nIn LINEAR mode, episodes continue sequentially through the dataset.")
    print("The AssetView timeframes are preserved between episodes.\n")

    # Generate data
    candles = generate_multi_day_data(n_days=5, candles_per_day=200)
    print(f"Generated {len(candles)} candles ({len(candles) // 200} days)")

    # Create environment in LINEAR mode
    env = TradingEnv(
        symbol="BTC/USDT",
        timeframes=["1T", "5T"],
        initial_balance=10000,
        data_mode=DataMode.LINEAR,
        max_steps=100,  # Short episodes for demo
        episode_start_time=time(9, 30),  # Start at 9:30am
        episode_end_time=time(11, 0),  # End at 11:00am
    )

    env.load_historical_data(candles)

    # Run 3 episodes
    for ep in range(3):
        obs, info = env.reset()
        print(f"\n--- Episode {ep + 1} ---")
        print(f"  Start idx: {info['episode_start_idx']}")
        print(f"  End idx: {info['episode_end_idx']}")
        print(f"  Start time: {candles[info['episode_start_idx']].date}")
        print(f"  End time: {candles[info['episode_end_idx']].date}")
        print(f"  Data mode: {info['data_mode']}")

        # Run episode
        done = False
        steps = 0
        while not done and steps < 50:  # Limit steps for demo
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        print(f"  Steps taken: {steps}")
        print(f"  Final candle idx: {info['candle_idx']}")

    print("\nNotice: Each episode continues from where the previous one ended!")


def test_shuffle_mode():
    """Test SHUFFLE mode: random day selection."""
    print("\n" + "=" * 70)
    print("SHUFFLE MODE DEMO")
    print("=" * 70)
    print("\nIn SHUFFLE mode, each episode starts at a random day.")
    print("The AssetView timeframes are rebuilt for each episode.\n")

    # Generate data
    candles = generate_multi_day_data(n_days=10, candles_per_day=200)
    print(f"Generated {len(candles)} candles ({len(candles) // 200} days)")

    # Create environment in SHUFFLE mode
    env = TradingEnv(
        symbol="BTC/USDT",
        timeframes=["1T", "5T"],
        initial_balance=10000,
        data_mode=DataMode.SHUFFLE,
        max_steps=100,
        episode_start_time=time(9, 30),
        episode_end_time=time(11, 0),
    )

    env.load_historical_data(candles)

    # Run 5 episodes
    for ep in range(5):
        obs, info = env.reset()
        print(f"\n--- Episode {ep + 1} ---")
        print(f"  Start idx: {info['episode_start_idx']}")
        print(f"  End idx: {info['episode_end_idx']}")
        print(f"  Start time: {candles[info['episode_start_idx']].date}")
        print(f"  End time: {candles[info['episode_end_idx']].date}")
        print(f"  Data mode: {info['data_mode']}")

        # Run episode
        done = False
        steps = 0
        while not done and steps < 50:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        print(f"  Steps taken: {steps}")

    print("\nNotice: Each episode starts at a different random day!")


def test_no_time_constraints():
    """Test without time constraints."""
    print("\n" + "=" * 70)
    print("NO TIME CONSTRAINTS DEMO")
    print("=" * 70)
    print("\nWithout time constraints, episodes use the full max_steps.\n")

    candles = generate_multi_day_data(n_days=5, candles_per_day=200)

    # SHUFFLE mode without time constraints
    env = TradingEnv(
        symbol="BTC/USDT",
        timeframes=["1T"],
        initial_balance=10000,
        data_mode=DataMode.SHUFFLE,
        max_steps=150,
        # No episode_start_time or episode_end_time
    )

    env.load_historical_data(candles)

    for ep in range(3):
        obs, info = env.reset()
        print(f"\n--- Episode {ep + 1} (SHUFFLE, no time limits) ---")
        print(f"  Start idx: {info['episode_start_idx']}")
        print(f"  End idx: {info['episode_end_idx']}")
        print(f"  Episode length: {info['episode_end_idx'] - info['episode_start_idx']}")

    print("\nNotice: Episodes use full max_steps without time restrictions!")


if __name__ == "__main__":
    test_linear_mode()
    test_shuffle_mode()
    test_no_time_constraints()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
