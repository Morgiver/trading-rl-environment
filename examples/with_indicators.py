"""Example using technical indicators."""

from datetime import datetime, timedelta
import numpy as np
from trading_frame import Candle
from trading_frame.indicators import RSI, SMA, BollingerBands, MACD
from trading_rl_environment import TradingEnv, RiskAdjustedReward


def generate_trending_data(n_candles: int = 1000, start_price: float = 50000.0):
    """Generate sample candle data with a trend."""
    candles = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    price = start_price

    for i in range(n_candles):
        # Create a trending market
        trend = np.sin(i / 100) * 500  # Sine wave trend
        noise = np.random.randn() * 100
        price_change = trend + noise

        price += price_change

        open_price = price
        close_price = price + np.random.randn() * 50

        # Ensure high/low contain open/close
        high_price = max(open_price, close_price) + abs(np.random.randn() * 80)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 80)
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

        price = close_price

    return candles


def main():
    """Run example with technical indicators."""
    print("=" * 60)
    print("Trading RL Environment - With Technical Indicators")
    print("=" * 60)

    # Generate trending data
    print("\n1. Generating trending market data...")
    candles = generate_trending_data(n_candles=800)
    print(f"   Generated {len(candles)} candles")

    # Create environment with risk-adjusted reward
    print("\n2. Creating environment...")
    env = TradingEnv(
        symbol="BTC/USDT",
        timeframes=["1T", "5T", "15T", "1H"],
        initial_balance=10000,
        reward_strategy=RiskAdjustedReward(
            profit_weight=1.0, drawdown_penalty=2.0, position_penalty=0.05
        ),
        max_periods=100,
        max_steps=500,
        trade_quantity=0.5,
        fee_rate=0.001,
    )

    # Load data first
    print("\n3. Loading historical data...")
    env.load_historical_data(candles)

    # Add technical indicators
    print("\n4. Adding technical indicators...")

    # RSI on 1-minute timeframe
    env.add_indicator("1T", RSI(length=14), "RSI_14")
    print("   - Added RSI(14) to 1T timeframe")

    # SMA on all timeframes
    env.add_indicator_to_all(SMA(period=20), "SMA_20")
    env.add_indicator_to_all(SMA(period=50), "SMA_50")
    print("   - Added SMA(20) and SMA(50) to all timeframes")

    # MACD on 5-minute timeframe
    env.add_indicator("5T", MACD(fast=12, slow=26, signal=9), ["MACD", "SIGNAL", "HIST"])
    print("   - Added MACD to 5T timeframe")

    # Bollinger Bands on 15-minute timeframe
    env.add_indicator("15T", BollingerBands(period=20, std_dev=2.0), ["BB_UPPER", "BB_MID", "BB_LOWER"])
    print("   - Added Bollinger Bands to 15T timeframe")

    print(f"\n   Updated observation space: {env.observation_space}")
    obs_info = env.observation_builder.get_observation_info()
    print(f"   Total features: {obs_info['total_features']}")
    print(f"   Features per timeframe: {obs_info['features_per_timeframe']}")

    # Run episode
    print("\n5. Running episode with random actions...")
    observation, info = env.reset()

    print(f"\n   Initial state:")
    print(f"   - Balance: ${info['balance']:,.2f}")
    print(f"   - Equity: ${info['equity']:,.2f}")

    total_reward = 0
    steps = 0
    done = False
    max_equity = info["equity"]
    max_drawdown = 0

    while not done and steps < 200:
        # Random action
        action = env.action_space.sample()

        # Step
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        # Track metrics
        current_equity = info["equity"]
        max_equity = max(max_equity, current_equity)
        drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

        # Print progress
        if steps % 50 == 0:
            action_name = ["HOLD", "BUY", "SELL"][action]
            print(f"\n   Step {steps}:")
            print(f"   - Action: {action_name}")
            print(f"   - Reward: {reward:.4f}")
            print(f"   - Equity: ${info['equity']:,.2f}")
            print(f"   - Net PnL: ${info['pnl']['net']:,.2f}")
            print(f"   - Position: {info['position']['quantity']:.2f} units")
            print(f"   - Drawdown: {drawdown * 100:.2f}%")

    # Final results
    print(f"\n6. Episode Results:")
    print(f"   Total steps: {steps}")
    print(f"   Total reward: {total_reward:.4f}")
    print(f"   Termination: {'Early stop' if terminated else 'Max steps'}")

    print(f"\n   Performance Metrics:")
    print(f"   - Final Equity: ${info['equity']:,.2f}")
    print(f"   - Net PnL: ${info['pnl']['net']:,.2f} ({(info['pnl']['net'] / 10000 * 100):.2f}%)")
    print(f"   - Realized PnL: ${info['pnl']['realized']:,.2f}")
    print(f"   - Unrealized PnL: ${info['pnl']['unrealized']:,.2f}")
    print(f"   - Total Fees: ${info['pnl']['fees']:,.2f}")
    print(f"   - Max Drawdown: {max_drawdown * 100:.2f}%")

    # Get asset view statistics
    stats = env.get_statistics()
    av_stats = stats["asset_view_stats"]

    print(f"\n   Asset View Statistics:")
    print(f"   - Price Range: ${av_stats['price_range']['min']:,.2f} - ${av_stats['price_range']['max']:,.2f}")
    print(f"   - Total Volume: {av_stats['volume_stats']['total']:,.2f}")

    for tf in env.timeframes:
        print(f"   - {tf} periods: {av_stats['total_periods'][tf]}")
        print(f"     Indicators: {', '.join(av_stats['indicators'][tf])}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
