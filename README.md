# Trading RL Environment

A reinforcement learning environment for trading using [trading-frame](https://github.com/Morgiver/trading-frame), [trading-asset-view](https://github.com/Morgiver/trading-asset-view), and [trading-simulator](https://github.com/Morgiver/trading-simulator).

## Features

- **Gymnasium-compatible Environment**: Standard RL interface
- **Multi-timeframe support**: Use multiple timeframes simultaneously via trading-asset-view
- **Flexible reward strategies**: Base class for custom reward functions
- **Trading simulator integration**: Realistic order execution and PnL tracking
- **Training API**: Manage training sessions and collect statistics
- **Normalized observations**: ML-ready data from trading-frame

## Installation

### From GitHub

```bash
pip install git+https://github.com/Morgiver/trading-rl-environment.git
```

### For Development

```bash
git clone https://github.com/Morgiver/trading-rl-environment.git
cd trading-rl-environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev]
```

## Quick Start

```python
from trading_rl_environment import TradingEnv
from trading_rl_environment.reward_strategies import SimplePnLReward
from trading_frame import Candle
import numpy as np

# Create environment
env = TradingEnv(
    symbol="BTC/USDT",
    timeframes=["1T", "5T", "1H"],
    initial_balance=10000,
    reward_strategy=SimplePnLReward(),
    max_periods=100
)

# Load historical data
candles = [...]  # Your candle data
env.load_historical_data(candles)

# Standard RL loop
observation, info = env.reset()
done = False

while not done:
    # Your RL agent picks an action
    action = env.action_space.sample()  # Random action for demo

    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        print(f"Episode finished. Final PnL: {info['pnl']['net']}")
```

## Action Space

The environment uses a discrete action space with 3 actions:

- **0**: HOLD (do nothing)
- **1**: BUY (open/increase long position)
- **2**: SELL (close long/open short position)

## Observation Space

Observations are normalized numpy arrays containing:
- OHLCV data from all configured timeframes
- Technical indicators (if added)
- Current position information
- Account balance and PnL

Shape: `(n_timeframes * n_features + position_features,)`

## Reward Strategies

### Built-in Strategies

#### SimplePnLReward
Rewards based on change in unrealized PnL:
```python
from trading_rl_environment.reward_strategies import SimplePnLReward

reward_strategy = SimplePnLReward(
    scale=0.01  # Scale factor for reward normalization
)
```

### Custom Reward Strategies

Create your own reward function by subclassing `RewardStrategy`:

```python
from trading_rl_environment import RewardStrategy

class MyCustomReward(RewardStrategy):
    def calculate(self, env_state: dict, action: int, next_env_state: dict) -> float:
        # env_state contains:
        # - 'simulator_state': balance, position, pnl, equity
        # - 'asset_view_stats': price data, volumes, indicators
        # - 'current_step': episode step count

        pnl_change = next_env_state['simulator_state']['pnl']['unrealized'] - \
                     env_state['simulator_state']['pnl']['unrealized']

        # Add custom logic
        return pnl_change * 0.01

env = TradingEnv(
    symbol="BTC/USDT",
    timeframes=["5T"],
    reward_strategy=MyCustomReward()
)
```

## Advanced Usage

### Adding Technical Indicators

```python
from trading_frame.indicators import RSI, SMA, BollingerBands

env = TradingEnv(
    symbol="BTC/USDT",
    timeframes=["1T", "5T", "1H"],
    initial_balance=10000,
)

# Add indicators to specific timeframes
env.add_indicator("1T", RSI(length=14), "RSI_14")
env.add_indicator("5T", SMA(period=20), "SMA_20")

# Add to all timeframes
env.add_indicator_to_all(SMA(period=50), "SMA_50")
```

### Training Statistics

```python
# Get comprehensive statistics
stats = env.get_statistics()

print(stats['episode_stats'])  # Episode-specific stats
print(stats['simulator_state'])  # Current trading state
print(stats['asset_view_stats'])  # Multi-timeframe data stats
```

### Loading Data

```python
# From Candle list
from trading_frame import Candle
from datetime import datetime

candles = [
    Candle(
        date=datetime(2024, 1, 1, 12, 0),
        open=50000,
        high=51000,
        low=49000,
        close=50500,
        volume=100
    ),
    # ... more candles
]

env.load_historical_data(candles)

# From pandas DataFrame
import pandas as pd

df = pd.DataFrame({
    'timestamp': [...],
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

env.load_from_dataframe(df, date_column='timestamp')
```

## Environment Configuration

```python
env = TradingEnv(
    symbol="BTC/USDT",
    timeframes=["1T", "5T", "1H"],  # Multiple timeframes
    initial_balance=10000,           # Starting capital
    reward_strategy=SimplePnLReward(), # Reward function
    max_periods=100,                 # Max periods per timeframe
    max_steps=1000,                  # Max steps per episode
    trade_quantity=1.0,              # Fixed trade size
    pnl_mode=PnLMode.FIAT,          # PnL calculation mode
    fee_rate=0.001,                  # 0.1% trading fee
)
```

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=trading_rl_environment

# Specific test
pytest tests/test_env.py -v
```

## Examples

See the `examples/` directory for:
- **basic_usage.py**: Simple training loop
- **custom_reward.py**: Custom reward strategy example
- **multi_timeframe.py**: Using multiple timeframes with indicators

## License

MIT

## Contributing

Issues and pull requests welcome!

## Related Projects

- [trading-frame](https://github.com/Morgiver/trading-frame) - Core timeframe aggregation engine
- [trading-asset-view](https://github.com/Morgiver/trading-asset-view) - Multi-timeframe management
- [trading-simulator](https://github.com/Morgiver/trading-simulator) - Trading simulation engine
