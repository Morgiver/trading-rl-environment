# Trading RL Environment

A highly flexible reinforcement learning environment for trading using [trading-frame](https://github.com/Morgiver/trading-frame), [trading-asset-view](https://github.com/Morgiver/trading-asset-view), and [trading-simulator](https://github.com/Morgiver/trading-simulator).

## Features

- **Gymnasium-compatible Environment**: Standard RL interface
- **Multi-timeframe support**: Use multiple timeframes simultaneously via trading-asset-view
- **Flexible reward strategies**: Base class for custom reward functions
- **Dynamic action strategies**: Choose from simple, extended, or continuous action spaces
- **Trading simulator integration**: Realistic order execution with MARKET, LIMIT, STOP_LOSS, TAKE_PROFIT orders
- **Real market data support**: Built-in utilities for loading candles from SQLite databases
- **Log-scale normalization**: Unbounded balance/equity growth tracking
- **Episode modes**: LINEAR (sequential) or SHUFFLE (random day selection)
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

## Action Spaces

The environment supports multiple action strategies via the `ActionStrategy` pattern:

### SimpleActionStrategy (Default)
Discrete(3) - Basic trading actions:
- **0**: HOLD (do nothing)
- **1**: BUY (market order)
- **2**: SELL (market order or close position)

```python
from trading_rl_environment import TradingEnv, SimpleActionStrategy

env = TradingEnv(
    symbol="CME_NQ",
    timeframes=["1T"],
    action_strategy=SimpleActionStrategy(trade_quantity=1.0)
)
```

### ExtendedActionStrategy
Discrete(9) - Advanced trading with limit orders and risk management:
- **0**: HOLD
- **1**: BUY_MARKET
- **2**: SELL_MARKET
- **3**: BUY_LIMIT (1% below current price)
- **4**: SELL_LIMIT (1% above current price)
- **5**: CLOSE_POSITION (close entire position)
- **6**: CANCEL_ALL_ORDERS
- **7**: SET_STOP_LOSS (-2% from current price)
- **8**: SET_TAKE_PROFIT (+2% from current price)

```python
from trading_rl_environment import ExtendedActionStrategy

env = TradingEnv(
    symbol="CME_NQ",
    timeframes=["1T"],
    action_strategy=ExtendedActionStrategy(
        trade_quantity=1.0,
        limit_offset_pct=0.01,
        stop_loss_pct=0.02,
        take_profit_pct=0.02
    )
)
```

### ContinuousActionStrategy
Box(3) - Fine-grained control over all order parameters:
- **action[0]**: Order type [-1, 1] → [MARKET, LIMIT, STOP_LOSS, TAKE_PROFIT]
- **action[1]**: Side [-1, 1] → [BUY, SELL]
- **action[2]**: Quantity ratio [-1, 1] → [0, 2×trade_quantity]

```python
from trading_rl_environment import ContinuousActionStrategy

env = TradingEnv(
    symbol="CME_NQ",
    timeframes=["1T"],
    action_strategy=ContinuousActionStrategy(
        trade_quantity=1.0,
        max_price_offset_pct=0.05
    )
)
```

### Custom Action Strategies
Create your own action strategy by subclassing `ActionStrategy`:

```python
from trading_rl_environment import ActionStrategy
from gymnasium import spaces

class MyCustomActionStrategy(ActionStrategy):
    def get_action_space(self):
        return spaces.MultiDiscrete([4, 2, 3])  # [order_type, side, quantity_level]

    def execute_action(self, action, simulator, env_state):
        # Your custom execution logic
        pass

    def get_action_description(self, action):
        return f"Custom action: {action}"
```

## Observation Space

Observations are normalized numpy arrays containing:
- OHLCV data from all configured timeframes
- Technical indicators (if added)
- Current position information
- Account balance and PnL

Shape: `(n_timeframes * n_features + position_features,)`

## Reward Strategies

The environment supports custom reward functions via the `RewardStrategy` pattern:

### Built-in Strategies

#### SimplePnLReward (Default)
Rewards based on change in unrealized PnL:
```python
from trading_rl_environment import SimplePnLReward

env = TradingEnv(
    symbol="CME_NQ",
    timeframes=["1T"],
    reward_strategy=SimplePnLReward(scale=0.01)
)
```

#### RealizedPnLReward
Only rewards when positions are closed (realized PnL):
```python
from trading_rl_environment import RealizedPnLReward

env = TradingEnv(
    symbol="CME_NQ",
    timeframes=["1T"],
    reward_strategy=RealizedPnLReward(scale=0.01)
)
```

#### SharpeRatioReward
Encourages consistent returns with low volatility:
```python
from trading_rl_environment import SharpeRatioReward

env = TradingEnv(
    symbol="CME_NQ",
    timeframes=["1T"],
    reward_strategy=SharpeRatioReward(window_size=20, scale=1.0)
)
```

#### RiskAdjustedReward
Balances profit with risk management (penalizes drawdowns):
```python
from trading_rl_environment import RiskAdjustedReward

env = TradingEnv(
    symbol="CME_NQ",
    timeframes=["1T"],
    reward_strategy=RiskAdjustedReward(
        profit_weight=1.0,
        drawdown_penalty=2.0,
        position_penalty=0.1,
        scale=0.01
    )
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
        # - 'current_candle_idx': episode step count

        pnl_change = next_env_state['simulator_state']['pnl']['unrealized'] - \
                     env_state['simulator_state']['pnl']['unrealized']

        # Add custom logic
        return pnl_change * 0.01

    def reset(self):
        # Reset any internal state
        pass

env = TradingEnv(
    symbol="CME_NQ",
    timeframes=["1T"],
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

## Data Modes

The environment supports two episode generation modes:

### LINEAR Mode (Default)
Episodes continue sequentially through the dataset, preserving timeframe state:
```python
env = TradingEnv(
    symbol="CME_NQ",
    timeframes=["1T"],
    data_mode=DataMode.LINEAR
)
```

### SHUFFLE Mode
Each episode starts from a random day:
```python
from trading_rl_environment import DataMode

env = TradingEnv(
    symbol="CME_NQ",
    timeframes=["1T"],
    data_mode=DataMode.SHUFFLE,
    episode_start_time=time(9, 30),  # Start at 9:30 AM
    episode_end_time=time(16, 0)      # End at 4:00 PM
)
```

## Loading Real Market Data

```python
from utils.load_candles import load_candles_from_db
from datetime import datetime

# Load from SQLite database
candles = load_candles_from_db(
    db_path="candle_CME_NQ.db",
    limit=100000,
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31)
)

env.load_historical_data(candles)
```

## Examples

See the `examples/` directory for:
- **basic_usage.py**: Simple training loop
- **data_modes_demo.py**: LINEAR vs SHUFFLE mode comparison
- **with_indicators.py**: Using multiple timeframes with technical indicators
- **action_strategies_demo.py**: Demonstration of all action strategies

## License

MIT

## Contributing

Issues and pull requests welcome!

## Related Projects

- [trading-frame](https://github.com/Morgiver/trading-frame) - Core timeframe aggregation engine
- [trading-asset-view](https://github.com/Morgiver/trading-asset-view) - Multi-timeframe management
- [trading-simulator](https://github.com/Morgiver/trading-simulator) - Trading simulation engine
