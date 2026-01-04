"""Utility to load candles from SQLite database."""

import sqlite3
from datetime import datetime
from typing import List, Optional
from trading_frame import Candle


def load_candles_from_db(
    db_path: str = "candle_CME_NQ.db",
    limit: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Candle]:
    """
    Load 1-minute candles from SQLite database.

    Parameters:
        db_path: Path to SQLite database file
        limit: Maximum number of candles to load (None for all)
        start_date: Start date filter (optional)
        end_date: End date filter (optional)

    Returns:
        List of Candle objects sorted by open_time

    Example:
        >>> candles = load_candles_from_db(limit=10000)
        >>> print(f"Loaded {len(candles)} candles")
        >>> print(f"Date range: {candles[0].date} to {candles[-1].date}")
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Build query with filters
    query = "SELECT open_time, open, high, low, close, volume FROM candles WHERE timeframe = '1m'"
    params = []

    if start_date:
        query += " AND open_time >= ?"
        params.append(start_date.timestamp())

    if end_date:
        query += " AND open_time <= ?"
        params.append(end_date.timestamp())

    query += " ORDER BY open_time ASC"

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Convert to Candle objects
    candles = []
    for row in rows:
        open_time, open_price, high_price, low_price, close_price, volume = row

        candle = Candle(
            date=datetime.fromtimestamp(open_time),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )
        candles.append(candle)

    return candles


if __name__ == "__main__":
    # Test the function
    print("Loading sample candles...")
    candles = load_candles_from_db(limit=1000)
    print(f"Loaded {len(candles)} candles")
    print(f"Date range: {candles[0].date} to {candles[-1].date}")
    print(f"First candle: O={candles[0].open_price}, H={candles[0].high_price}, L={candles[0].low_price}, C={candles[0].close_price}, V={candles[0].volume}")
