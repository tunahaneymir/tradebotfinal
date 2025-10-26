"""
Data Fetcher - Basit Simülasyon Versiyonu
Gerçek Binance bağlantısı eklenene kadar test verisi üretir
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import time


class DataFetcher:
    """Base data fetcher class"""
    
    def __init__(self, source: str = "simulation"):
        """
        Initialize data fetcher
        
        Args:
            source: 'simulation', 'binance', 'ccxt'
        """
        self.source = source
        
    def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int = 500,
        since: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '15m', '1h', '4h')
            limit: Number of candles
            since: Start date (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.source == "simulation":
            return self._generate_simulation_data(symbol, timeframe, limit, since)
        elif self.source == "binance":
            return self._fetch_from_binance(symbol, timeframe, limit, since)
        else:
            raise ValueError(f"Unknown source: {self.source}")
    
    def _generate_simulation_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        since: Optional[datetime]
    ) -> pd.DataFrame:
        """
        Generate simulated OHLCV data for testing
        
        Generates realistic price movements with:
        - Trends (up, down, sideways)
        - Support/Resistance zones
        - Volatility
        """
        # Timeframe to minutes
        tf_minutes = self._timeframe_to_minutes(timeframe)
        
        # Start date
        if since is None:
            since = datetime.now() - timedelta(minutes=tf_minutes * limit)
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=since,
            periods=limit,
            freq=f'{tf_minutes}T'
        )
        
        # Initial price (BTC around $40,000, ETH around $2,500)
        if 'BTC' in symbol:
            base_price = 40000
        elif 'ETH' in symbol:
            base_price = 2500
        else:
            base_price = 100
        
        # Generate price movement
        prices = self._generate_realistic_prices(base_price, limit)
        
        # Create OHLCV from prices
        data = []
        for i, ts in enumerate(timestamps):
            open_price = prices[i]
            close_price = prices[i]
            
            # Add some intra-candle movement
            high_offset = abs(np.random.normal(0, 0.003))  # 0.3% average
            low_offset = abs(np.random.normal(0, 0.003))
            
            high = close_price * (1 + high_offset)
            low = close_price * (1 - low_offset)
            
            # Ensure open is between high and low
            if open_price > high:
                open_price = high
            if open_price < low:
                open_price = low
            
            # Volume (random but realistic)
            volume = np.random.uniform(100, 10000)
            
            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _generate_realistic_prices(self, base_price: float, length: int) -> np.ndarray:
        """
        Generate realistic price series with trends and zones
        """
        prices = np.zeros(length)
        prices[0] = base_price
        
        # Trend parameters
        trend_length = np.random.randint(50, 150)
        trend_direction = np.random.choice([1, -1, 0])  # up, down, sideways
        trend_strength = np.random.uniform(0.0005, 0.002)  # 0.05% to 0.2% per candle
        
        # Volatility
        volatility = 0.002  # 0.2% average
        
        candles_in_trend = 0
        
        for i in range(1, length):
            # Change trend periodically
            if candles_in_trend >= trend_length:
                trend_direction = np.random.choice([1, -1, 0])
                trend_strength = np.random.uniform(0.0005, 0.002)
                trend_length = np.random.randint(50, 150)
                candles_in_trend = 0
            
            # Trend movement
            trend_move = trend_direction * trend_strength
            
            # Random volatility
            random_move = np.random.normal(0, volatility)
            
            # Total movement
            total_move = trend_move + random_move
            
            # Apply movement
            prices[i] = prices[i-1] * (1 + total_move)
            
            # Ensure price stays positive
            if prices[i] <= 0:
                prices[i] = prices[i-1] * 0.99
            
            candles_in_trend += 1
        
        # Add some support/resistance bounces
        prices = self._add_support_resistance(prices)
        
        return prices
    
    def _add_support_resistance(self, prices: np.ndarray) -> np.ndarray:
        """Add realistic support/resistance bounces"""
        # Find local highs/lows to create zones
        window = 20
        
        for i in range(window, len(prices) - window):
            # Check if local high
            if prices[i] == max(prices[i-window:i+window]):
                # Add resistance (price bounces down)
                resistance = prices[i]
                for j in range(i+1, min(i+50, len(prices))):
                    if prices[j] > resistance * 0.995:
                        # Bounce with some probability
                        if np.random.random() < 0.3:
                            prices[j] = resistance * np.random.uniform(0.985, 0.995)
            
            # Check if local low
            elif prices[i] == min(prices[i-window:i+window]):
                # Add support (price bounces up)
                support = prices[i]
                for j in range(i+1, min(i+50, len(prices))):
                    if prices[j] < support * 1.005:
                        # Bounce with some probability
                        if np.random.random() < 0.3:
                            prices[j] = support * np.random.uniform(1.005, 1.015)
        
        return prices
    
    def _fetch_from_binance(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        since: Optional[datetime]
    ) -> pd.DataFrame:
        """
        Fetch real data from Binance
        TODO: Implement when ready for live trading
        """
        try:
            import ccxt
            
            exchange = ccxt.binance({
                'enableRateLimit': True,
            })
            
            # Fetch data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except ImportError:
            print("⚠️  ccxt not installed. Using simulation data.")
            return self._generate_simulation_data(symbol, timeframe, limit, since)
        except Exception as e:
            print(f"⚠️  Error fetching from Binance: {e}")
            print("    Falling back to simulation data.")
            return self._generate_simulation_data(symbol, timeframe, limit, since)
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        multipliers = {
            'm': 1,
            'h': 60,
            'd': 1440,
            'w': 10080
        }
        
        number = int(timeframe[:-1])
        unit = timeframe[-1]
        
        return number * multipliers.get(unit, 1)


class BinanceDataFetcher(DataFetcher):
    """Convenience class for Binance data fetching"""
    
    def __init__(self, use_testnet: bool = True):
        super().__init__(source="simulation")  # Default to simulation for now
        self.use_testnet = use_testnet


# Quick usage example
if __name__ == "__main__":
    # Create fetcher
    fetcher = BinanceDataFetcher()
    
    # Fetch data
    print("Fetching simulated BTC data...")
    data = fetcher.fetch_ohlcv('BTCUSDT', '15m', limit=100)
    
    print(f"\nData shape: {data.shape}")
    print(f"\nFirst 5 candles:")
    print(data.head())
    
    print(f"\nLast 5 candles:")
    print(data.tail())
    
    print(f"\nPrice range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
