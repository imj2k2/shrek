import numpy as np
import pandas as pd

def macd(series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD line, signal line and histogram
    
    Args:
        series: Price data as pandas Series or numpy array
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    # Convert to pandas Series if it's a numpy array
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    elif not isinstance(series, pd.Series):
        # Try to convert other types to Series
        try:
            series = pd.Series(series)
        except:
            raise ValueError(f"Input must be convertible to pandas Series, got {type(series)}")
            
    # Calculate MACD components
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram
