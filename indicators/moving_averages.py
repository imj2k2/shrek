import numpy as np
import pandas as pd

def sma(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(window=period).mean()

def ema(series: pd.Series, period: int = 20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def ma(series: pd.Series, period: int = 20) -> pd.Series:
    return sma(series, period)
