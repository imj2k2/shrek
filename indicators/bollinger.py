import numpy as np
import pandas as pd

def bollinger_bands(series: pd.Series, period: int = 20, num_std: int = 2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band, sma, lower_band
