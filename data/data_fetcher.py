# Placeholder for data fetcher (Polygon.io, Yahoo Finance, etc.)
import requests
import os

class DataFetcher:
    def __init__(self):
        self.polygon_key = os.getenv("POLYGON_API_KEY", "")
    def fetch_stock_data(self, symbol: str, source: str = "yahoo"):
        if source == "polygon":
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2020-01-01/2020-12-31?adjusted=true&sort=asc&limit=120&apiKey={self.polygon_key}"
            resp = requests.get(url)
            return resp.json() if resp.ok else {}
        else:
            url = f"https://query1.finance.yahoo.com/v7/finance/chart/{symbol}"
            resp = requests.get(url)
            return resp.json() if resp.ok else {}
    def fetch_crypto_data(self, symbol: str):
        url = f"https://api.polygon.io/v2/aggs/ticker/X:{symbol}USD/range/1/day/2020-01-01/2020-12-31?adjusted=true&sort=asc&limit=120&apiKey={self.polygon_key}"
        resp = requests.get(url)
        return resp.json() if resp.ok else {}
    def fetch_options_data(self, symbol: str):
        url = f"https://api.polygon.io/v3/reference/options/contracts?ticker={symbol}&apiKey={self.polygon_key}"
        resp = requests.get(url)
        return resp.json() if resp.ok else {}
