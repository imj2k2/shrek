import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_API_SECRET: str = os.getenv("ALPACA_API_SECRET", "")
    ROBINHOOD_USERNAME: str = os.getenv("ROBINHOOD_USERNAME", "")
    ROBINHOOD_PASSWORD: str = os.getenv("ROBINHOOD_PASSWORD", "")
    DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")
    LUMIBOT_CONFIG: str = os.getenv("LUMIBOT_CONFIG", "")
    # Add more settings as needed

settings = Settings()
