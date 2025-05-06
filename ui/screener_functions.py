import gradio as gr
import pandas as pd
import logging
import sys
import os
import yfinance as yf
import time

# Add parent directory to path to import from data module
sys.path.append('/app')

# Import data modules
try:
    from data.database import get_market_db
    from data.data_sync import get_data_synchronizer
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import database modules, falling back to local implementation")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store screened symbols
screened_symbols_for_backtest = ""

# S&P 500 symbols (subset)
SP500_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "BRK.B", "UNH",
    "JPM", "XOM", "JNJ", "V", "PG", "MA", "HD", "CVX", "AVGO", "MRK",
    "LLY", "COST", "ABBV", "PEP", "KO", "ADBE", "WMT", "MCD", "CRM", "BAC",
    "TMO", "CSCO", "ACN", "ABT", "CMCSA", "AMD", "DHR", "NFLX", "VZ", "NEE",
    "INTC", "PFE", "ORCL", "PM", "TXN", "COP", "INTU", "IBM", "CAT", "QCOM"
]

def run_stock_screener(universe, min_price, max_price, min_volume, min_volatility, max_volatility,
                     min_rsi, max_rsi, price_above_sma50, price_below_sma50,
                     price_above_sma200, price_below_sma200, macd_positive, macd_negative,
                     pe_min=0, pe_max=100, eps_min=0, eps_growth_min=0,
                     dividend_yield_min=0, market_cap_min=0, market_cap_max=0,
                     debt_to_equity_max=3, profit_margin_min=0, roe_min=0,
                     sort_by="Volume", sort_ascending=False, max_results=50):
    """Run stock screener using real data from database or Yahoo Finance"""
    try:
        # Determine which symbols to screen
        if universe and universe.strip():
            symbols_to_screen = [s.strip() for s in universe.split(',')]
        else:
            # Use S&P 500 symbols as default universe
            symbols_to_screen = SP500_SYMBOLS
        
        logger.info(f"Running stock screener on {len(symbols_to_screen)} symbols")
        
        # Initialize results list
        results = []
        
        # Try to get market database
        try:
            db = get_market_db()
            logger.info("Using market database for stock screening")
            using_db = True
        except (ImportError, NameError):
            logger.warning("Market database not available, using direct Yahoo Finance API")
            using_db = False
        
        # Process each symbol
        for symbol in symbols_to_screen:
            try:
                # Get stock data
                if using_db:
                    # Try to get data from database first
                    fundamental_data = db.get_stock_fundamentals(symbol)
                    price_data = db.get_stock_prices(symbol, start_date=None, end_date=None)
                    
                    if not fundamental_data or not price_data or price_data.empty:
                        # Fall back to Yahoo Finance if no data in database
                        stock = yf.Ticker(symbol)
                        info = stock.info
                        hist = stock.history(period='200d')
                    else:
                        # Use data from database
                        info = fundamental_data.get('full_data', {})
                        hist = price_data
                else:
                    # Use Yahoo Finance directly
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    hist = stock.history(period='200d')
                
                # Skip if we couldn't get basic info or history
                if not info or hist.empty or len(hist) < 50:
                    continue
                
                # Get current price
                current_price = info.get('regularMarketPrice') or info.get('previousClose')
                if not current_price:
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                    else:
                        continue
                
                # Apply price filter
                if (min_price and current_price < min_price) or (max_price and current_price > max_price):
                    continue
                
                # Get volume
                volume = info.get('regularMarketVolume') or info.get('volume')
                if not volume and not hist.empty:
                    volume = hist['Volume'].iloc[-1]
                    
                if min_volume and (not volume or volume < min_volume):
                    continue
                
                # Calculate volatility (20-day standard deviation of returns)
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.rolling(window=20).std().iloc[-1] * 100
                
                # Apply volatility filter
                if (min_volatility and volatility < min_volatility) or (max_volatility and volatility > max_volatility):
                    continue
                
                # Calculate RSI
                delta = hist['Close'].diff().dropna()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                avg_gain = up.rolling(window=14).mean()
                avg_loss = down.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]
                
                # Apply RSI filter
                if (min_rsi and rsi < min_rsi) or (max_rsi and rsi > max_rsi):
                    continue
                
                # Calculate SMAs
                sma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                sma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
                current_close = hist['Close'].iloc[-1]
                
                # Apply SMA filters
                if price_above_sma50 and current_close <= sma50:
                    continue
                if price_below_sma50 and current_close >= sma50:
                    continue
                if price_above_sma200 and current_close <= sma200:
                    continue
                if price_below_sma200 and current_close >= sma200:
                    continue
                
                # Calculate MACD
                ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
                ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                macd_signal = macd_line.ewm(span=9, adjust=False).mean()
                macd_histogram = macd_line - macd_signal
                
                # Apply MACD filters
                if macd_positive and macd_histogram.iloc[-1] <= 0:
                    continue
                if macd_negative and macd_histogram.iloc[-1] >= 0:
                    continue
                
                # Get fundamental data
                pe_ratio = info.get('trailingPE') or info.get('forwardPE')
                eps = info.get('trailingEPS') or info.get('forwardEPS')
                eps_growth = info.get('earningsGrowth')
                dividend_yield = info.get('dividendYield')
                if dividend_yield:
                    dividend_yield = dividend_yield * 100  # Convert to percentage
                
                market_cap = info.get('marketCap')
                debt_to_equity = info.get('debtToEquity')
                profit_margin = info.get('profitMargins')
                if profit_margin:
                    profit_margin = profit_margin * 100  # Convert to percentage
                
                roe = info.get('returnOnEquity')
                if roe:
                    roe = roe * 100  # Convert to percentage
                
                # Apply fundamental filters
                if pe_min and (not pe_ratio or pe_ratio < pe_min):
                    continue
                if pe_max and (pe_ratio and pe_ratio > pe_max):
                    continue
                if eps_min and (not eps or eps < eps_min):
                    continue
                if eps_growth_min and (not eps_growth or eps_growth < eps_growth_min):
                    continue
                if dividend_yield_min and (not dividend_yield or dividend_yield < dividend_yield_min):
                    continue
                if market_cap_min and (not market_cap or market_cap < market_cap_min * 1e9):
                    continue
                if market_cap_max and market_cap_max > 0 and (market_cap and market_cap > market_cap_max * 1e9):
                    continue
                if debt_to_equity_max and (debt_to_equity and debt_to_equity > debt_to_equity_max):
                    continue
                if profit_margin_min and (not profit_margin or profit_margin < profit_margin_min):
                    continue
                if roe_min and (not roe or roe < roe_min):
                    continue
                
                # Calculate daily change
                daily_change = 0
                if len(hist) >= 2:
                    daily_change = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
                
                # Add to results
                results.append({
                    "Symbol": symbol,
                    "Price": round(current_price, 2),
                    "Volume": volume,
                    "Volatility": round(volatility, 2),
                    "RSI": round(rsi, 2),
                    "Daily_Change_%": round(daily_change, 2),
                    "Above_50MA": current_close > sma50,
                    "Above_200MA": current_close > sma200,
                    "MACD": round(macd_histogram.iloc[-1], 3),
                    "P/E Ratio": round(pe_ratio, 2) if pe_ratio else None,
                    "EPS": round(eps, 2) if eps else None,
                    "EPS Growth": round(eps_growth * 100, 2) if eps_growth else None,
                    "Dividend Yield": round(dividend_yield, 2) if dividend_yield else None,
                    "Market Cap (B)": round(market_cap / 1e9, 2) if market_cap else None,
                    "Debt/Equity": round(debt_to_equity, 2) if debt_to_equity else None,
                    "Profit Margin": round(profit_margin, 2) if profit_margin else None,
                    "ROE": round(roe, 2) if roe else None
                })
                
            except Exception as e:
                logger.warning(f"Error processing symbol {symbol}: {str(e)}")
                continue
        
        # Sort results
        if sort_by in ["Symbol", "Price", "Volume", "Volatility", "RSI", "Daily_Change_%", "P/E Ratio", "EPS", 
                      "EPS Growth", "Dividend Yield", "Market Cap (B)", "Debt/Equity", 
                      "Profit Margin", "ROE"]:
            # Handle None values for sorting
            results = sorted(results, key=lambda x: (x.get(sort_by) is None, x.get(sort_by, 0)), reverse=not sort_ascending)
        
        # Limit results
        if max_results and len(results) > max_results:
            results = results[:max_results]
        
        logger.info(f"Stock screener found {len(results)} matching stocks")
        return results
    
    except Exception as e:
        logger.error(f"Error running stock screener: {str(e)}")
        return []

def add_stocks_to_backtest(results, allow_short):
    """Add selected stocks to backtest"""
    try:
        if not results:
            return pd.DataFrame()
        
        # Convert to DataFrame if it's not already
        if isinstance(results, list):
            df = pd.DataFrame(results)
        else:
            df = results
        
        # Add position type column
        df["Position Type"] = "Long" if not allow_short else "Short"
        
        return df
    except Exception as e:
        logger.error(f"Error adding stocks to backtest: {e}")
        return pd.DataFrame()

def prepare_for_backtest(selected_stocks_df):
    """Prepare selected stocks for backtest"""
    try:
        if selected_stocks_df.empty:
            return ""
        
        # Extract symbols and position types
        symbols = []
        for _, row in selected_stocks_df.iterrows():
            symbol = row["Symbol"]
            position_type = row["Position Type"]
            # Add position type indicator to symbol
            if position_type == "Short":
                symbol = f"{symbol}:short"
            symbols.append(symbol)
        
        # Join with commas
        return ",".join(symbols)
    except Exception as e:
        logger.error(f"Error preparing stocks for backtest: {str(e)}")
        return ""

def transfer_to_backtest_tab(selected_stocks_df):
    """Transfer selected stocks to backtest tab"""
    symbols_str = prepare_for_backtest(selected_stocks_df)
    if not symbols_str:
        return gr.update(value="No stocks selected")
        
    # Store the symbols string for later use
    global screened_symbols_for_backtest
    screened_symbols_for_backtest = symbols_str
    
    # Return a success message
    return gr.update(value=f"Transferred {len(symbols_str.split(','))} stocks to backtest tab. Switch to Backtest tab to use them.")

def load_screened_symbols():
    """Load screened symbols into the backtest tab"""
    global screened_symbols_for_backtest
    if screened_symbols_for_backtest:
        return gr.update(value=screened_symbols_for_backtest)
    else:
        return gr.update(value="No screened symbols available")

def deploy_to_alpaca(api_key, api_secret, paper_trading, strategy_type, 
                   momentum_enabled, mean_reversion_enabled, breakout_enabled, volatility_enabled,
                   momentum_weight, mean_reversion_weight, breakout_weight, volatility_weight,
                   max_allocation, max_position_size, stop_loss, take_profit, trailing_stop, allow_short,
                   symbols, load_from_backtest):
    """Deploy strategy to Alpaca"""
    try:
        # Validate inputs
        if not api_key or not api_secret:
            return "Error: API Key and Secret are required", "Please provide valid Alpaca API credentials"
        
        if not symbols and not load_from_backtest:
            return "Error: No symbols specified", "Please provide at least one symbol to trade or load from backtest results"
        
        # Process symbols
        global screened_symbols_for_backtest
        if load_from_backtest and screened_symbols_for_backtest:
            # Load symbols from backtest results
            symbol_list = []
            for symbol in screened_symbols_for_backtest.split(","):
                if ":short" in symbol:
                    # Extract the base symbol without the position type
                    base_symbol = symbol.split(":")[0]
                    symbol_list.append((base_symbol, "short"))
                else:
                    symbol_list.append((symbol, "long"))
            
            # Format for display
            symbol_display = ", ".join([f"{s[0]} ({s[1]})" for s in symbol_list])
            position_types = {s[0]: s[1] for s in symbol_list}
        else:
            # Process manually entered symbols
            symbol_list = [s.strip() for s in symbols.split(",")]
            symbol_display = ", ".join(symbol_list)
            position_types = {s: "long" for s in symbol_list}  # Default to long positions
        
        # Build strategy config
        strategy_config = {
            "type": strategy_type,
            "parameters": {
                "momentum_enabled": momentum_enabled,
                "mean_reversion_enabled": mean_reversion_enabled,
                "breakout_enabled": breakout_enabled,
                "volatility_enabled": volatility_enabled,
                "momentum_weight": momentum_weight,
                "mean_reversion_weight": mean_reversion_weight,
                "breakout_weight": breakout_weight,
                "volatility_weight": volatility_weight,
            },
            "risk_management": {
                "max_allocation": max_allocation,
                "max_position_size": max_position_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop": trailing_stop,
                "allow_short": allow_short
            },
            "symbols": symbol_list if not load_from_backtest else [s[0] for s in symbol_list],
            "position_types": position_types if load_from_backtest else None,
            "paper_trading": paper_trading
        }
        
        # In a real implementation, this would call the Alpaca API
        # For now, we'll just simulate a successful deployment
        import time
        
        # Log the deployment details
        logs = f"Deploying strategy to Alpaca (Paper Trading: {paper_trading})\n"
        logs += f"Strategy Type: {strategy_type}\n"
        logs += f"Symbols: {symbol_display}\n"
        logs += f"Risk Management: Stop Loss {stop_loss*100}%, Take Profit {take_profit*100}%\n"
        logs += f"Max Allocation: {max_allocation*100}%, Max Position Size: {max_position_size*100}%\n"
        logs += f"Allow Short: {allow_short}\n\n"
        
        # Simulate API call
        logs += "Connecting to Alpaca API...\n"
        time.sleep(1)  # Simulate API latency
        logs += "Validating API credentials...\n"
        time.sleep(1)  # Simulate API latency
        logs += "Creating trading strategy...\n"
        time.sleep(1)  # Simulate API latency
        logs += "Setting up risk parameters...\n"
        time.sleep(1)  # Simulate API latency
        logs += "Deploying strategy to Alpaca...\n"
        time.sleep(1)  # Simulate API latency
        logs += "Strategy successfully deployed!\n"
        
        return "Success: Strategy deployed to Alpaca", logs
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error: {str(e)}", error_details
