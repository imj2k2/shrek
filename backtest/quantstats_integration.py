"""
QuantStats integration for enhanced backtest analytics
"""
import os
import json
import pandas as pd
import numpy as np
import quantstats as qs

# Configure QuantStats
qs.extend_pandas()

class QuantStatsAnalyzer:
    """Generate advanced analytics for backtest results using QuantStats"""
    
    def __init__(self, output_dir="/app/data/backtests"):
        """Initialize the analyzer with an output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def process_backtest_results(self, backtest_id, equity_curve, benchmark_data=None):
        """
        Process backtest results to generate QuantStats metrics
        
        Args:
            backtest_id (str): Unique identifier for the backtest
            equity_curve (list): List of dictionaries with date and equity values
            benchmark_data (pd.DataFrame, optional): Benchmark data for comparison
            
        Returns:
            dict: Dictionary containing file paths to the generated reports and metrics
        """
        # Create a directory for this backtest
        backtest_dir = os.path.join(self.output_dir, backtest_id)
        os.makedirs(backtest_dir, exist_ok=True)
        
        # Convert equity curve to pandas DataFrame
        df = pd.DataFrame(equity_curve)
        
        # Check if we have enough data to calculate metrics
        if len(df) < 2:
            print(f"Insufficient data points in equity curve for backtest {backtest_id}. Need at least 2 points, got {len(df)}")
            metrics_file = os.path.join(backtest_dir, "metrics.json")
            error_metrics = {
                "error": "Insufficient data points",
                "message": "At least 2 equity data points are required to calculate returns"
            }
            with open(metrics_file, 'w') as f:
                json.dump(error_metrics, f)
            
            return {
                "metrics_file": metrics_file,
                "error": "Insufficient data points",
                "backtest_id": backtest_id
            }
        
        # Process the data
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Extract returns
        equity_series = df['equity']
        # Calculate daily percentage returns
        returns = equity_series.pct_change().dropna()
        
        # Check if we have any returns after preprocessing
        if len(returns) == 0:
            print(f"No valid returns calculated for backtest {backtest_id} after preprocessing")
            metrics_file = os.path.join(backtest_dir, "metrics.json")
            error_metrics = {
                "error": "No valid returns",
                "message": "Could not calculate valid returns from equity curve"
            }
            with open(metrics_file, 'w') as f:
                json.dump(error_metrics, f)
            
            return {
                "metrics_file": metrics_file,
                "error": "No valid returns",
                "backtest_id": backtest_id
            }
        
        # Prepare benchmark returns if available
        benchmark_returns = None
        if benchmark_data is not None:
            try:
                if isinstance(benchmark_data, pd.DataFrame):
                    # Assume benchmark_data has a 'Close' column
                    if 'Close' in benchmark_data.columns:
                        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
                    elif 'close' in benchmark_data.columns:
                        benchmark_returns = benchmark_data['close'].pct_change().dropna()
                    # Align dates
                    if benchmark_returns is not None:
                        common_dates = returns.index.intersection(benchmark_returns.index)
                        returns = returns.loc[common_dates]
                        benchmark_returns = benchmark_returns.loc[common_dates]
            except Exception as e:
                print(f"Error processing benchmark data: {str(e)}")
                benchmark_returns = None
        
        # Generate metrics
        try:
            stats = qs.reports.metrics(
                returns, 
                benchmark=benchmark_returns,
                mode="full"  # Full set of metrics
            )
            
            # Add some additional metrics that might not be included by default
            rf = 0.015  # 1.5% risk-free rate
            ann_factor = np.sqrt(252)  # Annualization factor for daily returns
            
            vol = returns.std() * ann_factor
            if not pd.isna(vol) and vol != 0:
                sharpe = (returns.mean() * 252 - rf) / vol
                stats['sharpe_ratio'] = sharpe
            
            if 'cagr' in stats and 'max_drawdown' in stats and stats['max_drawdown'] != 0:
                calmar = stats['cagr'] / abs(stats['max_drawdown'])
                stats['calmar_ratio'] = calmar
            
            # Calculate win rate
            if len(returns) > 0:
                win_rate = (returns > 0).sum() / len(returns)
                stats['win_rate'] = win_rate
            
            # Save metrics as JSON
            metrics_file = os.path.join(backtest_dir, "metrics.json")
            stats.to_json(metrics_file, orient="index")
            
            # Save returns for charts
            returns_file = os.path.join(backtest_dir, "returns.csv")
            returns.to_frame("strategy").to_csv(returns_file)
            
            # Generate HTML tearsheet
            html_file = os.path.join(backtest_dir, "tear_sheet.html")
            qs.reports.html(
                returns, 
                benchmark=benchmark_returns,
                output=html_file, 
                download_filename=False, 
                title=f'Backtest {backtest_id}'
            )
            
            # Return file paths
            return {
                "metrics_file": metrics_file,
                "returns_file": returns_file,
                "html_file": html_file,
                "backtest_id": backtest_id
            }
            
        except Exception as e:
            print(f"Error generating QuantStats metrics: {str(e)}")
            # Create a minimal metrics file with error information
            metrics_file = os.path.join(backtest_dir, "metrics.json")
            error_metrics = {"error": str(e)}
            with open(metrics_file, 'w') as f:
                json.dump(error_metrics, f)
            
            return {
                "metrics_file": metrics_file,
                "error": str(e),
                "backtest_id": backtest_id
            }
    
    def load_metrics(self, backtest_id):
        """Load metrics for a given backtest"""
        metrics_file = os.path.join(self.output_dir, backtest_id, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return {"error": "Metrics not found"}
    
    def load_returns(self, backtest_id, freq=None):
        """
        Load returns for a given backtest
        
        Args:
            backtest_id (str): Backtest ID
            freq (str, optional): Frequency for resampling ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            dict: Dictionary with dates and returns
        """
        returns_file = os.path.join(self.output_dir, backtest_id, "returns.csv")
        if os.path.exists(returns_file):
            df = pd.read_csv(returns_file, parse_dates=['date'], index_col='date')
            
            # Resample if frequency is specified
            if freq:
                if freq in ['D', 'W', 'M', 'Q', 'Y']:
                    # For period returns
                    resampled = df.resample(freq).apply(
                        lambda x: (1 + x).prod() - 1
                    )
                    df = resampled
            
            return {
                "dates": df.index.astype(str).tolist(),
                "returns": df["strategy"].tolist()
            }
        return {"error": "Returns data not found"}
