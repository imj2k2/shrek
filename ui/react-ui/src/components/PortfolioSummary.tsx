

import React, { useState, useEffect } from 'react';

interface PortfolioStats {
  total_value: number;
  cash_balance: number;
  equity_value: number;
  daily_pnl: number;
  total_pnl: number;
  total_pnl_percent: number;
  positions: number;
  long_positions: number;
  short_positions: number;
}

const defaultStats: PortfolioStats = {
  total_value: 100000,
  cash_balance: 75000,
  equity_value: 25000,
  daily_pnl: 250,
  total_pnl: 1200,
  total_pnl_percent: 1.2,
  positions: 3,
  long_positions: 2,
  short_positions: 1
};

interface PortfolioSummaryProps {
  stats?: PortfolioStats;
  data?: PortfolioStats; // For backward compatibility
  isLoading?: boolean;
}

const PortfolioSummary: React.FC<PortfolioSummaryProps> = ({ stats: propStats, data, isLoading: propLoading }: PortfolioSummaryProps) => {
  const [localStats, setLocalStats] = useState<PortfolioStats>(defaultStats);
  const [localLoading, setLocalLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Use props if provided, otherwise use local state
  // First try 'stats' prop, then fall back to 'data' prop, then use local state
  const stats = propStats !== undefined ? propStats : data !== undefined ? data : localStats;
  const loading = propLoading !== undefined ? propLoading : localLoading;

  useEffect(() => {
    // Fetch portfolio summary data from API
    const fetchPortfolioData = async () => {
      try {
        setLocalLoading(true);
        const response = await fetch('/api/portfolio/summary');
        
        if (!response.ok) {
          throw new Error(`Error fetching portfolio data: ${response.statusText}`);
        }
        
        const data = await response.json();
        setLocalStats(data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch portfolio data:', err);
        setError('Failed to load portfolio data');
        // Keep using the default stats in development
      } finally {
        setLocalLoading(false);
      }
    };

    fetchPortfolioData();
    
    // Refresh every minute
    const intervalId = setInterval(fetchPortfolioData, 60000);
    
    return () => clearInterval(intervalId);
  }, []);

  if (loading) {
    return <div className="loading">Loading portfolio data...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  // Determine if PnL is positive or negative to apply styles
  const dailyPnlClass = stats.daily_pnl >= 0 ? 'positive' : 'negative';
  const totalPnlClass = stats.total_pnl >= 0 ? 'positive' : 'negative';

  return (
    <div className="portfolio-summary">
      <h3>Portfolio Summary</h3>
      <div className="portfolio-stats">
        <div className="stats-row">
          <div className="stat-item">
            <span className="stat-label">Total Value</span>
            <span className="stat-value">${stats.total_value.toLocaleString()}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Cash Balance</span>
            <span className="stat-value">${stats.cash_balance.toLocaleString()}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Equity Value</span>
            <span className="stat-value">${stats.equity_value.toLocaleString()}</span>
          </div>
        </div>
        
        <div className="stats-row">
          <div className="stat-item">
            <span className="stat-label">Daily P&L</span>
            <span className={`stat-value ${dailyPnlClass}`}>
              {stats.daily_pnl >= 0 ? '+' : ''}${stats.daily_pnl.toLocaleString()}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Total P&L</span>
            <span className={`stat-value ${totalPnlClass}`}>
              {stats.total_pnl >= 0 ? '+' : ''}${stats.total_pnl.toLocaleString()} 
              ({stats.total_pnl_percent >= 0 ? '+' : ''}{stats.total_pnl_percent.toFixed(2)}%)
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Positions</span>
            <span className="stat-value">
              {stats.positions} ({stats.long_positions} long, {stats.short_positions} short)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PortfolioSummary;
