import React, { useState, useEffect } from 'react';

interface Signal {
  id: number;
  symbol: string;
  strategy: string;
  action: string;
  price: number;
  timestamp: string;
  position_type: string;
}

interface RecentSignalsProps {
  limit?: number;
  signals?: Signal[];
  isLoading?: boolean;
}

const RecentSignals: React.FC<RecentSignalsProps> = ({ limit = 5, signals: propSignals, isLoading: propLoading }: RecentSignalsProps) => {
  const [localSignals, setLocalSignals] = useState<Signal[]>([]);
  const [localLoading, setLocalLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Use props if provided, otherwise use local state
  const signals = propSignals !== undefined ? propSignals : localSignals;
  const loading = propLoading !== undefined ? propLoading : localLoading;

  useEffect(() => {
    // Fetch recent trading signals from the API
    const fetchSignals = async () => {
      try {
        setLocalLoading(true);
        const response = await fetch(`/api/signals/recent?limit=${limit}`);
        
        if (!response.ok) {
          throw new Error(`Error fetching signals: ${response.statusText}`);
        }
        
        const data = await response.json();
        setLocalSignals(data.results || []);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch recent signals:', err);
        setError('Failed to load recent trading signals');
        // For development, create some sample data if API fails
        setLocalSignals([
          {
            id: 1,
            symbol: 'AAPL',
            strategy: 'Momentum',
            action: 'BUY',
            price: 182.54,
            timestamp: new Date().toISOString(),
            position_type: 'long'
          },
          {
            id: 2,
            symbol: 'MSFT',
            strategy: 'Mean Reversion',
            action: 'BUY',
            price: 322.78,
            timestamp: new Date().toISOString(),
            position_type: 'long'
          },
          {
            id: 3,
            symbol: 'TSLA',
            strategy: 'Breakout',
            action: 'SELL',
            price: 245.12,
            timestamp: new Date().toISOString(),
            position_type: 'short'
          }
        ]);
      } finally {
        setLocalLoading(false);
      }
    };

    fetchSignals();
    
    // Set up a refresh interval
    const intervalId = setInterval(fetchSignals, 60000); // Refresh every minute
    
    // Clean up the interval when component unmounts
    return () => clearInterval(intervalId);
  }, [limit]);

  if (loading) {
    return <div className="loading">Loading recent signals...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="recent-signals">
      <h3>Recent Trading Signals</h3>
      {signals.length === 0 ? (
        <p>No recent signals found</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Strategy</th>
              <th>Action</th>
              <th>Price</th>
              <th>Position Type</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {signals.map((signal) => (
              <tr key={signal.id}>
                <td>{signal.symbol}</td>
                <td>{signal.strategy}</td>
                <td className={signal.action === 'BUY' ? 'buy' : 'sell'}>
                  {signal.action}
                </td>
                <td>${signal.price.toFixed(2)}</td>
                <td>{signal.position_type}</td>
                <td>{new Date(signal.timestamp).toLocaleTimeString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default RecentSignals;
