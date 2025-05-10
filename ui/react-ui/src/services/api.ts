import axios from 'axios';

// Create an axios instance with default config
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor for auth tokens
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add a response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle 401 Unauthorized errors
    if (error.response && error.response.status === 401) {
      // Clear local storage and redirect to login
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Market Data API functions
export const fetchMarketSummary = async () => {
  const response = await api.get('/market/summary');
  return response.data;
};

export const fetchStockData = async (symbol: string, timeframe: string) => {
  const response = await api.get(`/market/stock/${symbol}`, {
    params: { timeframe },
  });
  return response.data;
};

// Portfolio API functions
export const fetchPortfolioSummary = async () => {
  const response = await api.get('/portfolio/summary');
  return response.data;
};

export const fetchPortfolioPositions = async () => {
  const response = await api.get('/portfolio/positions');
  return response.data;
};

export const fetchPortfolioHistory = async (timeframe: string) => {
  const response = await api.get('/portfolio/history', {
    params: { timeframe },
  });
  return response.data;
};

// Trading Signals API functions
export const fetchRecentSignals = async () => {
  const response = await api.get('/signals/recent');
  return response.data;
};

export const fetchSignalsBySymbol = async (symbol: string) => {
  const response = await api.get(`/signals/symbol/${symbol}`);
  return response.data;
};

// Stock Screener API functions
export const runStockScreener = async (criteria: any) => {
  const response = await api.post('/screener/run', criteria);
  return response.data;
};

export const saveScreenerCriteria = async (name: string, criteria: any) => {
  const response = await api.post('/screener/save', { name, criteria });
  return response.data;
};

export const getSavedScreeners = async () => {
  const response = await api.get('/screener/saved');
  return response.data;
};

// Backtesting API functions
export const runBacktest = async (params: any) => {
  // Format parameters to match backend API
  const backTestRequest = {
    agent_type: params.agentType || 'stocks',
    symbols: params.symbols.map((s: any) => s.symbol),
    start_date: params.startDate,
    end_date: params.endDate,
    timeframe: params.timeframe || 'day',
    strategy_name: params.strategy || 'momentum',
    initial_capital: params.initialCapital || 100000,
    use_mock_data: params.useMockData || false,
    strategy_config: {
      max_drawdown: params.stopLoss / 100 || 0.05,  // Convert to decimal
      trailing_stop: params.takeProfit / 100 || 0.1,  // Convert to decimal
      max_position_size: 0.2,
      position_types: params.symbols.reduce((acc: any, s: any) => {
        acc[s.symbol] = s.positionType || 'long';
        return acc;
      }, {})
    }
  };

  const response = await api.post('/backtest/run', backTestRequest);
  return response.data;
};

export const getBacktestResults = async (backtestId: string) => {
  const response = await api.get(`/backtest/results/${backtestId}`);
  return response.data;
};

export const getSavedBacktests = async () => {
  const response = await api.get('/backtest/saved');
  return response.data;
};

// User Authentication functions
export const login = async (email: string, password: string) => {
  const response = await api.post('/auth/login', { email, password });
  localStorage.setItem('auth_token', response.data.token);
  return response.data;
};

export const register = async (userData: any) => {
  const response = await api.post('/auth/register', userData);
  return response.data;
};

export const logout = async () => {
  localStorage.removeItem('auth_token');
  await api.post('/auth/logout');
};

export const getCurrentUser = async () => {
  const response = await api.get('/auth/me');
  return response.data;
};

// Export the api instance for direct use if needed
export default api;
