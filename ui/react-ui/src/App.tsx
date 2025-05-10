import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box } from '@chakra-ui/react';
import Dashboard from './pages/Dashboard';
import Screener from './pages/Screener';
import Backtest from './pages/Backtest';
import Portfolio from './pages/Portfolio';
import Analysis from './pages/Analysis';
import Navigation from './components/Navigation';
import { SocketProvider } from './hooks/useSocket';

const App: React.FC = () => {
  return (
    <SocketProvider>
      <Box display="flex" height="100vh" overflow="hidden">
        <Navigation />
        <Box flex="1" overflow="auto" p={4}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/screener" element={<Screener />} />
            <Route path="/backtest" element={<Backtest />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/analysis" element={<Analysis />} />
          </Routes>
        </Box>
      </Box>
    </SocketProvider>
  );
};

export default App;
