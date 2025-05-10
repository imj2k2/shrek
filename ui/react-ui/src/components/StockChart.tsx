import React, { useState } from 'react';
import {
  Box,
  Flex,
  Button,
  ButtonGroup,
  Spinner,
  Text,
  useColorModeValue,
} from '@chakra-ui/react';
import { useQuery } from '@tanstack/react-query';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { fetchStockData } from '../services/api';

interface StockChartProps {
  symbol?: string;
  timeframe: string;
}

const StockChart: React.FC<StockChartProps> = ({ 
  symbol = 'SPY',  // Default to SPY (S&P 500 ETF)
  timeframe = '1d'  // Default to 1 day timeframe
}) => {
  // Define all hooks at the top of the component, before any conditional returns
  const [selectedSymbol, setSelectedSymbol] = useState(symbol);
  const gridColor = useColorModeValue('gray.200', 'gray.700');
  const lineColor = useColorModeValue('blue.500', 'blue.300');
  const tooltipBgColor = useColorModeValue('#fff', '#2D3748');
  const tooltipBorderColor = useColorModeValue('#E2E8F0', '#4A5568');
  
  const { data, isLoading, error } = useQuery(
    ['stockData', selectedSymbol, timeframe],
    () => fetchStockData(selectedSymbol, timeframe),
    { staleTime: 60000 }  // 1 minute
  );
  
  const popularSymbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA'];
  
  if (isLoading) {
    return (
      <Flex justify="center" align="center" height="300px">
        <Spinner size="xl" />
      </Flex>
    );
  }
  
  if (error) {
    return (
      <Flex justify="center" align="center" height="300px">
        <Text color="red.500">Error loading chart data</Text>
      </Flex>
    );
  }
  
  return (
    <Box>
      <Flex justify="space-between" mb={4} align="center">
        <ButtonGroup size="sm" isAttached variant="outline">
          {popularSymbols.map((sym) => (
            <Button
              key={sym}
              onClick={() => setSelectedSymbol(sym)}
              colorScheme={selectedSymbol === sym ? 'blue' : 'gray'}
            >
              {sym}
            </Button>
          ))}
        </ButtonGroup>
        
        <Text fontWeight="bold">
          {selectedSymbol} {data?.currentPrice && `$${data.currentPrice.toFixed(2)}`}
          {data?.percentChange && (
            <Text
              as="span"
              ml={2}
              color={data.percentChange >= 0 ? 'green.500' : 'red.500'}
            >
              ({data.percentChange >= 0 ? '+' : ''}
              {data.percentChange.toFixed(2)}%)
            </Text>
          )}
        </Text>
      </Flex>
      
      <Box height="300px">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data?.chartData || []}
            margin={{ top: 5, right: 20, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
            <XAxis dataKey="date" />
            <YAxis domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{
                backgroundColor: tooltipBgColor,
                borderColor: tooltipBorderColor,
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="price"
              stroke={lineColor}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  );
};

export default StockChart;
