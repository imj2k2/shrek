import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  Grid,
  GridItem,
  Heading,
  Input,
  Select,
  Spinner,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  Text,
  useColorModeValue,
} from '@chakra-ui/react';
import { useQuery } from '@tanstack/react-query';
import { fetchStockData } from '../services/api';
import StockChart from '../components/StockChart';

const Analysis: React.FC = () => {
  const [symbol, setSymbol] = useState('SPY');
  const [timeframe, setTimeframe] = useState('1M');
  
  const cardBg = useColorModeValue('white', 'gray.700');
  
  // Fetch stock data for the selected symbol
  const { data, isLoading, error } = useQuery(
    ['stockAnalysis', symbol, timeframe],
    () => fetchStockData(symbol, timeframe),
    { staleTime: 60000 } // 1 minute
  );
  
  return (
    <Box>
      <Flex justify="space-between" align="center" mb={6}>
        <Heading size="lg">Technical Analysis</Heading>
        <Flex gap={4}>
          <FormControl maxW="200px">
            <Select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              variant="filled"
              size="sm"
            >
              <option value="SPY">SPY (S&P 500)</option>
              <option value="QQQ">QQQ (Nasdaq)</option>
              <option value="AAPL">AAPL (Apple)</option>
              <option value="MSFT">MSFT (Microsoft)</option>
              <option value="NVDA">NVDA (NVIDIA)</option>
              <option value="GOOGL">GOOGL (Alphabet)</option>
              <option value="AMZN">AMZN (Amazon)</option>
            </Select>
          </FormControl>
          
          <FormControl maxW="150px">
            <Select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              variant="filled"
              size="sm"
            >
              <option value="1D">1 Day</option>
              <option value="1W">1 Week</option>
              <option value="1M">1 Month</option>
              <option value="3M">3 Months</option>
              <option value="6M">6 Months</option>
              <option value="1Y">1 Year</option>
              <option value="5Y">5 Years</option>
            </Select>
          </FormControl>
        </Flex>
      </Flex>
      
      {isLoading ? (
        <Flex justify="center" align="center" height="400px">
          <Spinner size="xl" />
        </Flex>
      ) : error ? (
        <Flex justify="center" align="center" height="400px">
          <Text color="red.500">Error loading analysis data</Text>
        </Flex>
      ) : (
        <Tabs isLazy variant="enclosed">
          <TabList>
            <Tab>Price Chart</Tab>
            <Tab>Technical Indicators</Tab>
            <Tab>Patterns</Tab>
            <Tab>Sentiment</Tab>
          </TabList>
          
          <TabPanels>
            {/* Price Chart Tab */}
            <TabPanel px={0}>
              <Card bg={cardBg} mb={4}>
                <CardHeader>
                  <Heading size="md">Price Chart</Heading>
                </CardHeader>
                <CardBody>
                  <StockChart symbol={symbol} timeframe={timeframe} />
                </CardBody>
              </Card>
              
              <Grid templateColumns={{ base: '1fr', md: 'repeat(3, 1fr)' }} gap={4}>
                <Card bg={cardBg}>
                  <CardHeader>
                    <Heading size="sm">Key Levels</Heading>
                  </CardHeader>
                  <CardBody>
                    <Grid templateColumns="repeat(2, 1fr)" gap={2}>
                      <Box>
                        <Text color="gray.500" fontSize="sm">Support</Text>
                        <Text fontWeight="bold">${(data?.currentPrice * 0.95).toFixed(2)}</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">Resistance</Text>
                        <Text fontWeight="bold">${(data?.currentPrice * 1.05).toFixed(2)}</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">52-Week Low</Text>
                        <Text fontWeight="bold">${(data?.currentPrice * 0.8).toFixed(2)}</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">52-Week High</Text>
                        <Text fontWeight="bold">${(data?.currentPrice * 1.2).toFixed(2)}</Text>
                      </Box>
                    </Grid>
                  </CardBody>
                </Card>
                
                <Card bg={cardBg}>
                  <CardHeader>
                    <Heading size="sm">Moving Averages</Heading>
                  </CardHeader>
                  <CardBody>
                    <Grid templateColumns="repeat(2, 1fr)" gap={2}>
                      <Box>
                        <Text color="gray.500" fontSize="sm">SMA 20</Text>
                        <Text fontWeight="bold" color={data?.currentPrice > (data?.currentPrice * 0.98) ? 'green.500' : 'red.500'}>
                          ${(data?.currentPrice * 0.98).toFixed(2)}
                        </Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">SMA 50</Text>
                        <Text fontWeight="bold" color={data?.currentPrice > (data?.currentPrice * 0.97) ? 'green.500' : 'red.500'}>
                          ${(data?.currentPrice * 0.97).toFixed(2)}
                        </Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">SMA 100</Text>
                        <Text fontWeight="bold" color={data?.currentPrice > (data?.currentPrice * 0.96) ? 'green.500' : 'red.500'}>
                          ${(data?.currentPrice * 0.96).toFixed(2)}
                        </Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">SMA 200</Text>
                        <Text fontWeight="bold" color={data?.currentPrice > (data?.currentPrice * 0.95) ? 'green.500' : 'red.500'}>
                          ${(data?.currentPrice * 0.95).toFixed(2)}
                        </Text>
                      </Box>
                    </Grid>
                  </CardBody>
                </Card>
                
                <Card bg={cardBg}>
                  <CardHeader>
                    <Heading size="sm">Volatility & Volume</Heading>
                  </CardHeader>
                  <CardBody>
                    <Grid templateColumns="repeat(2, 1fr)" gap={2}>
                      <Box>
                        <Text color="gray.500" fontSize="sm">ATR (14)</Text>
                        <Text fontWeight="bold">${(data?.currentPrice * 0.02).toFixed(2)}</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">Bollinger %B</Text>
                        <Text fontWeight="bold">0.65</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">Volume</Text>
                        <Text fontWeight="bold">3.2M</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">Vol vs Avg</Text>
                        <Text fontWeight="bold" color="green.500">+12%</Text>
                      </Box>
                    </Grid>
                  </CardBody>
                </Card>
              </Grid>
            </TabPanel>
            
            {/* Technical Indicators Tab */}
            <TabPanel px={0}>
              <Grid templateColumns={{ base: '1fr', md: 'repeat(3, 1fr)' }} gap={4}>
                <Card bg={cardBg}>
                  <CardHeader>
                    <Heading size="sm">Momentum Indicators</Heading>
                  </CardHeader>
                  <CardBody>
                    <Grid templateColumns="repeat(2, 1fr)" gap={4}>
                      <Box>
                        <Text color="gray.500" fontSize="sm">RSI (14)</Text>
                        <Text fontWeight="bold">52.8</Text>
                        <Text fontSize="xs" color="gray.500">Neutral</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">MACD</Text>
                        <Text fontWeight="bold" color="green.500">0.75</Text>
                        <Text fontSize="xs" color="gray.500">Bullish</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">Stochastic</Text>
                        <Text fontWeight="bold">68.4</Text>
                        <Text fontSize="xs" color="gray.500">Neutral</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">ROC</Text>
                        <Text fontWeight="bold" color="green.500">2.1%</Text>
                        <Text fontSize="xs" color="gray.500">Bullish</Text>
                      </Box>
                    </Grid>
                  </CardBody>
                </Card>
                
                <Card bg={cardBg}>
                  <CardHeader>
                    <Heading size="sm">Trend Indicators</Heading>
                  </CardHeader>
                  <CardBody>
                    <Grid templateColumns="repeat(2, 1fr)" gap={4}>
                      <Box>
                        <Text color="gray.500" fontSize="sm">ADX</Text>
                        <Text fontWeight="bold">23.5</Text>
                        <Text fontSize="xs" color="gray.500">Trending</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">Ichimoku</Text>
                        <Text fontWeight="bold" color="green.500">Above Cloud</Text>
                        <Text fontSize="xs" color="gray.500">Bullish</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">Parabolic SAR</Text>
                        <Text fontWeight="bold" color="red.500">Above Price</Text>
                        <Text fontSize="xs" color="gray.500">Bearish</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">MA Direction</Text>
                        <Text fontWeight="bold" color="green.500">Upward</Text>
                        <Text fontSize="xs" color="gray.500">Bullish</Text>
                      </Box>
                    </Grid>
                  </CardBody>
                </Card>
                
                <Card bg={cardBg}>
                  <CardHeader>
                    <Heading size="sm">Volatility Indicators</Heading>
                  </CardHeader>
                  <CardBody>
                    <Grid templateColumns="repeat(2, 1fr)" gap={4}>
                      <Box>
                        <Text color="gray.500" fontSize="sm">Bollinger Width</Text>
                        <Text fontWeight="bold">2.1</Text>
                        <Text fontSize="xs" color="gray.500">Average</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">ATR Ratio</Text>
                        <Text fontWeight="bold" color="green.500">0.85</Text>
                        <Text fontSize="xs" color="gray.500">Low Volatility</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">Historical Vol</Text>
                        <Text fontWeight="bold">18.5%</Text>
                        <Text fontSize="xs" color="gray.500">Moderate</Text>
                      </Box>
                      <Box>
                        <Text color="gray.500" fontSize="sm">Keltner Width</Text>
                        <Text fontWeight="bold">1.9</Text>
                        <Text fontSize="xs" color="gray.500">Average</Text>
                      </Box>
                    </Grid>
                  </CardBody>
                </Card>
              </Grid>
            </TabPanel>
            
            {/* Patterns Tab */}
            <TabPanel px={0}>
              <Card bg={cardBg}>
                <CardHeader>
                  <Heading size="md">Chart Patterns</Heading>
                </CardHeader>
                <CardBody>
                  <Text textAlign="center" color="gray.500" py={8}>
                    No significant chart patterns detected in the current timeframe.
                  </Text>
                </CardBody>
              </Card>
            </TabPanel>
            
            {/* Sentiment Tab */}
            <TabPanel px={0}>
              <Card bg={cardBg}>
                <CardHeader>
                  <Heading size="md">Market Sentiment</Heading>
                </CardHeader>
                <CardBody>
                  <Text textAlign="center" color="gray.500" py={8}>
                    Sentiment analysis data will be displayed here.
                  </Text>
                </CardBody>
              </Card>
            </TabPanel>
          </TabPanels>
        </Tabs>
      )}
    </Box>
  );
};

export default Analysis;
