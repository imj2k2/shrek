import React, { useState, useEffect } from 'react';
import {
  Box, Button, Card, CardBody, CardHeader, Divider, Flex, FormControl,
  FormLabel, Grid, GridItem, Heading, Input, Select, Spinner, Switch, Table,
  Thead, Tbody, Tr, Th, Td, Text, useToast, NumberInput, NumberInputField,
  NumberInputStepper, NumberIncrementStepper, NumberDecrementStepper,
  Badge, Radio, RadioGroup, Stack, Tab, TabList, TabPanel, TabPanels, Tabs
} from '@chakra-ui/react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { runBacktest, getBacktestResults } from '../services/api';
import { FiX, FiInfo } from 'react-icons/fi';

const Backtest: React.FC = () => {
  const toast = useToast();
  
  // State for backtest parameters
  const [params, setParams] = useState({
    symbols: [] as { symbol: string, positionType: string }[],
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    initialCapital: 100000,
    strategy: 'momentum',
    positionsPerSymbol: 1,
    maxPositions: 5,
    stopLoss: 5, // percentage
    takeProfit: 15, // percentage
    timeframe: 'day',
    agentType: 'stocks',
    useMockData: false, // option to use mock data when real data isn't available
  });
  
  // State for backtest results
  const [backtestId, setBacktestId] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [usingMockData, setUsingMockData] = useState(false);
  
  // Load symbols from localStorage if they exist (transferred from screener)
  useEffect(() => {
    const storedStocks = localStorage.getItem('backtestStocks');
    if (storedStocks) {
      try {
        const parsedStocks = JSON.parse(storedStocks);
        setParams((prev) => ({
          ...prev,
          symbols: parsedStocks,
        }));
        // Clear the storage to avoid reloading on page refresh
        localStorage.removeItem('backtestStocks');
        
        toast({
          title: 'Stocks loaded from Screener',
          description: `${parsedStocks.length} stocks loaded for backtesting`,
          status: 'info',
          duration: 3000,
          isClosable: true,
        });
      } catch (error) {
        console.error('Failed to parse stored stocks:', error);
      }
    }
  }, [toast]);
  
  // Run backtest mutation
  const { mutate: runBacktestMutation } = useMutation(
    () => {
      setIsRunning(true);
      return runBacktest(params);
    },
    {
      onSuccess: (data) => {
        console.log('Backtest API response:', data); // Debug log
        setIsRunning(false);
        
        // Check if we got error from the API
        if (data.error) {
          toast({
            title: 'Backtest Error',
            description: data.error,
            status: 'error',
            duration: 5000,
            isClosable: true,
          });
          return;
        }
        
        // Store the result directly
        setResults(data);
        
        // Check for mock data usage
        if (data.data_sources && Object.values(data.data_sources).some(source => source === 'mock')) {
          setUsingMockData(true);
          toast({
            title: 'Backtest completed with mock data',
            description: 'Real market data was not available for some symbols, so mock data was used instead.',
            status: 'warning',
            duration: 5000,
            isClosable: true,
          });
        } else {
          setUsingMockData(false);
          toast({
            title: 'Backtest completed',
            status: 'success',
            duration: 3000,
            isClosable: true,
          });
        }
      },
      onError: (error) => {
        setIsRunning(false);
        toast({
          title: 'Error running backtest',
          description: 'Failed to run backtest. Please try again.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      },
    }
  );
  
  // Note: We're now using the 'results' state directly from the runBacktest mutation
  // Previously, we were using a separate query but this caused naming conflicts
  const resultsLoading = isRunning;
  
  // Handle adding a symbol
  const addSymbol = () => {
    const symbol = prompt('Enter stock symbol:');
    if (symbol) {
      // Default to long position for manually added symbols
      setParams({
        ...params,
        symbols: [...params.symbols, { symbol: symbol.toUpperCase(), positionType: 'long' }],
      });
    }
  };
  
  // Handle removing a symbol
  const removeSymbol = (index: number) => {
    const newSymbols = [...params.symbols];
    newSymbols.splice(index, 1);
    setParams({ ...params, symbols: newSymbols });
  };
  
  // Toggle position type (long/short) for a symbol
  const togglePositionType = (index: number) => {
    const newSymbols = [...params.symbols];
    newSymbols[index].positionType = 
      newSymbols[index].positionType === 'long' ? 'short' : 'long';
    setParams({ ...params, symbols: newSymbols });
  };
  
  // Update backtest parameter
  const updateParam = (field: string, value: any) => {
    setParams({ ...params, [field]: value });
  };
  
  return (
    <Box>
      <Flex justify="space-between" align="center" mb={6}>
        <Heading size="lg">Backtest</Heading>
        <Button
          colorScheme="blue"
          onClick={() => runBacktestMutation()}
          isLoading={isRunning}
          isDisabled={params.symbols.length === 0}
        >
          Run Backtest
        </Button>
      </Flex>
      
      <Grid templateColumns={{ base: '1fr', lg: '1fr 2fr' }} gap={6}>
        {/* Backtest Parameters */}
        <GridItem>
          <Card mb={6}>
            <CardHeader>
              <Heading size="md">Parameters</Heading>
            </CardHeader>
            <CardBody>
              <FormControl mb={4}>
                <FormLabel>Date Range</FormLabel>
                <Flex gap={2}>
                  <Input
                    type="date"
                    value={params.startDate}
                    onChange={(e) => updateParam('startDate', e.target.value)}
                  />
                  <Input
                    type="date"
                    value={params.endDate}
                    onChange={(e) => updateParam('endDate', e.target.value)}
                  />
                </Flex>
              </FormControl>
              
              <FormControl display="flex" alignItems="center">
                <FormLabel htmlFor="useMockData" mb="0">
                  Use Mock Data When Needed
                </FormLabel>
                <Switch
                  id="useMockData"
                  isChecked={params.useMockData}
                  onChange={(e) => updateParam('useMockData', e.target.checked)}
                />
              </FormControl>
              
              <FormControl mb={4}>
                <FormLabel>Initial Capital ($)</FormLabel>
                <NumberInput
                  value={params.initialCapital}
                  min={1000}
                  step={1000}
                  onChange={(_, value) => updateParam('initialCapital', value)}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
              
              <FormControl mb={4}>
                <FormLabel>Strategy</FormLabel>
                <Select
                  value={params.strategy}
                  onChange={(e) => updateParam('strategy', e.target.value)}
                >
                  <option value="momentum">Momentum</option>
                  <option value="mean_reversion">Mean Reversion</option>
                  <option value="breakout">Breakout</option>
                </Select>
              </FormControl>
              
              <FormControl mb={4}>
                <FormLabel>Risk Management</FormLabel>
                <Flex gap={4}>
                  <Box flex="1">
                    <Text fontSize="sm" mb={1}>Stop Loss (%)</Text>
                    <NumberInput
                      value={params.stopLoss}
                      min={0}
                      max={50}
                      onChange={(_, value) => updateParam('stopLoss', value)}
                    >
                      <NumberInputField />
                      <NumberInputStepper>
                        <NumberIncrementStepper />
                        <NumberDecrementStepper />
                      </NumberInputStepper>
                    </NumberInput>
                  </Box>
                  <Box flex="1">
                    <Text fontSize="sm" mb={1}>Take Profit (%)</Text>
                    <NumberInput
                      value={params.takeProfit}
                      min={0}
                      max={100}
                      onChange={(_, value) => updateParam('takeProfit', value)}
                    >
                      <NumberInputField />
                      <NumberInputStepper>
                        <NumberIncrementStepper />
                        <NumberDecrementStepper />
                      </NumberInputStepper>
                    </NumberInput>
                  </Box>
                </Flex>
              </FormControl>
              
              <Divider my={4} />
              
              <Flex justify="space-between" align="center" mb={2}>
                <Text fontWeight="bold">Symbols</Text>
                <Button size="sm" onClick={addSymbol}>Add Symbol</Button>
              </Flex>
              
              {params.symbols.length > 0 ? (
                <Table size="sm" variant="simple">
                  <Thead>
                    <Tr>
                      <Th>Symbol</Th>
                      <Th>Position Type</Th>
                      <Th></Th>
                    </Tr>
                  </Thead>
                  <Tbody>
                    {params.symbols.map((item, index) => (
                      <Tr key={index}>
                        <Td>{item.symbol}</Td>
                        <Td>
                          <Badge
                            colorScheme={item.positionType === 'long' ? 'green' : 'red'}
                            cursor="pointer"
                            onClick={() => togglePositionType(index)}
                          >
                            {item.positionType.toUpperCase()}
                          </Badge>
                        </Td>
                        <Td textAlign="right">
                          <Button
                            size="xs"
                            variant="ghost"
                            colorScheme="red"
                            onClick={() => removeSymbol(index)}
                          >
                            <FiX />
                          </Button>
                        </Td>
                      </Tr>
                    ))}
                  </Tbody>
                </Table>
              ) : (
                <Text color="gray.500" fontSize="sm" textAlign="center" py={4}>
                  No symbols added. Add symbols manually or import from the Stock Screener.
                </Text>
              )}
            </CardBody>
          </Card>
        </GridItem>
        
        {/* Backtest Results */}
        <GridItem>
          {usingMockData && (
            <Box mb={4} p={4} bg="yellow.100" borderRadius="md">
              <Flex align="center">
                <Box color="yellow.800" mr={2}>
                  <FiInfo size={24} />
                </Box>
                <Box>
                  <Text fontWeight="bold" color="yellow.800">Mock Data Used</Text>
                  <Text color="yellow.800">
                    This backtest is using generated mock data instead of real market data. Results may not accurately reflect actual market behavior.
                  </Text>
                </Box>
              </Flex>
            </Box>
          )}
          
          {isRunning ? (
            <Flex direction="column" align="center" justify="center" minH="400px">
              <Spinner size="xl" mb={4} />
              <Text>Running backtest...</Text>
            </Flex>
          ) : backtestId && results ? (
            <Tabs isLazy>
              <TabList>
                <Tab>Summary</Tab>
                <Tab>Performance</Tab>
                <Tab>Trades</Tab>
                <Tab>Charts</Tab>
              </TabList>
              
              <TabPanels>
                {/* Summary Tab */}
                <TabPanel>
                  <Grid templateColumns="repeat(2, 1fr)" gap={6}>
                    <Card>
                      <CardHeader>
                        <Heading size="sm">Key Metrics</Heading>
                      </CardHeader>
                      <CardBody>
                        <Grid templateColumns="repeat(2, 1fr)" gap={4}>
                          <Box>
                            <Text color="gray.500" fontSize="sm">Total Return</Text>
                            <Text fontSize="xl" fontWeight="bold" color={results.totalReturn >= 0 ? 'green.500' : 'red.500'}>
                              {results.totalReturn.toFixed(2)}%
                            </Text>
                          </Box>
                          <Box>
                            <Text color="gray.500" fontSize="sm">Sharpe Ratio</Text>
                            <Text fontSize="xl" fontWeight="bold">
                              {results.sharpeRatio.toFixed(2)}
                            </Text>
                          </Box>
                          <Box>
                            <Text color="gray.500" fontSize="sm">Max Drawdown</Text>
                            <Text fontSize="xl" fontWeight="bold" color="red.500">
                              {results.maxDrawdown.toFixed(2)}%
                            </Text>
                          </Box>
                          <Box>
                            <Text color="gray.500" fontSize="sm">Win Rate</Text>
                            <Text fontSize="xl" fontWeight="bold">
                              {results.winRate.toFixed(2)}%
                            </Text>
                          </Box>
                        </Grid>
                      </CardBody>
                    </Card>
                    
                    <Card>
                      <CardHeader>
                        <Heading size="sm">Portfolio Value</Heading>
                      </CardHeader>
                      <CardBody>
                        <Grid templateColumns="repeat(2, 1fr)" gap={4}>
                          <Box>
                            <Text color="gray.500" fontSize="sm">Initial Capital</Text>
                            <Text fontSize="xl" fontWeight="bold">
                              ${params.initialCapital.toLocaleString()}
                            </Text>
                          </Box>
                          <Box>
                            <Text color="gray.500" fontSize="sm">Final Capital</Text>
                            <Text fontSize="xl" fontWeight="bold" color={results.finalCapital >= params.initialCapital ? 'green.500' : 'red.500'}>
                              ${results.finalCapital.toLocaleString()}
                            </Text>
                          </Box>
                          <Box>
                            <Text color="gray.500" fontSize="sm">Profit/Loss</Text>
                            <Text fontSize="xl" fontWeight="bold" color={results.finalCapital - params.initialCapital >= 0 ? 'green.500' : 'red.500'}>
                              ${(results.finalCapital - params.initialCapital).toLocaleString()}
                            </Text>
                          </Box>
                          <Box>
                            <Text color="gray.500" fontSize="sm">Annualized Return</Text>
                            <Text fontSize="xl" fontWeight="bold" color={results.annualizedReturn >= 0 ? 'green.500' : 'red.500'}>
                              {results.annualizedReturn.toFixed(2)}%
                            </Text>
                          </Box>
                        </Grid>
                      </CardBody>
                    </Card>
                  </Grid>
                </TabPanel>
                
                {/* Performance Tab */}
                <TabPanel>
                  <Text>Performance metrics will be displayed here</Text>
                </TabPanel>
                
                {/* Trades Tab */}
                <TabPanel>
                  <Text>Trades history will be displayed here</Text>
                </TabPanel>
                
                {/* Charts Tab */}
                <TabPanel>
                  <Text>Performance charts will be displayed here</Text>
                </TabPanel>
              </TabPanels>
            </Tabs>
          ) : (
            <Flex direction="column" align="center" justify="center" minH="400px">
              <Text color="gray.500">
                Configure parameters and run a backtest to see results
              </Text>
            </Flex>
          )}
        </GridItem>
      </Grid>
    </Box>
  );
};

export default Backtest;
