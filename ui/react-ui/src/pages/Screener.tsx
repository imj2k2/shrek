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
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Select,
  Spinner,
  Switch,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  useToast,
  Radio,
  RadioGroup,
  Stack,
  Checkbox,
} from '@chakra-ui/react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { runStockScreener, getSavedScreeners, saveScreenerCriteria } from '../services/api';
import { useNavigate } from 'react-router-dom';

const Screener: React.FC = () => {
  const toast = useToast();
  const navigate = useNavigate();
  
  // Screener criteria state
  const [criteria, setCriteria] = useState({
    universe: 'SP500',
    minPrice: 10,
    maxPrice: 1000,
    minVolume: 100000,
    minVolatility: 0,
    maxVolatility: 100,
    minRSI: 0,
    maxRSI: 100,
    priceAboveSMA50: false,
    priceBelowSMA50: false,
    priceAboveSMA200: false,
    priceBelowSMA200: false,
    macdPositive: false,
    macdNegative: false,
    peMin: 0,
    peMax: 100,
    positionType: 'long', // Added position type (long/short)
  });
  
  // Screener results state
  const [results, setResults] = useState<any[]>([]);
  
  // Selected stocks for backtesting
  const [selectedStocks, setSelectedStocks] = useState<any[]>([]);
  
  // Get saved screeners
  const { data: savedScreeners } = useQuery(
    ['savedScreeners'],
    getSavedScreeners,
    { staleTime: 300000 } // 5 minutes
  );
  
  // Run screener mutation
  const { mutate: runScreener, isLoading: isScreening } = useMutation(
    () => runStockScreener(criteria),
    {
      onSuccess: (data) => {
        setResults(data.results || []);
        toast({
          title: 'Screener completed',
          description: `Found ${data.results?.length || 0} matching stocks`,
          status: 'success',
          duration: 5000,
          isClosable: true,
        });
      },
      onError: (error) => {
        toast({
          title: 'Error running screener',
          description: 'Failed to run stock screener. Please try again.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      },
    }
  );
  
  // Save screener mutation
  const { mutate: saveScreener } = useMutation(
    (name: string) => saveScreenerCriteria(name, criteria),
    {
      onSuccess: () => {
        toast({
          title: 'Screener saved',
          description: 'Your screener criteria have been saved successfully.',
          status: 'success',
          duration: 5000,
          isClosable: true,
        });
      },
    }
  );
  
  // Handle criteria change
  const handleCriteriaChange = (field: string, value: any) => {
    setCriteria((prev) => ({
      ...prev,
      [field]: value,
    }));
  };
  
  // Handle saving the screener
  const handleSaveScreener = () => {
    const name = prompt('Enter a name for this screener:');
    if (name) {
      saveScreener(name);
    }
  };
  
  // Handle stock selection for backtesting
  const handleStockSelection = (stock: any) => {
    const isSelected = selectedStocks.some((s) => s.symbol === stock.symbol);
    
    if (isSelected) {
      setSelectedStocks(selectedStocks.filter((s) => s.symbol !== stock.symbol));
    } else {
      // Include position type (long/short) with the selected stock
      setSelectedStocks([...selectedStocks, { 
        ...stock, 
        positionType: criteria.positionType  // Include the position type from criteria
      }]);
    }
  };
  
  // Start backtest with selected stocks
  const startBacktest = () => {
    if (selectedStocks.length === 0) {
      toast({
        title: 'No stocks selected',
        description: 'Please select at least one stock for backtesting.',
        status: 'warning',
        duration: 5000,
        isClosable: true,
      });
      return;
    }
    
    // Store selected stocks in localStorage for the backtest page
    localStorage.setItem('backtestStocks', JSON.stringify(selectedStocks));
    
    // Navigate to backtest page
    navigate('/backtest');
    
    toast({
      title: 'Stocks transferred to Backtest',
      description: `${selectedStocks.length} stocks ready for backtesting`,
      status: 'info',
      duration: 5000,
      isClosable: true,
    });
  };
  
  return (
    <Box>
      <Flex justify="space-between" align="center" mb={6}>
        <Heading size="lg">Stock Screener</Heading>
        <Flex gap={2}>
          <Button colorScheme="blue" onClick={handleSaveScreener}>Save Criteria</Button>
          <Button colorScheme="green" onClick={startBacktest} isDisabled={selectedStocks.length === 0}>
            Backtest Selected ({selectedStocks.length})
          </Button>
        </Flex>
      </Flex>
      
      <Grid templateColumns={{ base: '1fr', lg: '350px 1fr' }} gap={6}>
        {/* Screener Criteria */}
        <GridItem>
          <Card>
            <CardHeader>
              <Heading size="md">Screening Criteria</Heading>
            </CardHeader>
            <CardBody>
              <FormControl mb={4}>
                <FormLabel>Universe</FormLabel>
                <Select
                  value={criteria.universe}
                  onChange={(e) => handleCriteriaChange('universe', e.target.value)}
                >
                  <option value="SP500">S&P 500</option>
                  <option value="NASDAQ100">NASDAQ 100</option>
                  <option value="DOW30">DOW 30</option>
                  <option value="RUSSELL2000">Russell 2000</option>
                  <option value="ALL">All US Stocks</option>
                </Select>
              </FormControl>
              
              <FormControl mb={4}>
                <FormLabel>Position Type</FormLabel>
                <RadioGroup
                  value={criteria.positionType}
                  onChange={(value) => handleCriteriaChange('positionType', value)}
                >
                  <Stack direction="row">
                    <Radio value="long">Long</Radio>
                    <Radio value="short">Short</Radio>
                  </Stack>
                </RadioGroup>
              </FormControl>
              
              <Divider my={4} />
              
              <FormControl mb={4}>
                <FormLabel>Price Range ($)</FormLabel>
                <Flex gap={2}>
                  <NumberInput
                    min={0}
                    value={criteria.minPrice}
                    onChange={(_, value) => handleCriteriaChange('minPrice', value)}
                  >
                    <NumberInputField placeholder="Min" />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                  <NumberInput
                    min={0}
                    value={criteria.maxPrice}
                    onChange={(_, value) => handleCriteriaChange('maxPrice', value)}
                  >
                    <NumberInputField placeholder="Max" />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </Flex>
              </FormControl>
              
              <FormControl mb={4}>
                <FormLabel>Minimum Volume</FormLabel>
                <NumberInput
                  min={0}
                  step={10000}
                  value={criteria.minVolume}
                  onChange={(_, value) => handleCriteriaChange('minVolume', value)}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
              
              <FormControl mb={4}>
                <FormLabel>RSI Range</FormLabel>
                <Flex gap={2}>
                  <NumberInput
                    min={0}
                    max={100}
                    value={criteria.minRSI}
                    onChange={(_, value) => handleCriteriaChange('minRSI', value)}
                  >
                    <NumberInputField placeholder="Min" />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                  <NumberInput
                    min={0}
                    max={100}
                    value={criteria.maxRSI}
                    onChange={(_, value) => handleCriteriaChange('maxRSI', value)}
                  >
                    <NumberInputField placeholder="Max" />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </Flex>
              </FormControl>
              
              <Divider my={4} />
              
              <FormControl mb={4}>
                <FormLabel>Moving Averages</FormLabel>
                <Stack spacing={2}>
                  <Checkbox
                    isChecked={criteria.priceAboveSMA50}
                    onChange={(e) => handleCriteriaChange('priceAboveSMA50', e.target.checked)}
                  >
                    Price Above 50-day SMA
                  </Checkbox>
                  <Checkbox
                    isChecked={criteria.priceBelowSMA50}
                    onChange={(e) => handleCriteriaChange('priceBelowSMA50', e.target.checked)}
                  >
                    Price Below 50-day SMA
                  </Checkbox>
                  <Checkbox
                    isChecked={criteria.priceAboveSMA200}
                    onChange={(e) => handleCriteriaChange('priceAboveSMA200', e.target.checked)}
                  >
                    Price Above 200-day SMA
                  </Checkbox>
                  <Checkbox
                    isChecked={criteria.priceBelowSMA200}
                    onChange={(e) => handleCriteriaChange('priceBelowSMA200', e.target.checked)}
                  >
                    Price Below 200-day SMA
                  </Checkbox>
                </Stack>
              </FormControl>
              
              <FormControl mb={4}>
                <FormLabel>MACD</FormLabel>
                <Stack spacing={2}>
                  <Checkbox
                    isChecked={criteria.macdPositive}
                    onChange={(e) => handleCriteriaChange('macdPositive', e.target.checked)}
                  >
                    MACD is Positive
                  </Checkbox>
                  <Checkbox
                    isChecked={criteria.macdNegative}
                    onChange={(e) => handleCriteriaChange('macdNegative', e.target.checked)}
                  >
                    MACD is Negative
                  </Checkbox>
                </Stack>
              </FormControl>
              
              <Button
                colorScheme="blue"
                width="full"
                mt={4}
                onClick={() => runScreener()}
                isLoading={isScreening}
              >
                Run Screener
              </Button>
              
              {savedScreeners && savedScreeners.length > 0 && (
                <FormControl mt={6}>
                  <FormLabel>Saved Screeners</FormLabel>
                  <Select
                    placeholder="Select a saved screener"
                    onChange={(e) => {
                      const selectedScreener = savedScreeners.find(
                        (s: any) => s.name === e.target.value
                      );
                      if (selectedScreener) {
                        setCriteria(selectedScreener.criteria);
                      }
                    }}
                  >
                    {savedScreeners.map((screener: any) => (
                      <option key={screener.name} value={screener.name}>
                        {screener.name}
                      </option>
                    ))}
                  </Select>
                </FormControl>
              )}
            </CardBody>
          </Card>
        </GridItem>
        
        {/* Screener Results */}
        <GridItem>
          <Card>
            <CardHeader>
              <Heading size="md">Results</Heading>
            </CardHeader>
            <CardBody>
              {isScreening ? (
                <Flex justify="center" my={8}>
                  <Spinner size="xl" />
                </Flex>
              ) : results.length > 0 ? (
                <Box overflowX="auto">
                  <Table variant="simple" size="sm">
                    <Thead>
                      <Tr>
                        <Th></Th>
                        <Th>Symbol</Th>
                        <Th>Name</Th>
                        <Th isNumeric>Price ($)</Th>
                        <Th isNumeric>Change (%)</Th>
                        <Th isNumeric>Volume</Th>
                        <Th isNumeric>Market Cap</Th>
                        <Th isNumeric>RSI</Th>
                        <Th>Position</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {results.map((stock) => (
                        <Tr key={stock.symbol}>
                          <Td>
                            <Checkbox
                              isChecked={selectedStocks.some(
                                (s) => s.symbol === stock.symbol
                              )}
                              onChange={() => handleStockSelection(stock)}
                            />
                          </Td>
                          <Td fontWeight="bold">{stock.symbol}</Td>
                          <Td>{stock.name}</Td>
                          <Td isNumeric>${stock.price.toFixed(2)}</Td>
                          <Td isNumeric color={stock.change >= 0 ? 'green.500' : 'red.500'}>
                            {stock.change >= 0 ? '+' : ''}
                            {stock.change.toFixed(2)}%
                          </Td>
                          <Td isNumeric>{stock.volume.toLocaleString()}</Td>
                          <Td isNumeric>
                            {(stock.marketCap / 1e9).toFixed(1)}B
                          </Td>
                          <Td isNumeric>{stock.rsi.toFixed(1)}</Td>
                          <Td>{criteria.positionType.toUpperCase()}</Td>
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                </Box>
              ) : (
                <Flex justify="center" my={8}>
                  <Text color="gray.500">
                    No results to display. Run the screener to see results.
                  </Text>
                </Flex>
              )}
            </CardBody>
          </Card>
        </GridItem>
      </Grid>
    </Box>
  );
};

export default Screener;
