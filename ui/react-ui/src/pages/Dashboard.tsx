import React, { useState } from 'react';
import {
  Box,
  Grid,
  Heading,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Text,
  Flex,
  Button,
  Select,
  Badge,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  useColorModeValue,
} from '@chakra-ui/react';
import { useQuery } from '@tanstack/react-query';
import StockChart from '../components/StockChart';
import RecentSignals from '../components/RecentSignals';
import PortfolioSummary from '../components/PortfolioSummary';
import { fetchMarketSummary, fetchPortfolioSummary, fetchRecentSignals } from '../services/api';

const Dashboard: React.FC = () => {
  const [timeframe, setTimeframe] = useState('1D');
  const statBg = useColorModeValue('white', 'gray.700');
  
  // Fetch market summary
  const { data: marketData, isLoading: marketLoading } = useQuery(
    ['marketSummary'],
    fetchMarketSummary,
    { staleTime: 60000 } // 1 minute
  );
  
  // Fetch portfolio summary
  const { data: portfolioData, isLoading: portfolioLoading } = useQuery(
    ['portfolioSummary'],
    fetchPortfolioSummary,
    { staleTime: 300000 } // 5 minutes
  );
  
  // Fetch recent signals
  const { data: signalsData, isLoading: signalsLoading } = useQuery(
    ['recentSignals'],
    fetchRecentSignals,
    { staleTime: 60000 } // 1 minute
  );
  
  return (
    <Box>
      <Flex justify="space-between" align="center" mb={6}>
        <Heading size="lg">Dashboard</Heading>
        <Flex>
          <Select 
            value={timeframe} 
            onChange={(e) => setTimeframe(e.target.value)} 
            width="100px" 
            mr={3}
          >
            <option value="1D">1D</option>
            <option value="1W">1W</option>
            <option value="1M">1M</option>
            <option value="3M">3M</option>
            <option value="1Y">1Y</option>
          </Select>
          <Button colorScheme="blue">Refresh</Button>
        </Flex>
      </Flex>
      
      {/* Market Stats */}
      <SimpleGrid columns={{ base: 1, md: 4 }} spacing={4} mb={6}>
        {marketLoading ? (
          <Text>Loading market data...</Text>
        ) : marketData ? (
          <>
            <Stat bg={statBg} p={4} borderRadius="md" boxShadow="sm">
              <StatLabel>S&P 500</StatLabel>
              <StatNumber>{marketData.sp500.price.toFixed(2)}</StatNumber>
              <StatHelpText>
                <StatArrow type={marketData.sp500.change > 0 ? 'increase' : 'decrease'} />
                {marketData.sp500.change.toFixed(2)}%
              </StatHelpText>
            </Stat>
            
            <Stat bg={statBg} p={4} borderRadius="md" boxShadow="sm">
              <StatLabel>NASDAQ</StatLabel>
              <StatNumber>{marketData.nasdaq.price.toFixed(2)}</StatNumber>
              <StatHelpText>
                <StatArrow type={marketData.nasdaq.change > 0 ? 'increase' : 'decrease'} />
                {marketData.nasdaq.change.toFixed(2)}%
              </StatHelpText>
            </Stat>
            
            <Stat bg={statBg} p={4} borderRadius="md" boxShadow="sm">
              <StatLabel>Bitcoin</StatLabel>
              <StatNumber>${marketData.bitcoin.price.toLocaleString()}</StatNumber>
              <StatHelpText>
                <StatArrow type={marketData.bitcoin.change > 0 ? 'increase' : 'decrease'} />
                {marketData.bitcoin.change.toFixed(2)}%
              </StatHelpText>
            </Stat>
            
            <Stat bg={statBg} p={4} borderRadius="md" boxShadow="sm">
              <StatLabel>Market Status</StatLabel>
              <Flex mt={2}>
                <Badge 
                  colorScheme={marketData.marketStatus === 'Open' ? 'green' : 'red'} 
                  p={1} 
                  borderRadius="md"
                >
                  {marketData.marketStatus}
                </Badge>
              </Flex>
              <StatHelpText>
                {marketData.nextEvent}
              </StatHelpText>
            </Stat>
          </>
        ) : (
          <Text>Failed to load market data</Text>
        )}
      </SimpleGrid>
      
      {/* Main Content */}
      <Grid templateColumns={{ base: '1fr', lg: '2fr 1fr' }} gap={6}>
        <Box>
          <Card mb={6}>
            <CardHeader>
              <Heading size="md">Market Overview</Heading>
            </CardHeader>
            <CardBody>
              <StockChart timeframe={timeframe} />
            </CardBody>
          </Card>
          
          <Card>
            <CardHeader>
              <Heading size="md">Recent Signals</Heading>
            </CardHeader>
            <CardBody>
              <RecentSignals signals={signalsData || []} isLoading={signalsLoading} />
            </CardBody>
          </Card>
        </Box>
        
        <Box>
          <Card mb={6}>
            <CardHeader>
              <Heading size="md">Portfolio Summary</Heading>
            </CardHeader>
            <CardBody>
              <PortfolioSummary data={portfolioData} isLoading={portfolioLoading} />
            </CardBody>
          </Card>
          
          <Card>
            <CardHeader>
              <Heading size="md">Market Insights</Heading>
            </CardHeader>
            <CardBody>
              <Tabs isFitted variant="enclosed">
                <TabList mb="1em">
                  <Tab>News</Tab>
                  <Tab>Sentiment</Tab>
                  <Tab>Events</Tab>
                </TabList>
                <TabPanels>
                  <TabPanel>
                    <Text>Latest market news will appear here.</Text>
                  </TabPanel>
                  <TabPanel>
                    <Text>Market sentiment analysis will appear here.</Text>
                  </TabPanel>
                  <TabPanel>
                    <Text>Upcoming market events will appear here.</Text>
                  </TabPanel>
                </TabPanels>
              </Tabs>
            </CardBody>
          </Card>
        </Box>
      </Grid>
    </Box>
  );
};

export default Dashboard;
