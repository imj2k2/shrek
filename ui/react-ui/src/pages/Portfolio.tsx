import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Flex,
  Grid,
  GridItem,
  Heading,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  useColorModeValue,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Badge,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  IconButton,
} from '@chakra-ui/react';
import { FiMoreVertical, FiTrendingUp, FiTrendingDown, FiDollarSign } from 'react-icons/fi';
import { useQuery } from '@tanstack/react-query';
import { 
  fetchPortfolioSummary, 
  fetchPortfolioPositions,
  fetchPortfolioHistory 
} from '../services/api';

const Portfolio: React.FC = () => {
  const [timeframe, setTimeframe] = useState('1M'); // Default to 1 month view
  
  // Fetch portfolio data
  const { data: summary, isLoading: summaryLoading } = useQuery(
    ['portfolioSummary'],
    fetchPortfolioSummary,
    { staleTime: 60000 } // 1 minute
  );
  
  const { data: positions, isLoading: positionsLoading } = useQuery(
    ['portfolioPositions'],
    fetchPortfolioPositions,
    { staleTime: 60000 } // 1 minute
  );
  
  const { data: history, isLoading: historyLoading } = useQuery(
    ['portfolioHistory', timeframe],
    () => fetchPortfolioHistory(timeframe),
    { staleTime: 60000 } // 1 minute
  );
  
  const cardBg = useColorModeValue('white', 'gray.700');
  
  return (
    <Box>
      <Flex justify="space-between" align="center" mb={6}>
        <Heading size="lg">Portfolio</Heading>
        <Button colorScheme="blue">Add Funds</Button>
      </Flex>
      
      {/* Portfolio Summary Stats */}
      <Grid templateColumns={{ base: '1fr', md: 'repeat(4, 1fr)' }} gap={4} mb={6}>
        <Card bg={cardBg}>
          <CardBody>
            <Stat>
              <StatLabel>Total Value</StatLabel>
              <StatNumber>
                ${summary?.totalValue.toLocaleString() || '0.00'}
              </StatNumber>
              <StatHelpText>
                <StatArrow type={summary?.dailyChange >= 0 ? 'increase' : 'decrease'} />
                {summary?.dailyChange || 0}% today
              </StatHelpText>
            </Stat>
          </CardBody>
        </Card>
        
        <Card bg={cardBg}>
          <CardBody>
            <Stat>
              <StatLabel>Cash Available</StatLabel>
              <StatNumber>
                ${summary?.cashAvailable.toLocaleString() || '0.00'}
              </StatNumber>
              <StatHelpText>
                {((summary?.cashAvailable / summary?.totalValue) * 100).toFixed(2) || 0}% of portfolio
              </StatHelpText>
            </Stat>
          </CardBody>
        </Card>
        
        <Card bg={cardBg}>
          <CardBody>
            <Stat>
              <StatLabel>Open Positions</StatLabel>
              <StatNumber>{positions?.length || 0}</StatNumber>
              <StatHelpText>
                {summary?.longPositions || 0} long, {summary?.shortPositions || 0} short
              </StatHelpText>
            </Stat>
          </CardBody>
        </Card>
        
        <Card bg={cardBg}>
          <CardBody>
            <Stat>
              <StatLabel>All-Time Return</StatLabel>
              <StatNumber>
                {summary?.allTimeReturn >= 0 ? '+' : ''}
                {summary?.allTimeReturn || 0}%
              </StatNumber>
              <StatHelpText>
                ${summary?.allTimeProfit.toLocaleString() || '0.00'} profit
              </StatHelpText>
            </Stat>
          </CardBody>
        </Card>
      </Grid>
      
      <Tabs isLazy variant="enclosed">
        <TabList>
          <Tab>Positions</Tab>
          <Tab>Orders</Tab>
          <Tab>History</Tab>
          <Tab>Performance</Tab>
        </TabList>
        
        <TabPanels>
          {/* Positions Tab */}
          <TabPanel px={0}>
            <Card>
              <CardHeader>
                <Heading size="md">Current Positions</Heading>
              </CardHeader>
              <CardBody>
                {positionsLoading ? (
                  <Text>Loading positions...</Text>
                ) : positions && positions.length > 0 ? (
                  <Box overflowX="auto">
                    <Table variant="simple">
                      <Thead>
                        <Tr>
                          <Th>Symbol</Th>
                          <Th>Type</Th>
                          <Th isNumeric>Qty</Th>
                          <Th isNumeric>Entry Price</Th>
                          <Th isNumeric>Current Price</Th>
                          <Th isNumeric>Market Value</Th>
                          <Th isNumeric>P&L</Th>
                          <Th isNumeric>P&L %</Th>
                          <Th></Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        {positions.map((position: any) => (
                          <Tr key={position.symbol}>
                            <Td fontWeight="bold">{position.symbol}</Td>
                            <Td>
                              <Badge colorScheme={position.type === 'long' ? 'green' : 'red'}>
                                {position.type.toUpperCase()}
                              </Badge>
                            </Td>
                            <Td isNumeric>{position.quantity}</Td>
                            <Td isNumeric>${position.entryPrice.toFixed(2)}</Td>
                            <Td isNumeric>${position.currentPrice.toFixed(2)}</Td>
                            <Td isNumeric>${position.marketValue.toFixed(2)}</Td>
                            <Td isNumeric color={position.profitLoss >= 0 ? 'green.500' : 'red.500'}>
                              ${position.profitLoss.toFixed(2)}
                            </Td>
                            <Td isNumeric color={position.profitLossPercent >= 0 ? 'green.500' : 'red.500'}>
                              {position.profitLossPercent >= 0 ? '+' : ''}
                              {position.profitLossPercent.toFixed(2)}%
                            </Td>
                            <Td>
                              <Menu>
                                <MenuButton
                                  as={IconButton}
                                  icon={<FiMoreVertical />}
                                  variant="ghost"
                                  size="sm"
                                  aria-label="More options"
                                />
                                <MenuList>
                                  <MenuItem icon={<FiDollarSign />}>Close Position</MenuItem>
                                  <MenuItem icon={position.type === 'long' ? <FiTrendingUp /> : <FiTrendingDown />}>
                                    Add to Position
                                  </MenuItem>
                                </MenuList>
                              </Menu>
                            </Td>
                          </Tr>
                        ))}
                      </Tbody>
                    </Table>
                  </Box>
                ) : (
                  <Text color="gray.500" textAlign="center" py={8}>
                    No open positions. Start trading to build your portfolio.
                  </Text>
                )}
              </CardBody>
            </Card>
          </TabPanel>
          
          {/* Orders Tab */}
          <TabPanel px={0}>
            <Card>
              <CardHeader>
                <Heading size="md">Recent Orders</Heading>
              </CardHeader>
              <CardBody>
                <Text color="gray.500" textAlign="center" py={8}>
                  No recent orders to display.
                </Text>
              </CardBody>
            </Card>
          </TabPanel>
          
          {/* History Tab */}
          <TabPanel px={0}>
            <Card>
              <CardHeader>
                <Heading size="md">Transaction History</Heading>
              </CardHeader>
              <CardBody>
                <Text color="gray.500" textAlign="center" py={8}>
                  No transaction history to display.
                </Text>
              </CardBody>
            </Card>
          </TabPanel>
          
          {/* Performance Tab */}
          <TabPanel px={0}>
            <Card>
              <CardHeader>
                <Heading size="md">Performance Metrics</Heading>
              </CardHeader>
              <CardBody>
                <Text color="gray.500" textAlign="center" py={8}>
                  Performance metrics will be displayed here.
                </Text>
              </CardBody>
            </Card>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Box>
  );
};

export default Portfolio;
