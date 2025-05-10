import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Box,
  Flex,
  VStack,
  Icon,
  Text,
  Heading,
  Divider,
  Tooltip,
} from '@chakra-ui/react';
import {
  FiHome,
  FiSearch,
  FiActivity,
  FiBarChart2,
  FiSettings,
  FiHelpCircle,
  FiPieChart,
} from 'react-icons/fi';

interface NavItemProps {
  icon: React.ReactElement;
  path: string;
  label: string;
}

const NavItem: React.FC<NavItemProps> = ({ icon, path, label }) => {
  const location = useLocation();
  const isActive = location.pathname === path;

  return (
    <Tooltip label={label} placement="right" hasArrow>
      <Box as={NavLink} to={path} w="100%">
        <Flex
          align="center"
          py={3}
          px={4}
          mx={2}
          borderRadius="md"
          role="group"
          cursor="pointer"
          _hover={{
            bg: 'blue.50',
            color: 'blue.500',
          }}
          bg={isActive ? 'blue.50' : 'transparent'}
          color={isActive ? 'blue.500' : 'gray.600'}
          transition="all 0.2s"
        >
          {React.cloneElement(icon as React.ReactElement, {
            size: 20,
          })}
          <Text ml={4} fontWeight={isActive ? 'bold' : 'normal'}>
            {label}
          </Text>
        </Flex>
      </Box>
    </Tooltip>
  );
};

const Navigation: React.FC = () => {
  return (
    <Box
      as="nav"
      bg="white"
      w="250px"
      h="100vh"
      py={5}
      boxShadow="sm"
      display={{ base: 'none', md: 'block' }}
    >
      <Flex px={6} mb={6} alignItems="center">
        <Heading size="md" color="blue.600">
          Shrek Trading
        </Heading>
      </Flex>
      <Divider mb={6} />
      <VStack spacing={1} align="stretch">
        <NavItem 
          icon={<Icon as={FiHome} />} 
          path="/" 
          label="Dashboard" 
        />
        <NavItem
          icon={<Icon as={FiSearch} />}
          path="/screener"
          label="Stock Screener"
        />
        <NavItem
          icon={<Icon as={FiActivity} />}
          path="/backtest"
          label="Backtest"
        />
        <NavItem
          icon={<Icon as={FiPieChart} />}
          path="/portfolio"
          label="Portfolio"
        />
        <NavItem
          icon={<Icon as={FiBarChart2} />}
          path="/analytics"
          label="Analytics"
        />
        <Divider my={4} />
        <NavItem
          icon={<Icon as={FiSettings} />}
          path="/settings"
          label="Settings"
        />
        <NavItem
          icon={<Icon as={FiHelpCircle} />}
          path="/help"
          label="Help & Documentation"
        />
      </VStack>
    </Box>
  );
};

export default Navigation;
