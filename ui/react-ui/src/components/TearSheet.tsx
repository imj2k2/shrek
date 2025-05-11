import React, { useEffect, useState } from "react";

// Define interfaces for our data structures
interface MetricsData {
  cagr?: number;
  sharpe?: number;
  sharpe_ratio?: number;
  max_drawdown?: number;
  win_rate?: number;
  sortino?: number;
  calmar?: number;
  calmar_ratio?: number;
  volatility?: number;
  beta?: number;
  alpha?: number;
  information_ratio?: number;
  kurtosis?: number;
  skew?: number;
  tail_ratio?: number;
  var?: number;
  best_month?: number;
  worst_month?: number;
  avg_win?: number;
  avg_loss?: number;
  profit_factor?: number;
  recovery_factor?: number;
  [key: string]: number | undefined;
}

interface ReturnsData {
  dates: string[];
  returns: number[];
  benchmark?: number[];
}

interface ReturnDataPoint {
  date: string;
  return: number;
  cumReturn: number;
  drawdown?: number;
  [key: string]: string | number | undefined;
}
import {
  Box,
  Card,
  CardBody,
  CardHeader,
  Grid,
  GridItem,
  Heading,
  Spinner,
  Text,
  Select,
  Flex,
} from "@chakra-ui/react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  Legend,
} from "recharts";
import api from "../services/api";

interface TearSheetProps {
  backtestId: string;
}

interface MetricsCardProps {
  title: string;
  value: number | string;
  precision?: number;
  format?: string; // 'percent', 'currency', etc.
}

const MetricsCard = ({ title, value, precision = 2, format = 'number' }: MetricsCardProps) => {
  let formattedValue = value;
  
  if (typeof value === 'number') {
    if (format === 'percent') {
      formattedValue = `${(value * 100).toFixed(precision)}%`;
    } else if (format === 'currency') {
      formattedValue = `$${value.toLocaleString(undefined, {
        minimumFractionDigits: precision,
        maximumFractionDigits: precision,
      })}`;
    } else {
      formattedValue = value.toFixed(precision);
    }
  }
  
  return (
    <Card>
      <CardBody p={4} textAlign="center">
        <Text fontSize="sm" fontWeight="medium" color="gray.500" textTransform="uppercase">
          {title}
        </Text>
        <Text fontSize="2xl" fontWeight="bold" mt={1}>
          {formattedValue}
        </Text>
      </CardBody>
    </Card>
  );
};

export const TearSheet = ({ backtestId }: TearSheetProps) => {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [returns, setReturns] = useState<ReturnsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [frequency, setFrequency] = useState<string>("D");

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const metricsResponse = await api.get(`/quantstats/metrics/${backtestId}`);
        const data = metricsResponse.data;
        
        // Check if response contains error message
        if (data.error) {
          console.error("QuantStats metrics error:", data.error, data.message);
          setError(data.message || data.error || "Failed to compute metrics. Insufficient data provided.");
          setMetrics(null);
          return;
        }
        
        // No error, set metrics
        setMetrics(data);
      } catch (err) {
        console.error("Error fetching metrics:", err);
        setError("Failed to load performance metrics");
      }
    };

    const fetchReturns = async () => {
      try {
        const returnsResponse = await api.get(`/quantstats/returns/${backtestId}?freq=${frequency}`);
        const data = returnsResponse.data;
        
        // Check if response contains error message
        if (data.error) {
          console.error("QuantStats returns error:", data.error);
          // Don't set error here if we already have metrics data
          if (!metrics) {
            setError(data.message || data.error || "Failed to compute returns. Insufficient trade data provided.");
          }
          setReturns(null);
        } else {
          setReturns(data);
        }
      } catch (err) {
        console.error("Error fetching returns:", err);
        if (!metrics) {
          setError("Failed to load return data");
        }
      } finally {
        setLoading(false);
      }
    };

    setLoading(true);
    setError(null); // Reset error state
    fetchMetrics();
    fetchReturns();
  }, [backtestId, frequency, metrics]);

  const handleFrequencyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setFrequency(e.target.value);
  };

  // Format cumulative returns data for charts
  const cumulativeReturnsData = React.useMemo<ReturnDataPoint[]>(() => {
    if (!returns || !returns.dates || !returns.returns) return [];

    let cumReturn = 0;
    return returns.dates.map((date: string, i: number) => {
      const returnValue = returns.returns[i];
      cumReturn = i === 0 ? returnValue : (1 + cumReturn) * (1 + returnValue) - 1;
      
      return {
        date,
        return: returnValue,
        cumReturn: cumReturn,
      };
    });
  }, [returns]);

  // Calculate drawdown data
  const drawdownData = React.useMemo<ReturnDataPoint[]>(() => {
    if (!cumulativeReturnsData || cumulativeReturnsData.length === 0) return [];
    
    let peak = -Infinity;
    
    return cumulativeReturnsData.map((item: ReturnDataPoint) => {
      if (item.cumReturn > peak) {
        peak = item.cumReturn;
      }
      
      const drawdown = item.cumReturn < peak ? (item.cumReturn / peak) - 1 : 0;
      
      return {
        ...item,
        drawdown: drawdown
      };
    });
  }, [cumulativeReturnsData]);

  if (loading) {
    return (
      <Flex justify="center" align="center" height="400px">
        <Spinner size="xl" />
      </Flex>
    );
  }

  if (error) {
    return (
      <Box p={6} textAlign="center">
        <Text color="red.500">{error}</Text>
        <Text mt={2}>
          Note: QuantStats analysis requires sufficient trade history to generate meaningful metrics.
        </Text>
      </Box>
    );
  }

  return (
    <Box>
      {/* Frequency Selector */}
      <Flex justify="flex-end" mb={4}>
        <Select
          value={frequency}
          onChange={handleFrequencyChange}
          width="200px"
        >
          <option value="D">Daily</option>
          <option value="W">Weekly</option>
          <option value="M">Monthly</option>
          <option value="Q">Quarterly</option>
          <option value="Y">Yearly</option>
        </Select>
      </Flex>
      
      {/* Key Metrics */}
      <Grid templateColumns="repeat(4, 1fr)" gap={4} mb={8}>
        <MetricsCard
          title="CAGR"
          value={metrics?.cagr || 0}
          format="percent"
        />
        <MetricsCard
          title="Sharpe Ratio"
          value={metrics?.sharpe || metrics?.sharpe_ratio || 0}
        />
        <MetricsCard
          title="Max Drawdown"
          value={metrics?.max_drawdown || 0}
          format="percent"
        />
        <MetricsCard
          title="Win Rate"
          value={metrics?.win_rate || 0}
          format="percent"
        />
      </Grid>

      {/* Secondary Metrics */}
      <Grid templateColumns="repeat(4, 1fr)" gap={4} mb={8}>
        <MetricsCard
          title="Sortino Ratio"
          value={metrics?.sortino || 0}
        />
        <MetricsCard
          title="Calmar Ratio"
          value={metrics?.calmar || metrics?.calmar_ratio || 0}
        />
        <MetricsCard
          title="Volatility"
          value={metrics?.volatility || 0}
          format="percent"
        />
        <MetricsCard
          title="Beta"
          value={metrics?.beta || 0}
        />
      </Grid>

      {/* Cumulative Returns Chart */}
      <Card mb={6}>
        <CardHeader>
          <Heading size="md">Cumulative Returns</Heading>
        </CardHeader>
        <CardBody>
          <Box height="300px">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={cumulativeReturnsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(date) => new Date(date).toLocaleDateString()}
                  tick={{ fontSize: 12 }}
                />
                <YAxis
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip
                  formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Return']}
                  labelFormatter={(label: string) => new Date(label).toLocaleDateString()}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="cumReturn"
                  name="Cumulative Return"
                  stroke="#4299E1"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </CardBody>
      </Card>

      {/* Drawdown Chart */}
      <Card mb={6}>
        <CardHeader>
          <Heading size="md">Drawdowns</Heading>
        </CardHeader>
        <CardBody>
          <Box height="200px">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={drawdownData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(date) => new Date(date).toLocaleDateString()}
                  tick={{ fontSize: 12 }}
                />
                <YAxis
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                  domain={[
                    (dataMin: number) => Math.min(dataMin, -0.01),
                    0
                  ]}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip
                  formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Drawdown']}
                  labelFormatter={(label: string) => new Date(label).toLocaleDateString()}
                />
                <Area
                  type="monotone"
                  dataKey="drawdown"
                  name="Drawdown"
                  fill="#F56565"
                  stroke="#E53E3E"
                  fillOpacity={0.5}
                />
              </AreaChart>
            </ResponsiveContainer>
          </Box>
        </CardBody>
      </Card>

      {/* Additional Stats */}
      {metrics && (
        <Card>
          <CardHeader>
            <Heading size="md">Additional Metrics</Heading>
          </CardHeader>
          <CardBody>
            <Grid templateColumns="repeat(3, 1fr)" gap={4}>
              <MetricsCard title="Alpha" value={metrics.alpha || 0} />
              <MetricsCard title="Information Ratio" value={metrics.information_ratio || 0} />
              <MetricsCard title="Kurtosis" value={metrics.kurtosis || 0} />
              <MetricsCard title="Skew" value={metrics.skew || 0} />
              <MetricsCard title="Tail Ratio" value={metrics.tail_ratio || 0} />
              <MetricsCard title="Value at Risk" value={metrics.var || 0} format="percent" />
              <MetricsCard title="Best Month" value={metrics.best_month || 0} format="percent" />
              <MetricsCard title="Worst Month" value={metrics.worst_month || 0} format="percent" />
              <MetricsCard title="Avg Win" value={metrics.avg_win || 0} format="percent" />
              <MetricsCard title="Avg Loss" value={metrics.avg_loss || 0} format="percent" />
              <MetricsCard title="Profit Factor" value={metrics.profit_factor || 0} />
              <MetricsCard title="Recovery Factor" value={metrics.recovery_factor || 0} />
            </Grid>
          </CardBody>
        </Card>
      )}
    </Box>
  );
};

export default TearSheet;
