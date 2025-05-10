import React, { createContext, useContext, useEffect, useState } from 'react';

// Define types
interface SocketContextType {
  isConnected: boolean;
  messages: any[];
  sendMessage: (type: string, payload: any) => void;
  subscribe: (channel: string) => void;
  unsubscribe: (channel: string) => void;
}

// Create context with default values
const SocketContext = createContext<SocketContextType>({
  isConnected: false,
  messages: [],
  sendMessage: () => {},
  subscribe: () => {},
  unsubscribe: () => {},
});

// Custom hook to use the socket context
export const useSocket = () => useContext(SocketContext);

interface SocketProviderProps {
  children: React.ReactNode;
}

// Provider component
export const SocketProvider: React.FC<SocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<any[]>([]);
  
  // Initialize WebSocket connection
  useEffect(() => {
    // Get WebSocket URL from environment or default to localhost
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = process.env.REACT_APP_WS_HOST || window.location.hostname;
    const wsPort = process.env.REACT_APP_WS_PORT || '8000';
    const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}/ws`;
    
    const newSocket = new WebSocket(wsUrl);
    
    newSocket.onopen = () => {
      console.log('WebSocket connection established');
      setIsConnected(true);
    };
    
    newSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setMessages((prev) => [...prev, data]);
        
        // If messages exceed 100, trim the array to prevent memory issues
        if (messages.length > 100) {
          setMessages((prev) => prev.slice(-100));
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    newSocket.onclose = () => {
      console.log('WebSocket connection closed');
      setIsConnected(false);
      
      // Try to reconnect after 5 seconds
      setTimeout(() => {
        console.log('Attempting to reconnect WebSocket...');
      }, 5000);
    };
    
    newSocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    setSocket(newSocket);
    
    // Clean up the WebSocket connection when the component unmounts
    return () => {
      if (newSocket.readyState === WebSocket.OPEN) {
        newSocket.close();
      }
    };
  }, []);
  
  // Send a message through the WebSocket
  const sendMessage = (type: string, payload: any) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({
        type,
        payload,
        timestamp: new Date().toISOString(),
      });
      socket.send(message);
    } else {
      console.error('WebSocket is not connected');
    }
  };
  
  // Subscribe to a specific channel
  const subscribe = (channel: string) => {
    sendMessage('subscribe', { channel });
  };
  
  // Unsubscribe from a specific channel
  const unsubscribe = (channel: string) => {
    sendMessage('unsubscribe', { channel });
  };
  
  // Context value
  const value = {
    isConnected,
    messages,
    sendMessage,
    subscribe,
    unsubscribe,
  };
  
  return <SocketContext.Provider value={value}>{children}</SocketContext.Provider>;
};
