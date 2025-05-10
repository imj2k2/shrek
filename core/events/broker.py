"""
Message broker implementation for the Shrek Trading Platform.
This module provides a message broker abstraction for implementing
event-driven architecture with pub/sub patterns.
"""

import json
import logging
import asyncio
from typing import Dict, List, Callable, Any, Optional, Union
from datetime import datetime
from enum import Enum
import redis
from redis import Redis

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types for the message broker."""
    MARKET_DATA = "market_data"
    TRADE_SIGNAL = "trade_signal"
    PORTFOLIO_UPDATE = "portfolio_update"
    STRATEGY_UPDATE = "strategy_update"
    BACKTEST_RESULT = "backtest_result"
    NEWS_SENTIMENT = "news_sentiment"
    SYSTEM_NOTIFICATION = "system_notification"


class Event:
    """Base event class for the message broker."""
    
    def __init__(
        self, 
        event_type: EventType,
        payload: Dict[str, Any],
        source: str,
        timestamp: Optional[datetime] = None
    ):
        self.event_type = event_type
        self.payload = payload
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.id = f"{self.event_type.value}_{self.timestamp.isoformat()}_{id(self)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert the event to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create an event from a dictionary."""
        return cls(
            event_type=EventType(data["event_type"]),
            payload=data["payload"],
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Create an event from a JSON string."""
        return cls.from_dict(json.loads(json_str))


class MessageBroker:
    """
    Message broker implementation using Redis.
    This provides a pub/sub mechanism for event-driven communication.
    """
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.pubsub = self.redis.pubsub()
        self.subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self._running = False
        self._listener_task = None
    
    def publish(self, event: Event) -> bool:
        """
        Publish an event to the message broker.
        
        Args:
            event: The event to publish
            
        Returns:
            bool: True if the event was published successfully
        """
        try:
            self.redis.publish(event.event_type.value, event.to_json())
            # Also store the event in a time-series list for replay/history
            self.redis.lpush(f"history:{event.event_type.value}", event.to_json())
            # Trim the list to keep only recent events (e.g., last 1000)
            self.redis.ltrim(f"history:{event.event_type.value}", 0, 999)
            logger.debug(f"Published event: {event.id} of type {event.event_type.value}")
            return True
        except Exception as e:
            logger.error(f"Error publishing event: {str(e)}")
            return False
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: The type of events to subscribe to
            callback: The function to call when an event is received
        """
        channel = event_type.value
        if channel not in self.subscribers:
            self.subscribers[channel] = []
            self.pubsub.subscribe(channel)
        
        self.subscribers[channel].append(callback)
        logger.debug(f"Subscribed to {channel} events")
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: The type of events to unsubscribe from
            callback: The callback function to remove
        """
        channel = event_type.value
        if channel in self.subscribers:
            self.subscribers[channel] = [cb for cb in self.subscribers[channel] if cb != callback]
            if not self.subscribers[channel]:
                self.pubsub.unsubscribe(channel)
                del self.subscribers[channel]
    
    async def start_listening(self) -> None:
        """Start listening for events in the background."""
        if self._running:
            return
        
        self._running = True
        self._listener_task = asyncio.create_task(self._listener_loop())
        logger.info("Message broker listener started")
    
    async def stop_listening(self) -> None:
        """Stop listening for events."""
        if not self._running:
            return
        
        self._running = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
        
        logger.info("Message broker listener stopped")
    
    async def _listener_loop(self) -> None:
        """Background task that listens for events."""
        while self._running:
            try:
                message = self.pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    channel = message["channel"].decode("utf-8")
                    if channel in self.subscribers:
                        try:
                            event = Event.from_json(message["data"].decode("utf-8"))
                            for callback in self.subscribers[channel]:
                                # Run callbacks in separate tasks to prevent blocking
                                asyncio.create_task(self._run_callback(callback, event))
                        except Exception as e:
                            logger.error(f"Error processing message: {str(e)}")
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in listener loop: {str(e)}")
                await asyncio.sleep(1)  # Longer sleep on error
    
    async def _run_callback(self, callback: Callable[[Event], None], event: Event) -> None:
        """Run a callback function with error handling."""
        try:
            # Support both async and sync callbacks
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            logger.error(f"Error in event callback: {str(e)}")
    
    def get_event_history(self, event_type: EventType, limit: int = 100) -> List[Event]:
        """
        Get the history of events of a specific type.
        
        Args:
            event_type: The type of events to get
            limit: The maximum number of events to return
            
        Returns:
            List[Event]: A list of events
        """
        try:
            history = self.redis.lrange(f"history:{event_type.value}", 0, limit - 1)
            return [Event.from_json(item.decode("utf-8")) for item in history]
        except Exception as e:
            logger.error(f"Error getting event history: {str(e)}")
            return []


def create_broker(redis_url: str = "redis://localhost:6379/0") -> MessageBroker:
    """
    Create a message broker instance.
    
    Args:
        redis_url: The Redis connection URL
        
    Returns:
        MessageBroker: A message broker instance
    """
    redis_client = redis.from_url(redis_url)
    return MessageBroker(redis_client)
