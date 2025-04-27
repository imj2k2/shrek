import json
from typing import Dict, Any, List, Optional
from datetime import datetime

class MCPMessage:
    """Structured message format for agent communication"""
    def __init__(self, sender: str, message_type: str, content: Dict[str, Any], timestamp: Optional[datetime] = None):
        self.sender = sender
        self.message_type = message_type  # signal, execution, risk_alert, etc.
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        timestamp = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else None
        return cls(
            sender=data["sender"],
            message_type=data["message_type"],
            content=data["content"],
            timestamp=timestamp
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

class MCPProtocol:
    """Model Context Protocol for agent communication and context sharing"""
    def __init__(self):
        self.context = {}
        self.message_history: List[MCPMessage] = []
        self.subscribers = {}
    
    def update_context(self, key: str, value: Any) -> None:
        """Update a specific context value"""
        self.context[key] = value
        # Notify subscribers for this context key
        if key in self.subscribers:
            for callback in self.subscribers[key]:
                callback(key, value)
    
    def get_context(self, key: Optional[str] = None) -> Any:
        """Get context value(s)"""
        if key:
            return self.context.get(key)
        return self.context
    
    def send_message(self, message: MCPMessage) -> None:
        """Send a message to all agents through the protocol"""
        self.message_history.append(message)
        # Update context with the latest message of each type from each sender
        context_key = f"latest_{message.message_type}_{message.sender}"
        self.update_context(context_key, message.content)
        # Also store in a list of all messages of this type
        list_key = f"all_{message.message_type}"
        if list_key not in self.context:
            self.context[list_key] = []
        self.context[list_key].append(message.content)
    
    def get_messages(self, sender: Optional[str] = None, message_type: Optional[str] = None) -> List[MCPMessage]:
        """Get filtered message history"""
        if not sender and not message_type:
            return self.message_history
        
        filtered = self.message_history
        if sender:
            filtered = [m for m in filtered if m.sender == sender]
        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]
        return filtered
    
    def subscribe(self, key: str, callback) -> None:
        """Subscribe to context changes"""
        if key not in self.subscribers:
            self.subscribers[key] = []
        self.subscribers[key].append(callback)
    
    def unsubscribe(self, key: str, callback) -> None:
        """Unsubscribe from context changes"""
        if key in self.subscribers and callback in self.subscribers[key]:
            self.subscribers[key].remove(callback)
    def receive(self, payload: str):
        # Deserialize and update context
        msg = json.loads(payload)
        self.context[msg['recipient']] = msg['message']
        return msg
