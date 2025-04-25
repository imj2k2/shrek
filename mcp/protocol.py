import json
from typing import Dict, Any

class MCPProtocol:
    """Model Context Protocol for structured agent communication."""
    def __init__(self):
        self.context = {}
    def send(self, sender: str, recipient: str, message: Dict[str, Any]):
        # Serialize message as JSON for transmission
        payload = json.dumps({
            'sender': sender,
            'recipient': recipient,
            'message': message
        })
        # In a real system, this would be sent over a message bus or socket
        return payload
    def receive(self, payload: str):
        # Deserialize and update context
        msg = json.loads(payload)
        self.context[msg['recipient']] = msg['message']
        return msg
