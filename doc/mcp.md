# MCP (Model Context Protocol) Module Documentation

## Overview
The `mcp/` directory provides a protocol for structured communication and context sharing between agents. It enables message passing, context updates, and event subscription in a multi-agent trading system.

## Files & Purpose
- **protocol.py**: Implements the message format and protocol for agent communication.

---

## protocol.py

### Classes
- **MCPMessage**: Structured message format for agent communication.
    - `__init__(sender, message_type, content, timestamp=None)`: Initializes a message with sender, type (signal, execution, risk_alert, etc.), content, and timestamp.
    - `to_dict()`: Serializes to a dictionary.
    - `from_dict(data)`: Deserializes from a dictionary.
    - `to_json()`: Serializes to JSON string.

- **MCPProtocol**: Main protocol for context and message sharing among agents.
    - `__init__()`: Initializes context, message history, and subscribers.
    - `update_context(key, value)`: Updates a context value and notifies subscribers.
    - `get_context(key=None)`: Gets a specific context value or all context.
    - `send_message(message)`: Appends a message to history, updates context, and stores by type.
    - `get_messages(sender=None, message_type=None)`: Retrieves filtered message history.
    - `subscribe(key, callback)`: Subscribes a callback to context changes.
    - `unsubscribe(key, callback)`: Unsubscribes a callback.
    - `receive(payload)`: Receives a message payload, updates context.

---

## Design Notes
- **Extensible**: Supports arbitrary message types and agent communication patterns.
- **Context-aware**: Maintains both latest and historical context for each message type and sender.
- **Event-driven**: Allows agents to subscribe to context changes for reactive behavior.
- **Serialization**: Supports JSON and dictionary formats for easy transport and storage.

---

*Extend this document as the protocol is enhanced or new agent communication patterns are introduced.*
