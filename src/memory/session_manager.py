"""Session management for multi-agent conversations."""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from .agent_registry import AgentRegistryMemory


@dataclass
class Message:
    """Represents a message in a conversation."""
    role: str  # "user", "assistant", "system", "agent"
    content: str
    agent_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InMemorySessionService:
    """In-memory session service for managing conversation state."""
    
    def __init__(self, max_history_length: int = 50, enable_compaction: bool = True):
        """
        Initialize session service.
        
        Args:
            max_history_length: Maximum number of messages to keep
            enable_compaction: Whether to enable context compaction
        """
        self.sessions: Dict[str, List[Message]] = {}
        self.max_history_length = max_history_length
        self.enable_compaction = enable_compaction
    
    def create_session(self, session_id: str) -> None:
        """Create a new session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
    
    def add_message(self, session_id: str, message: Message) -> None:
        """Add a message to a session."""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        self.sessions[session_id].append(message)
        
        # Apply compaction if enabled and threshold exceeded
        if self.enable_compaction and len(self.sessions[session_id]) > self.max_history_length:
            self._compact_history(session_id)
    
    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Message]:
        """Get conversation history for a session."""
        if session_id not in self.sessions:
            return []
        
        history = self.sessions[session_id]
        if limit:
            return history[-limit:]
        return history
    
    def get_context(self, session_id: str, include_metadata: bool = False) -> str:
        """
        Get formatted context from session history.
        
        Args:
            session_id: Session identifier
            include_metadata: Whether to include metadata in context
            
        Returns:
            Formatted context string
        """
        history = self.get_history(session_id)
        context_parts = []
        
        for msg in history:
            role_prefix = f"[{msg.agent_name}]" if msg.agent_name else ""
            context_parts.append(f"{role_prefix}{msg.role}: {msg.content}")
            
            if include_metadata and msg.metadata:
                context_parts.append(f"  Metadata: {msg.metadata}")
        
        return "\n".join(context_parts)
    
    def _compact_history(self, session_id: str) -> None:
        """
        Compact session history by keeping only essential messages.
        Keeps first system message, last N user/assistant pairs.
        """
        history = self.sessions[session_id]
        
        if len(history) <= self.max_history_length:
            return
        
        # Keep system messages
        system_messages = [msg for msg in history if msg.role == "system"]
        
        # Keep recent messages (last 80% of max)
        recent_count = int(self.max_history_length * 0.8)
        recent_messages = history[-recent_count:]
        
        # Combine: system messages + recent messages
        compacted = system_messages + recent_messages
        
        # Remove duplicates while preserving order
        seen = set()
        unique_messages = []
        for msg in compacted:
            msg_key = (msg.role, msg.content, msg.agent_name)
            if msg_key not in seen:
                seen.add(msg_key)
                unique_messages.append(msg)
        
        self.sessions[session_id] = unique_messages
    
    def clear_session(self, session_id: str) -> None:
        """Clear a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]


class SessionManager:
    """High-level session manager."""
    
    def __init__(self, session_service: Optional[InMemorySessionService] = None):
        """
        Initialize session manager.
        
        Args:
            session_service: Optional session service (creates default if None)
        """
        self.session_service = session_service or InMemorySessionService()
        self.current_session_id: Optional[str] = None
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new session.
        
        Args:
            session_id: Optional session ID (generates if None)
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_service.create_session(session_id)
        self.current_session_id = session_id
        return session_id
    
    def add_user_message(self, content: str, session_id: Optional[str] = None) -> None:
        """Add a user message to the session."""
        sid = session_id or self.current_session_id
        if not sid:
            sid = self.start_session()
        
        message = Message(role="user", content=content)
        self.session_service.add_message(sid, message)
    
    def add_agent_message(
        self, 
        content: str, 
        agent_name: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an agent message to the session."""
        sid = session_id or self.current_session_id
        if not sid:
            sid = self.start_session()
        
        message = Message(
            role="agent",
            content=content,
            agent_name=agent_name,
            metadata=metadata or {}
        )
        self.session_service.add_message(sid, message)
    
    def get_context(self, session_id: Optional[str] = None) -> str:
        """Get formatted context for the session."""
        sid = session_id or self.current_session_id
        if not sid:
            return ""
        
        return self.session_service.get_context(sid)




