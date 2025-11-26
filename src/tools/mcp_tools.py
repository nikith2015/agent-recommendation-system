"""MCP (Model Context Protocol) tools for agent interoperability."""

from typing import List, Dict, Any, Optional
from ..models.agent_spec import ToolRequirement


class MCPToolRegistry:
    """Registry for MCP tools that agents can use."""
    
    def __init__(self):
        """Initialize MCP tool registry."""
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_tools()
    
    def _initialize_default_tools(self) -> None:
        """Initialize default MCP tools."""
        default_tools = [
            {
                "name": "google_search",
                "type": "built-in",
                "description": "Search the web using Google",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"}
                }
            },
            {
                "name": "code_execution",
                "type": "built-in",
                "description": "Execute Python code",
                "parameters": {
                    "code": {"type": "string", "description": "Python code to execute"}
                }
            },
            {
                "name": "file_read",
                "type": "custom",
                "description": "Read content from a file",
                "parameters": {
                    "file_path": {"type": "string", "description": "Path to the file"}
                }
            },
            {
                "name": "file_write",
                "type": "custom",
                "description": "Write content to a file",
                "parameters": {
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"}
                }
            },
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool_def: Dict[str, Any]) -> None:
        """
        Register an MCP tool.
        
        Args:
            tool_def: Tool definition dictionary
        """
        tool_name = tool_def["name"]
        self.tools[tool_name] = tool_def
    
    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool definition by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool definition or None if not found
        """
        return self.tools.get(tool_name)
    
    def list_tools(self, tool_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered tools.
        
        Args:
            tool_type: Optional filter by tool type
            
        Returns:
            List of tool definitions
        """
        tools = list(self.tools.values())
        if tool_type:
            tools = [t for t in tools if t.get("type") == tool_type]
        return tools
    
    def tool_to_requirement(self, tool_name: str) -> Optional[ToolRequirement]:
        """
        Convert a tool definition to a ToolRequirement.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolRequirement or None if not found
        """
        tool_def = self.get_tool(tool_name)
        if not tool_def:
            return None
        
        return ToolRequirement(
            name=tool_def["name"],
            type=tool_def.get("type", "custom"),
            description=tool_def.get("description", ""),
            required=True
        )
    
    def check_tool_availability(self, required_tools: List[ToolRequirement]) -> Dict[str, bool]:
        """
        Check which required tools are available.
        
        Args:
            required_tools: List of required tools
            
        Returns:
            Dictionary mapping tool names to availability
        """
        availability = {}
        for tool_req in required_tools:
            availability[tool_req.name] = tool_req.name in self.tools
        return availability




