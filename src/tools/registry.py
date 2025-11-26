"""Agent registry tool for accessing agent specifications."""

from typing import List, Optional
from ..memory.agent_registry import AgentRegistryMemory
from ..models.agent_spec import AgentSpecification


class AgentRegistry:
    """Tool for interacting with the agent registry."""
    
    def __init__(self, registry_memory: AgentRegistryMemory):
        """
        Initialize agent registry tool.
        
        Args:
            registry_memory: Agent registry memory instance
        """
        self.registry = registry_memory
    
    def get_agent(self, agent_id: str) -> Optional[AgentSpecification]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent specification or None if not found
        """
        return self.registry.get_agent(agent_id)
    
    def list_agents(self, limit: Optional[int] = None) -> List[AgentSpecification]:
        """
        List all agents in the registry.
        
        Args:
            limit: Optional limit on number of agents to return
            
        Returns:
            List of agent specifications
        """
        agents = self.registry.get_all_agents()
        if limit:
            return agents[:limit]
        return agents
    
    def search_agents(
        self, 
        keyword: str,
        domain: Optional[str] = None,
        capabilities: Optional[List[str]] = None
    ) -> List[AgentSpecification]:
        """
        Search agents by keyword, domain, or capabilities.
        
        Args:
            keyword: Search keyword
            domain: Optional domain filter
            capabilities: Optional list of required capabilities
            
        Returns:
            List of matching agents
        """
        results = self.registry.search_by_keyword(keyword)
        
        # Filter by domain if specified
        if domain:
            results = [a for a in results if a.domain and domain.lower() in a.domain.lower()]
        
        # Filter by capabilities if specified
        if capabilities:
            capability_set = set(c.lower() for c in capabilities)
            results = [
                a for a in results 
                if any(cap.value.lower() in capability_set for cap in a.capabilities)
            ]
        
        return results




