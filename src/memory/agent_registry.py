"""Agent registry for storing and retrieving agent specifications."""

from typing import List, Optional, Dict, Any
import json
from pathlib import Path
from ..models.agent_spec import AgentSpecification


class AgentRegistryMemory:
    """In-memory agent registry with persistence support."""
    
    def __init__(self, registry_file: Optional[str] = None):
        """
        Initialize agent registry.
        
        Args:
            registry_file: Optional path to JSON file for persistence
        """
        self.agents: Dict[str, AgentSpecification] = {}
        self.registry_file = registry_file
        
        if registry_file and Path(registry_file).exists():
            self.load_from_file(registry_file)
    
    def add_agent(self, agent: AgentSpecification) -> None:
        """Add an agent to the registry."""
        self.agents[agent.id] = agent
    
    def get_agent(self, agent_id: str) -> Optional[AgentSpecification]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[AgentSpecification]:
        """Get all agents in the registry."""
        return list(self.agents.values())
    
    def search_by_keyword(self, keyword: str) -> List[AgentSpecification]:
        """
        Simple keyword search in agent descriptions and names.
        
        Args:
            keyword: Search keyword
            
        Returns:
            List of matching agents
        """
        keyword_lower = keyword.lower()
        results = []
        
        for agent in self.agents.values():
            if (keyword_lower in agent.name.lower() or 
                keyword_lower in agent.description.lower() or
                any(keyword_lower in cap.value.lower() for cap in agent.capabilities)):
                results.append(agent)
        
        return results
    
    def load_from_file(self, file_path: str) -> None:
        """Load agents from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for agent_data in data.get('agents', []):
                agent = AgentSpecification(**agent_data)
                self.agents[agent.id] = agent
    
    def save_to_file(self, file_path: Optional[str] = None) -> None:
        """Save agents to a JSON file."""
        save_path = file_path or self.registry_file
        if not save_path:
            return
        
        agents_data = {
            'agents': [agent.model_dump() for agent in self.agents.values()]
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(agents_data, f, indent=2, default=str)




