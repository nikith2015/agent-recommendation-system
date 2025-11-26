"""ADK-compatible tools for the recommendation system."""

from typing import List, Optional
from google.adk.tools import FunctionTool as Tool
from pydantic import BaseModel, Field
from ..models.agent_spec import AgentSpecification
from .registry import AgentRegistry
from .vector_search import VectorSearchTool


class SearchAgentsInput(BaseModel):
    """Input schema for search_agents tool."""
    query: str = Field(..., description="Search query or goal description")
    top_k: int = Field(default=10, description="Number of results to return")


class SearchAgentsOutput(BaseModel):
    """Output schema for search_agents tool."""
    agents: List[dict] = Field(..., description="List of matching agents with similarity scores")


def create_search_agents_tool(
    agent_registry: AgentRegistry,
    vector_search: VectorSearchTool
) -> Tool:
    """
    Create ADK tool for searching agents.
    
    Args:
        agent_registry: Agent registry instance
        vector_search: Vector search tool instance
        
    Returns:
        ADK Tool instance
    """
    def search_agents(query: str, top_k: int = 10) -> dict:
        """
        Search for similar agents using hybrid retrieval.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Dictionary with matching agents
        """
        # Get all agents
        all_agents = agent_registry.list_agents()
        
        if not all_agents:
            return {"agents": []}
        
        # Perform semantic search
        results = vector_search.semantic_search(query, all_agents, top_k=top_k)
        
        # Format results
        agents_data = []
        for agent, score in results:
            agents_data.append({
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "capabilities": [cap.value for cap in agent.capabilities],
                "similarity_score": score
            })
        
        return {"agents": agents_data}
    
    return Tool(
        # name="search_agents",
        # description="Search for similar agents in the registry using semantic search. Returns agents ranked by similarity to the query.",
        # input_schema=SearchAgentsInput,
        # output_schema=SearchAgentsOutput,
        func=search_agents
    )


class GetAgentInput(BaseModel):
    """Input schema for get_agent tool."""
    agent_id: str = Field(..., description="ID of the agent to retrieve")


class GetAgentOutput(BaseModel):
    """Output schema for get_agent tool."""
    agent: Optional[dict] = Field(None, description="Agent specification if found")


def create_get_agent_tool(agent_registry: AgentRegistry) -> Tool:
    """
    Create ADK tool for getting a specific agent.
    
    Args:
        agent_registry: Agent registry instance
        
    Returns:
        ADK Tool instance
    """
    def get_agent(agent_id: str) -> dict:
        """
        Get an agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent data or None
        """
        agent = agent_registry.get_agent(agent_id)
        if not agent:
            return {"agent": None}
        
        return {
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "capabilities": [cap.value for cap in agent.capabilities],
                "tools": [tool.name for tool in agent.tools],
                "domain": agent.domain,
                "complexity_score": agent.complexity_score,
                "dependencies": agent.dependencies
            }
        }
    
    return Tool(
        # name="get_agent",
        # description="Retrieve a specific agent from the registry by its ID.",
        # input_schema=GetAgentInput,
        # output_schema=GetAgentOutput,
        func=get_agent
    )


