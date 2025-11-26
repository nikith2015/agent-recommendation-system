"""Retrieval Agent using ADK."""

from typing import List, Tuple
from google.adk.agents import LlmAgent
from google.adk.models import Gemini as GeminiModel
from ..models.agent_spec import AgentSpecification, GoalSpecification
from ..tools.registry import AgentRegistry
from ..tools.vector_search import VectorSearchTool
from ..tools.adk_tools import create_search_agents_tool
from ..observability.logging import get_logger

logger = get_logger(__name__)


class ADKRetrievalAgent:
    """Retrieval Agent using ADK with tools."""
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        vector_search: VectorSearchTool,
        model_name: str = "gemini-1.5-pro"
    ):
        """
        Initialize Retrieval Agent using ADK.
        
        Args:
            agent_registry: Agent registry tool
            vector_search: Vector search tool
            model_name: Gemini model name
        """
        self.registry = agent_registry
        self.vector_search = vector_search
        
        # Create tools
        search_tool = create_search_agents_tool(agent_registry, vector_search)
        
        # Create ADK agent with tools
        model = GeminiModel(model_name=model_name)
        self.agent = LlmAgent(
            name="retrieval_agent",
            instruction=self._create_agent_instruction(),
            model=model,
            tools=[search_tool]
        )
    
    @property
    def adk_agent(self) -> LlmAgent:
        """Expose the underlying ADK agent for use as sub_agent."""
        return self.agent
    
    def retrieve_similar_agents(
        self,
        goal_spec: GoalSpecification,
        top_k: int = 20
    ) -> List[Tuple[AgentSpecification, float]]:
        """
        Retrieve similar agents using hybrid retrieval.
        
        Args:
            goal_spec: Goal specification
            top_k: Number of candidates to retrieve
            
        Returns:
            List of (agent, combined_score) tuples, sorted by score
        """
        logger.info(f"Retrieving similar agents for goal: {goal_spec.goal[:100]}...")
        
        # Build query from goal specification
        query_parts = [
            goal_spec.goal,
            ", ".join([cap.value for cap in goal_spec.capabilities]),
            goal_spec.domain or "",
            ", ".join(goal_spec.keywords)
        ]
        query = " ".join(query_parts)
        
        # Direct vector search (ADK has model_copy issues, so we use direct search)
        all_agents = self.registry.list_agents()
        if not all_agents:
            logger.warning("No agents found in registry")
            return []
        
        # Use direct vector search - more reliable than ADK agent
        logger.info(f"Using direct vector search on {len(all_agents)} agents")
        try:
            semantic_results = self.vector_search.semantic_search(query, all_agents, top_k=top_k)
            logger.info(f"Retrieved {len(semantic_results)} candidate agents via vector search")
            
            # If we got results, return them
            if semantic_results:
                return semantic_results
            
            # Fallback to keyword search if vector search returned no results
            logger.info("Vector search returned no results, trying keyword fallback...")
            return self._keyword_fallback_search(query, all_agents, top_k)
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            # Fallback to keyword search
            return self._keyword_fallback_search(query, all_agents, top_k)
    
    def _keyword_fallback_search(
        self,
        query: str,
        agents: List[AgentSpecification],
        top_k: int = 20
    ) -> List[Tuple[AgentSpecification, float]]:
        """Fallback keyword-based search when vector search fails."""
        logger.info("Using keyword-based fallback search")
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for agent in agents:
            # Build searchable text
            agent_text = " ".join([
                agent.name or "",
                agent.description or "",
                ", ".join([cap.value for cap in agent.capabilities]) if agent.capabilities else "",
                agent.domain or "",
            ]).lower()
            
            agent_words = set(agent_text.split())
            
            # Calculate simple keyword match score
            if query_words:
                matches = len(query_words.intersection(agent_words))
                score = matches / len(query_words)
            else:
                score = 0.0
            
            if score > 0:
                results.append((agent, float(score)))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Keyword search found {len(results)} agents")
        return results[:top_k]
    
    def _create_agent_instruction(self) -> str:
        """Create the instruction for the ADK agent."""
        return """You are a retrieval agent specialized in finding similar AI agents.

Your task is to:
1. Understand goal specifications
2. Use the search_agents tool to find similar agents
3. Return the most relevant matches

Focus on semantic similarity and capability matching."""

