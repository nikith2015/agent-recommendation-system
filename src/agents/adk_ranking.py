"""Ranking Agent using ADK."""

from typing import List, Tuple, Optional
from google.adk.agents import LlmAgent
from google.adk.models import Gemini as GeminiModel
from ..models.agent_spec import (
    AgentSpecification,
    GoalSpecification,
    AgentRecommendation,
    ReuseDifficulty
)
from ..tools.mcp_tools import MCPToolRegistry
from ..observability.logging import get_logger
from ..observability.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class ADKRankingAgent:
    """Ranking Agent using ADK."""
    
    def __init__(
        self,
        mcp_tool_registry: MCPToolRegistry,
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        similarity_weight: float = 0.4,
        compatibility_weight: float = 0.3,
        complexity_weight: float = 0.2,
        dependency_weight: float = 0.1
    ):
        """
        Initialize Ranking Agent using ADK.
        
        Args:
            mcp_tool_registry: MCP tool registry
            model_name: Gemini model name
            api_key: Optional API key
            similarity_weight: Weight for similarity score
            compatibility_weight: Weight for compatibility score
            complexity_weight: Weight for complexity score
            dependency_weight: Weight for dependency score
        """
        self.mcp_tools = mcp_tool_registry
        
        # Normalize weights
        total = similarity_weight + compatibility_weight + complexity_weight + dependency_weight
        if total > 0:
            self.similarity_weight = similarity_weight / total
            self.compatibility_weight = compatibility_weight / total
            self.complexity_weight = complexity_weight / total
            self.dependency_weight = dependency_weight / total
        else:
            self.similarity_weight = 0.4
            self.compatibility_weight = 0.3
            self.complexity_weight = 0.2
            self.dependency_weight = 0.1
        
        # Create ADK agent
        model = GeminiModel(model_name=model_name)
        self.agent = LlmAgent(
            name="ranking_agent",
            instruction=self._create_agent_instruction(),
            model=model,
        )
    
    @property
    def adk_agent(self) -> LlmAgent:
        """Expose the underlying ADK agent for use as sub_agent."""
        return self.agent
    
    def rank_agents(
        self,
        goal_spec: GoalSpecification,
        candidates: List[Tuple[AgentSpecification, float]],
        top_k: int = 5
    ) -> List[AgentRecommendation]:
        """
        Rank agents by reuse difficulty and compatibility.
        
        Args:
            goal_spec: Goal specification
            candidates: List of (agent, similarity_score) tuples
            top_k: Number of top recommendations to return
            
        Returns:
            List of ranked agent recommendations
        """
        with tracer.start_as_current_span("agent_ranking") as span:
            span.set_attribute("candidates_count", len(candidates))
            span.set_attribute("top_k", top_k)
            
            logger.info(f"Ranking {len(candidates)} candidate agents")
            
            recommendations = []
            
            for agent, similarity_score in candidates[:top_k * 2]:  # Evaluate more than needed
                # Calculate compatibility score
                compatibility_score = self._calculate_compatibility(goal_spec, agent)
                
                # Calculate complexity score
                complexity_score = self._calculate_complexity(agent)
                
                # Calculate dependency score
                dependency_score = self._calculate_dependency_score(agent)
                
                # Calculate overall reuse difficulty
                reuse_difficulty_score = self._calculate_reuse_difficulty(
                    similarity_score,
                    compatibility_score,
                    complexity_score,
                    dependency_score
                )
                
                reuse_difficulty = self._score_to_difficulty(reuse_difficulty_score)
                
                recommendation = AgentRecommendation(
                    agent=agent,
                    similarity_score=similarity_score,
                    reuse_difficulty=reuse_difficulty,
                    reuse_difficulty_score=reuse_difficulty_score,
                    explanation="",  # Will be filled by ExplanationAgent
                    compatibility_score=compatibility_score
                )
                
                recommendations.append(recommendation)
            
            # Sort by reuse difficulty score (lower is easier)
            recommendations.sort(key=lambda x: x.reuse_difficulty_score)
            
            logger.info(f"Ranked {len(recommendations)} agents")
            span.set_attribute("ranked_count", len(recommendations))
            
            return recommendations[:top_k]
    
    def _create_agent_instruction(self) -> str:
        """Create the instruction for the ADK agent."""
        return """You are a ranking agent specialized in evaluating AI agents for reuse.

Your task is to:
1. Assess compatibility between goals and agents
2. Evaluate complexity and dependencies
3. Calculate reuse difficulty scores
4. Rank agents by ease of reuse

Be thorough and accurate in your assessments."""
    
    def _calculate_compatibility(
        self,
        goal_spec: GoalSpecification,
        agent: AgentSpecification
    ) -> float:
        """Calculate compatibility score between goal and agent."""
        score = 0.0
        factors = 0
        
        # Capability matching
        goal_caps = set(cap.value for cap in goal_spec.capabilities)
        agent_caps = set(cap.value for cap in agent.capabilities)
        
        if goal_caps:
            overlap = len(goal_caps & agent_caps)
            capability_match = overlap / len(goal_caps)
            score += capability_match * 0.4
            factors += 0.4
        
        # Tool requirement matching
        if goal_spec.tool_requirements:
            agent_tool_names = set(tool.name for tool in agent.tools)
            goal_tool_names = set(tool.name for tool in goal_spec.tool_requirements)
            
            if goal_tool_names:
                tool_overlap = len(goal_tool_names & agent_tool_names)
                tool_match = tool_overlap / len(goal_tool_names)
                score += tool_match * 0.3
                factors += 0.3
        
        # Domain matching
        if goal_spec.domain and agent.domain:
            if goal_spec.domain.lower() == agent.domain.lower():
                score += 0.2
                factors += 0.2
            elif goal_spec.domain.lower() in agent.domain.lower() or agent.domain.lower() in goal_spec.domain.lower():
                score += 0.1
                factors += 0.1
        
        # Input/output format matching (simplified)
        if goal_spec.input_format and agent.input_format:
            goal_input_words = set(goal_spec.input_format.lower().split())
            agent_input_words = set(agent.input_format.lower().split())
            if goal_input_words and agent_input_words:
                format_overlap = len(goal_input_words & agent_input_words)
                format_match = format_overlap / max(len(goal_input_words), len(agent_input_words))
                score += format_match * 0.1
                factors += 0.1
        
        # Normalize
        if factors > 0:
            return min(score / factors, 1.0)
        return 0.5  # Default moderate compatibility
    
    def _calculate_complexity(self, agent: AgentSpecification) -> float:
        """Calculate complexity score (higher = more complex)."""
        complexity = agent.complexity_score
        
        # Adjust based on code lines if available
        if agent.code_lines:
            if agent.code_lines > 5000:
                complexity = max(complexity, 0.8)
            elif agent.code_lines > 2000:
                complexity = max(complexity, 0.6)
            elif agent.code_lines > 500:
                complexity = max(complexity, 0.4)
        
        # Adjust based on number of tools
        if len(agent.tools) > 10:
            complexity = max(complexity, 0.7)
        elif len(agent.tools) > 5:
            complexity = max(complexity, 0.5)
        
        return complexity
    
    def _calculate_dependency_score(self, agent: AgentSpecification) -> float:
        """Calculate dependency complexity score (higher = more dependencies)."""
        if not agent.dependencies:
            return 0.1  # Low dependency score
        
        # More dependencies = higher score
        dep_count = len(agent.dependencies)
        if dep_count > 10:
            return 0.9
        elif dep_count > 5:
            return 0.7
        elif dep_count > 2:
            return 0.5
        else:
            return 0.3
    
    def _calculate_reuse_difficulty(
        self,
        similarity: float,
        compatibility: float,
        complexity: float,
        dependency: float
    ) -> float:
        """
        Calculate overall reuse difficulty score.
        Lower score = easier to reuse.
        """
        # Invert similarity and compatibility (higher = easier)
        # Higher complexity and dependency = harder
        difficulty = (
            (1.0 - similarity) * self.similarity_weight +
            (1.0 - compatibility) * self.compatibility_weight +
            complexity * self.complexity_weight +
            dependency * self.dependency_weight
        )
        
        return min(max(difficulty, 0.0), 1.0)
    
    def _score_to_difficulty(self, score: float) -> ReuseDifficulty:
        """Convert numeric score to difficulty level."""
        if score < 0.25:
            return ReuseDifficulty.EASY
        elif score < 0.5:
            return ReuseDifficulty.MODERATE
        elif score < 0.75:
            return ReuseDifficulty.HARD
        else:
            return ReuseDifficulty.VERY_HARD

