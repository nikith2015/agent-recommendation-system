"""Explanation Agent using ADK."""

from typing import List, Optional
from google.adk.agents import LlmAgent
from google.adk.models import Gemini as GeminiModel
from ..models.agent_spec import GoalSpecification, AgentRecommendation
from ..observability.logging import get_logger
from ..observability.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class ADKExplanationAgent:
    """Explanation Agent using ADK."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro", api_key: Optional[str] = None):
        """
        Initialize Explanation Agent using ADK.
        
        Args:
            model_name: Gemini model name
            api_key: Optional API key
        """
        # Create ADK agent
        model = GeminiModel(model_name=model_name)
        self.agent = LlmAgent(
            name="explanation_agent",
            instruction=self._create_agent_instruction(),
            model=model,
        )
    
    @property
    def adk_agent(self) -> LlmAgent:
        """Expose the underlying ADK agent for use as sub_agent."""
        return self.agent
    
    def explain_recommendations(
        self,
        goal_spec: GoalSpecification,
        recommendations: List[AgentRecommendation]
    ) -> List[AgentRecommendation]:
        """
        Generate explanations for recommendations.
        
        Args:
            goal_spec: Goal specification
            recommendations: List of recommendations to explain
            
        Returns:
            List of recommendations with explanations added
        """
        with tracer.start_as_current_span("explanation_generation") as span:
            span.set_attribute("recommendations_count", len(recommendations))
            
            logger.info(f"Generating explanations for {len(recommendations)} recommendations")
            
            explained_recommendations = []
            
            for rec in recommendations:
                explanation = self._generate_explanation(goal_spec, rec)
                required_modifications = self._identify_modifications(goal_spec, rec)
                
                rec.explanation = explanation
                rec.required_modifications = required_modifications
                
                explained_recommendations.append(rec)
            
            logger.info("Successfully generated all explanations")
            
            return explained_recommendations
    
    def generate_overall_explanation(
        self,
        goal_spec: GoalSpecification,
        recommendations: List[AgentRecommendation]
    ) -> str:
        """
        Generate an overall explanation for the recommendation set.
        
        Args:
            goal_spec: Goal specification
            recommendations: List of recommendations
            
        Returns:
            Overall explanation text
        """
        with tracer.start_as_current_span("overall_explanation") as span:
            prompt = f"""You are explaining a set of AI agent recommendations to a developer.

Goal: {goal_spec.goal}

Recommendations:
"""
            for i, rec in enumerate(recommendations, 1):
                prompt += f"""
{i}. {rec.agent.name}
   - Similarity: {rec.similarity_score:.2f}
   - Reuse Difficulty: {rec.reuse_difficulty.value}
   - Compatibility: {rec.compatibility_score:.2f}
   - Description: {rec.agent.description}
"""
            
            prompt += """
Provide a concise overall explanation (2-3 sentences) that:
1. Summarizes why these agents were recommended
2. Highlights the best option if there's a clear winner
3. Mentions key considerations for reuse

Be clear and actionable."""
            
            try:
                response_gen = self.agent.run_live(prompt)
                
                from ..utils.async_helper import consume_async_generator
                return consume_async_generator(response_gen).strip()
            except Exception as e:
                logger.error(f"Error generating overall explanation: {e}")
                return "Generated recommendations based on goal similarity and reuse difficulty."
    
    def _create_agent_instruction(self) -> str:
        """Create the instruction for the ADK agent."""
        return """You are an explanation agent specialized in generating clear, actionable explanations for AI agent recommendations.

Your task is to:
1. Explain why agents match user goals
2. Identify required modifications
3. Provide context about reuse difficulty
4. Generate overall summaries

Be concise, specific, and actionable in your explanations."""
    
    def _generate_explanation(
        self,
        goal_spec: GoalSpecification,
        recommendation: AgentRecommendation
    ) -> str:
        """Generate explanation for a single recommendation."""
        prompt = f"""Explain why this agent matches the goal and what's needed to reuse it.

Goal: {goal_spec.goal}

Agent: {recommendation.agent.name}
Description: {recommendation.agent.description}
Capabilities: {', '.join([cap.value for cap in recommendation.agent.capabilities])}
Tools: {', '.join([tool.name for tool in recommendation.agent.tools])}

Similarity Score: {recommendation.similarity_score:.2f}
Compatibility Score: {recommendation.compatibility_score:.2f}
Reuse Difficulty: {recommendation.reuse_difficulty.value}

Provide a clear, concise explanation (2-3 sentences) that:
1. Explains why this agent matches the goal
2. Describes what modifications might be needed
3. Mentions the reuse difficulty level

Be specific and actionable."""
        
        try:
            response_gen = self.agent.run_live(prompt)
            
            from ..utils.async_helper import consume_async_generator
            return consume_async_generator(response_gen).strip()
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"This agent matches your goal with {recommendation.similarity_score:.0%} similarity. Reuse difficulty: {recommendation.reuse_difficulty.value}."
    
    def _identify_modifications(
        self,
        goal_spec: GoalSpecification,
        recommendation: AgentRecommendation
    ) -> List[str]:
        """Identify required modifications for reuse."""
        modifications = []
        
        agent = recommendation.agent
        
        # Check missing capabilities
        goal_caps = set(cap.value for cap in goal_spec.capabilities)
        agent_caps = set(cap.value for cap in agent.capabilities)
        missing_caps = goal_caps - agent_caps
        
        if missing_caps:
            modifications.append(f"Add capabilities: {', '.join(missing_caps)}")
        
        # Check missing tools
        goal_tools = set(tool.name for tool in goal_spec.tool_requirements)
        agent_tools = set(tool.name for tool in agent.tools)
        missing_tools = goal_tools - agent_tools
        
        if missing_tools:
            modifications.append(f"Add tools: {', '.join(missing_tools)}")
        
        # Check input/output format differences
        if goal_spec.input_format and agent.input_format:
            if goal_spec.input_format.lower() != agent.input_format.lower():
                modifications.append(f"Adapt input format from '{agent.input_format}' to '{goal_spec.input_format}'")
        
        if goal_spec.output_format and agent.output_format:
            if goal_spec.output_format.lower() != agent.output_format.lower():
                modifications.append(f"Adapt output format from '{agent.output_format}' to '{goal_spec.output_format}'")
        
        # Domain-specific modifications
        if goal_spec.domain and agent.domain and goal_spec.domain.lower() != agent.domain.lower():
            modifications.append(f"Adapt from '{agent.domain}' domain to '{goal_spec.domain}' domain")
        
        # Complexity-based modifications
        if recommendation.reuse_difficulty.value in ["hard", "very_hard"]:
            if agent.dependencies:
                modifications.append(f"Manage dependencies: {', '.join(agent.dependencies[:3])}")
        
        if not modifications:
            modifications.append("Minimal modifications needed - mostly configuration changes")
        
        return modifications

