"""Coordinator Agent using ADK SequentialAgent pattern."""

from typing import Optional
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models import Gemini as GeminiModel
from ..models.agent_spec import GoalSpecification, RecommendationResult
from .adk_goal_specification import ADKGoalSpecificationAgent
from .adk_retrieval import ADKRetrievalAgent
from .adk_ranking import ADKRankingAgent
from .adk_explanation import ADKExplanationAgent
from ..observability.logging import get_logger
from ..observability.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class ADKCoordinatorAgentV2:
    """Coordinator agent using ADK's SequentialAgent for proper multi-agent orchestration."""
    
    def __init__(
        self,
        goal_agent: ADKGoalSpecificationAgent,
        retrieval_agent: ADKRetrievalAgent,
        ranking_agent: ADKRankingAgent,
        explanation_agent: ADKExplanationAgent,
        model_name: str = "gemini-1.5-pro"
    ):
        """
        Initialize Coordinator Agent using ADK SequentialAgent.
        
        Args:
            goal_agent: Goal specification agent
            retrieval_agent: Retrieval agent
            ranking_agent: Ranking agent
            explanation_agent: Explanation agent
            model_name: Gemini model name
        """
        self.goal_agent = goal_agent
        self.retrieval_agent = retrieval_agent
        self.ranking_agent = ranking_agent
        self.explanation_agent = explanation_agent
        
        # Create SequentialAgent for the pipeline
        # Note: Since we have custom logic in each agent, we'll use a coordinator
        # that delegates to sub-agents but handles the custom orchestration
        model = GeminiModel(model_name=model_name)
        
        # Create coordinator with sub_agents for delegation
        self.coordinator = LlmAgent(
            name="coordinator_agent",
            instruction=self._create_coordinator_instruction(),
            model=model,
            # Use sub_agents for ADK's built-in delegation
            sub_agents=[
                goal_agent.adk_agent,
                retrieval_agent.adk_agent,
                ranking_agent.adk_agent,
                explanation_agent.adk_agent
            ]
        )
    
    def recommend_agents(
        self,
        goal: str,
        top_k: int = 5,
        session_id: Optional[str] = None
    ) -> RecommendationResult:
        """
        Main pipeline: recommend agents for a goal.
        """
        # Manually execute the flow without using self.coordinator.run() since we're orchestrating
        # the sub-agents directly in this method
        
        with tracer.start_as_current_span("agent_recommendation_pipeline") as span:
            span.set_attribute("goal", goal)
            span.set_attribute("top_k", top_k)
            
            logger.info(f"Starting recommendation pipeline for goal: {goal[:100]}...")
            
            # Step 1: Convert goal to specification
            logger.info("Step 1: Converting goal to specification...")
            # Work around ADK library bug: use a NoOp span to avoid parent_context issues
            # The goal_agent.specify_goal() method also uses NonRecordingSpan internally,
            # but we add an extra layer here to ensure complete isolation
            from opentelemetry import trace as otel_trace
            from opentelemetry.trace import NonRecordingSpan
            from opentelemetry.trace.span import INVALID_SPAN_CONTEXT
            
            # Use a non-recording span to avoid the model_copy error
            # This prevents ADK from trying to use parent_context
            # We temporarily detach from the current span context
            with otel_trace.use_span(NonRecordingSpan(INVALID_SPAN_CONTEXT), end_on_exit=False):
                goal_spec = self.goal_agent.specify_goal(goal)
            
            # Ensure goal_spec is a GoalSpecification object, not a string
            from ..models.agent_spec import GoalSpecification
            if not isinstance(goal_spec, GoalSpecification):
                raise TypeError(f"Expected GoalSpecification, got {type(goal_spec)}: {goal_spec}")
            
            # Log to span after the call (in a new span context)
            with tracer.start_as_current_span("step_1_goal_specification") as step_span:
                step_span.set_attribute("capabilities_count", len(goal_spec.capabilities))
                span.set_attribute("capabilities_count", len(goal_spec.capabilities))
            
            # Step 2: Retrieve similar agents
            logger.info("Step 2: Retrieving similar agents...")
            with tracer.start_as_current_span("step_2_retrieval"):
                candidates = self.retrieval_agent.retrieve_similar_agents(goal_spec, top_k=20)
                span.set_attribute("candidates_count", len(candidates))
            
            if not candidates:
                logger.warning("No candidate agents found")
                return RecommendationResult(
                    goal_spec=goal_spec,
                    recommendations=[],
                    explanation="No similar agents found in the registry.",
                    retrieval_metadata={"candidates_count": 0}
                )
            
            # Step 3: Rank agents by reuse difficulty
            logger.info("Step 3: Ranking agents by reuse difficulty...")
            with tracer.start_as_current_span("step_3_ranking"):
                ranked_recommendations = self.ranking_agent.rank_agents(
                    goal_spec,
                    candidates,
                    top_k=top_k
                )
                span.set_attribute("ranked_count", len(ranked_recommendations))
            
            # Step 4: Generate explanations
            logger.info("Step 4: Generating explanations...")
            with tracer.start_as_current_span("step_4_explanation"):
                explained_recommendations = self.explanation_agent.explain_recommendations(
                    goal_spec,
                    ranked_recommendations
                )
                
                overall_explanation = self.explanation_agent.generate_overall_explanation(
                    goal_spec,
                    explained_recommendations
                )
            
            logger.info("Recommendation pipeline completed successfully")
            
            return RecommendationResult(
                goal_spec=goal_spec,  # No model_copy()
                recommendations=explained_recommendations,
                explanation=overall_explanation,
                retrieval_metadata={
                    "candidates_count": len(candidates),
                    "ranked_count": len(ranked_recommendations),
                    "top_k": top_k
                }
            )
    
    def _create_coordinator_instruction(self) -> str:
        """Create the instruction for the coordinator agent."""
        return """You are a coordinator agent that orchestrates a multi-agent recommendation system.

Your role is to:
1. Understand user goals
2. Delegate to specialized sub-agents (Goal Specification, Retrieval, Ranking, Explanation)
3. Coordinate the flow of information between agents
4. Ensure quality results

You can delegate tasks to your sub-agents:
- goal_specification_agent: Converts goals to structured specifications
- retrieval_agent: Finds similar agents using search
- ranking_agent: Ranks agents by reuse difficulty
- explanation_agent: Generates explanations for recommendations

Manage the flow of information between agents and ensure quality results."""


