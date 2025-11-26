"""Basic tests for the agent recommendation system."""

import pytest
from src.models.agent_spec import (
    AgentSpecification,
    AgentCapability,
    ToolRequirement,
    GoalSpecification,
    ReuseDifficulty
)


def test_agent_specification_model():
    """Test AgentSpecification model."""
    agent = AgentSpecification(
        id="test_001",
        name="Test Agent",
        description="A test agent",
        capabilities=[AgentCapability.TEXT_PROCESSING],
        tools=[],
        complexity_score=0.5
    )
    
    assert agent.id == "test_001"
    assert agent.name == "Test Agent"
    assert AgentCapability.TEXT_PROCESSING in agent.capabilities


def test_goal_specification_model():
    """Test GoalSpecification model."""
    goal_spec = GoalSpecification(
        goal="Test goal",
        capabilities=[AgentCapability.SUMMARIZATION],
        tool_requirements=[],
        keywords=["test", "goal"]
    )
    
    assert goal_spec.goal == "Test goal"
    assert AgentCapability.SUMMARIZATION in goal_spec.capabilities
    assert "test" in goal_spec.keywords


def test_reuse_difficulty_enum():
    """Test ReuseDifficulty enum."""
    assert ReuseDifficulty.EASY.value == "easy"
    assert ReuseDifficulty.MODERATE.value == "moderate"
    assert ReuseDifficulty.HARD.value == "hard"
    assert ReuseDifficulty.VERY_HARD.value == "very_hard"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




