"""Data models for agent specifications."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentCapability(str, Enum):
    """Types of capabilities an agent can have."""
    TEXT_PROCESSING = "text_processing"
    DATA_ANALYSIS = "data_analysis"
    API_INTEGRATION = "api_integration"
    CODE_EXECUTION = "code_execution"
    SEARCH = "search"
    DATABASE_ACCESS = "database_access"
    FILE_OPERATIONS = "file_operations"
    EMAIL = "email"
    CALENDAR = "calendar"
    WEB_SCRAPING = "web_scraping"
    IMAGE_PROCESSING = "image_processing"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


class ToolRequirement(BaseModel):
    """Represents a tool requirement for an agent."""
    name: str = Field(..., description="Name of the required tool")
    type: str = Field(..., description="Type of tool (MCP, custom, built-in, OpenAPI)")
    description: str = Field(..., description="Description of what the tool does")
    required: bool = Field(default=True, description="Whether the tool is required")


class GoalSpecification(BaseModel):
    """Structured specification derived from a high-level goal."""
    goal: str = Field(..., description="Original high-level goal")
    capabilities: List[AgentCapability] = Field(..., description="Required capabilities")
    tool_requirements: List[ToolRequirement] = Field(default_factory=list, description="Required tools")
    input_format: Optional[str] = Field(None, description="Expected input format")
    output_format: Optional[str] = Field(None, description="Expected output format")
    domain: Optional[str] = Field(None, description="Domain or industry context")
    constraints: List[str] = Field(default_factory=list, description="Additional constraints")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords for search")


class ReuseDifficulty(str, Enum):
    """Difficulty levels for reusing an agent."""
    EASY = "easy"  # Minimal changes needed
    MODERATE = "moderate"  # Some modifications required
    HARD = "hard"  # Significant changes needed
    VERY_HARD = "very_hard"  # Major refactoring required


class AgentSpecification(BaseModel):
    """Complete specification of an existing agent."""
    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of what the agent does")
    capabilities: List[AgentCapability] = Field(..., description="Agent capabilities")
    tools: List[ToolRequirement] = Field(default_factory=list, description="Tools used by agent")
    input_format: Optional[str] = Field(None, description="Input format")
    output_format: Optional[str] = Field(None, description="Output format")
    domain: Optional[str] = Field(None, description="Domain or industry")
    dependencies: List[str] = Field(default_factory=list, description="External dependencies")
    complexity_score: float = Field(default=0.0, description="Complexity score (0-1)")
    code_lines: Optional[int] = Field(None, description="Approximate lines of code")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for semantic search")


class AgentRecommendation(BaseModel):
    """A recommendation result for an agent."""
    agent: AgentSpecification = Field(..., description="The recommended agent")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    reuse_difficulty: ReuseDifficulty = Field(..., description="Difficulty of reusing this agent")
    reuse_difficulty_score: float = Field(..., description="Numeric difficulty score (0-1)")
    explanation: str = Field(..., description="Explanation of the match and reuse requirements")
    required_modifications: List[str] = Field(default_factory=list, description="Required modifications")
    compatibility_score: float = Field(..., description="Compatibility score (0-1)")


class RecommendationResult(BaseModel):
    """Complete recommendation result."""
    goal_spec: GoalSpecification = Field(..., description="Structured goal specification")
    recommendations: List[AgentRecommendation] = Field(..., description="Ranked recommendations")
    explanation: str = Field(..., description="Overall explanation of the recommendations")
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict, description="Retrieval process metadata")




