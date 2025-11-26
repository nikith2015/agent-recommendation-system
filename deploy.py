"""Deployment script for Agent Engine.

This script deploys the Agent Recommendation System to Google Cloud Vertex AI Agent Engine.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vertexai.preview.reasoning_engines import ReasoningEngine
from google.adk.apps.executor.app_executor import AppExecutor
from src.main import AgentRecommendationSystem
from src.observability.logging import setup_logging, get_logger

logger = setup_logging()


def create_agent_engine_app():
    """
    Create the app instance for Agent Engine deployment.
    
    Returns:
        AppExecutor instance ready for Agent Engine
    """
    logger.info("Initializing Agent Recommendation System for deployment...")
    
    # Initialize the system
    system = AgentRecommendationSystem()
    
    # Get the ADK app
    app = system.app
    
    # Create executor for Agent Engine
    executor = AppExecutor(app=app)
    
    logger.info("App executor created successfully")
    return executor


def deploy_to_agent_engine(
    project_id: str,
    location: str = "us-central1",
    agent_id: str = "agent-recommendation-system",
    display_name: str = "Agent Recommendation System"
):
    """
    Deploy the agent to Vertex AI Agent Engine.
    
    Args:
        project_id: Google Cloud project ID
        location: GCP region (default: us-central1)
        agent_id: Unique identifier for the agent
        display_name: Display name for the agent
    """
    from google.cloud import aiplatform
    
    logger.info(f"Deploying to Agent Engine: {agent_id}")
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Create executor
    executor = create_agent_engine_app()
    
    # Create reasoning engine
    reasoning_engine = ReasoningEngine.from_app_executor(
        executor=executor,
        agent_id=agent_id,
        display_name=display_name,
    )
    
    logger.info(f"Successfully deployed agent: {agent_id}")
    logger.info(f"Agent endpoint: {reasoning_engine.resource_name}")
    
    return reasoning_engine


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Agent Recommendation System to Agent Engine")
    parser.add_argument("--project-id", type=str, required=True, help="Google Cloud project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="GCP region")
    parser.add_argument("--agent-id", type=str, default="agent-recommendation-system", help="Agent ID")
    parser.add_argument("--display-name", type=str, default="Agent Recommendation System", help="Display name")
    
    args = parser.parse_args()
    
    try:
        reasoning_engine = deploy_to_agent_engine(
            project_id=args.project_id,
            location=args.location,
            agent_id=args.agent_id,
            display_name=args.display_name
        )
        print(f"\n✅ Successfully deployed to Agent Engine!")
        print(f"Agent ID: {args.agent_id}")
        print(f"Resource Name: {reasoning_engine.resource_name}")
    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=True)
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)




