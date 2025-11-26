"""Entry point for Agent Engine deployment.

This file is used by Vertex AI Agent Engine to load and execute the agent.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from google.adk.apps.executor.app_executor import AppExecutor
from src.main import AgentRecommendationSystem
from src.observability.logging import setup_logging

# Setup logging
logger = setup_logging()


def create_app_executor():
    """
    Create the AppExecutor instance for Agent Engine.
    
    This function is called by Agent Engine to initialize the agent.
    
    Returns:
        AppExecutor instance
    """
    logger.info("Initializing Agent Recommendation System for Agent Engine...")
    
    try:
        # Ensure API key is set
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found in environment, trying .env file")
            load_dotenv()
            api_key = os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                logger.error("GOOGLE_API_KEY not found - agent may not work correctly")
        else:
            logger.info("GOOGLE_API_KEY found in environment")
        
        # Initialize the system
        logger.info("Creating AgentRecommendationSystem instance...")
        system = AgentRecommendationSystem()
        
        # Get the ADK app
        logger.info("Getting ADK app from system...")
        app = system.app
        
        # Create executor
        logger.info("Creating AppExecutor...")
        executor = AppExecutor(app=app)
        
        logger.info("App executor created successfully")
        logger.info("Agent is ready to handle requests")
        return executor
        
    except Exception as e:
        logger.error(f"Failed to create app executor: {e}", exc_info=True)
        logger.error("This may be due to:")
        logger.error("1. Missing dependencies - check requirements.txt")
        logger.error("2. Missing API key - set GOOGLE_API_KEY environment variable")
        logger.error("3. Missing configuration - check config/config.yaml")
        logger.error("4. Missing data files - run python src/main.py --init-registry")
        raise


# Agent Engine will call this function
app_executor = create_app_executor()


