"""ADK App structure for the Agent Recommendation System."""

import yaml
from google.adk.apps import App
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.apps.app import EventsCompactionConfig
from google.adk.agents import LlmAgent
from src.observability.logging import get_logger

# Setup observability (use existing root logger without adding handlers)
logger = get_logger(__name__)


def create_recommendation_app(root_agent: LlmAgent, config_path: str = "config/config.yaml") -> App:
    """
    Create ADK App for the recommendation system.
    
    This function creates an ADK App with:
    - Root agent (coordinator with sub_agents)
    - Context filtering plugin (keeps last N turns)
    - Events compaction (summarizes old conversations)
    
    Args:
        root_agent: The root ADK agent (coordinator with sub_agents)
        config_path: Path to configuration file
        
    Returns:
        ADK App instance with plugins configured
    """
    logger.info("Creating ADK App for Agent Recommendation System...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get session configuration
    session_config = config.get("session", {})
    
    # Create ADK App with plugins
    app = App(
        name="agent_recommendation_system",
        root_agent=root_agent,
        plugins=[
            # Context filtering: keep last N turns to manage context window
            ContextFilterPlugin(
                num_invocations_to_keep=session_config.get("max_history_length", 50)
            ),
        ],
        # Events compaction: summarize old conversations to reduce tokens
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=session_config.get("compaction_threshold", 30),
            overlap_size=1,
        ) if session_config.get("enable_compaction", True) else None,
    )
    
    logger.info("ADK App created successfully with plugins")
    return app
