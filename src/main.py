"""Main entry point for Agent Recommendation System using ADK."""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Add src to path
# sys.path.insert(0, str(Path(__file__).parent))

from src.models.agent_spec import AgentSpecification, AgentCapability, ToolRequirement
from src.memory.agent_registry import AgentRegistryMemory
from src.tools.registry import AgentRegistry
from src.tools.vector_search import VectorSearchTool
from src.tools.mcp_tools import MCPToolRegistry
from src.agents.adk_goal_specification import ADKGoalSpecificationAgent
from src.agents.adk_retrieval import ADKRetrievalAgent
from src.agents.adk_ranking import ADKRankingAgent
from src.agents.adk_explanation import ADKExplanationAgent
from src.agents.adk_coordinator_v2 import ADKCoordinatorAgentV2
from src.app import create_recommendation_app
from src.observability.logging import setup_logging, get_logger
from src.observability.tracing import setup_tracing
import warnings

# Suppress SSL warnings from urllib3/requests during initialization
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='.*SSL.*')
warnings.filterwarnings('ignore', message='.*certificate verify failed.*')
warnings.filterwarnings('ignore', message='.*MaxRetryError.*')

# Setup observability
logger = setup_logging()
tracer = setup_tracing()


def _mask_key(key: str, keep: int = 4) -> str:
    """Return a masked representation of a secret for safe logging."""
    if not key:
        return ""
    if len(key) <= keep:
        return "*" * len(key)
    return "*" * (len(key) - keep) + key[-keep:]


def verify_google_api_key() -> bool:
    """
    Check whether GOOGLE_API_KEY is configured and log a friendly confirmation.
    Returns True if present, False otherwise.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        logger.info(f"GOOGLE_API_KEY detected: {_mask_key(api_key)}")
        return True
    logger.error(
        "GOOGLE_API_KEY not found. Configure it with one of the following:\n"
        " - export GOOGLE_API_KEY='YOUR_KEY'   # temporary for current shell\n"
        " - or add it to a .env file in the project root:\n"
        "     GOOGLE_API_KEY=YOUR_KEY\n"
        "In Cloud Shell, you can also run:\n"
        "  echo \"export GOOGLE_API_KEY=YOUR_KEY\" >> ~/.bashrc && source ~/.bashrc"
    )
    return False


class AgentRecommendationSystem:
    """Main system class for agent recommendations using ADK."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the recommendation system using ADK App.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        else:
            logger.info(f"Using GOOGLE_API_KEY: {_mask_key(api_key)}")
        
        # Initialize components
        logger.info("Initializing Agent Recommendation System with ADK App...")
        
        # Memory
        registry_file = "data/agent_registry.json"
        self.registry_memory = AgentRegistryMemory(registry_file)
        
        # Tools
        self.agent_registry = AgentRegistry(self.registry_memory)
        
        # Set SSL environment variables to handle certificate issues
        # (for corporate networks or systems with SSL certificate problems)
        if not os.getenv("CURL_CA_BUNDLE") and not os.getenv("REQUESTS_CA_BUNDLE"):
            # Try to use system certs, but don't fail if unavailable
            pass
        
        self.vector_search = VectorSearchTool(
            model_name=self.config.get("vector_search", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.mcp_tools = MCPToolRegistry()
        
        # Agents (using ADK)
        gemini_config = self.config.get("gemini", {})
        model_name = gemini_config.get("model", "gemini-1.5-pro")
        
        self.goal_agent = ADKGoalSpecificationAgent(
            model_name=model_name,
            api_key=api_key
        )
        
        self.retrieval_agent = ADKRetrievalAgent(
            self.agent_registry,
            self.vector_search,
            model_name=model_name
        )
        
        ranking_config = self.config.get("ranking", {})
        self.ranking_agent = ADKRankingAgent(
            self.mcp_tools,
            model_name=model_name,
            api_key=api_key,
            similarity_weight=ranking_config.get("similarity_weight", 0.4),
            compatibility_weight=ranking_config.get("compatibility_weight", 0.3),
            complexity_weight=ranking_config.get("complexity_weight", 0.2),
            dependency_weight=ranking_config.get("dependency_weight", 0.1)
        )
        
        self.explanation_agent = ADKExplanationAgent(
            model_name=model_name,
            api_key=api_key
        )
        
        # Coordinator (using ADK with sub_agents)
        self.coordinator = ADKCoordinatorAgentV2(
            self.goal_agent,
            self.retrieval_agent,
            self.ranking_agent,
            self.explanation_agent,
            model_name=model_name
        )
        
        # Create ADK App with plugins
        self.app = create_recommendation_app(self.coordinator.coordinator, config_path)
        
        logger.info("System initialized successfully with ADK App")
    
    def recommend_agents(self, goal: str, top_k: int = 5) -> dict:
        """
        Get agent recommendations for a goal.
        
        Args:
            goal: High-level goal description
            top_k: Number of recommendations
            
        Returns:
            Recommendation result as dictionary
        """
        result = self.coordinator.recommend_agents(goal, top_k=top_k)
        return result.model_dump()
    
    def initialize_registry(self, sample_data_path: str = "data/sample_agents.json"):
        """Initialize the agent registry with sample data."""
        logger.info(f"Initializing registry from {sample_data_path}")
        
        if not Path(sample_data_path).exists():
            logger.warning(f"Sample data file not found: {sample_data_path}")
            return
        
        self.registry_memory.load_from_file(sample_data_path)
        
        # Update embeddings
        agents = self.registry_memory.get_all_agents()
        self.vector_search.update_agent_embeddings(agents)
        
        # Save with embeddings
        self.registry_memory.save_to_file()
        
        logger.info(f"Registry initialized with {len(agents)} agents")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agent Recommendation System")
    parser.add_argument("--goal", type=str, help="High-level goal description")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--init-registry", action="store_true", help="Initialize agent registry")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--check-api-key", action="store_true", help="Check GOOGLE_API_KEY presence and exit")
    
    args = parser.parse_args()
    
    try:
        if args.check_api_key:
            ok = verify_google_api_key()
            print("GOOGLE_API_KEY present" if ok else "GOOGLE_API_KEY missing")
            return
        
        system = AgentRecommendationSystem(config_path=args.config)
        
        if args.init_registry:
            system.initialize_registry()
            print("Registry initialized successfully!")
            return
        
        if args.goal:
            print(f"\nFinding agents for goal: {args.goal}\n")
            print("=" * 60)
            
            result = system.recommend_agents(args.goal, top_k=args.top_k)
            
            print(f"\n{result['explanation']}\n")
            print("=" * 60)
            print("\nRecommendations:\n")
            
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"{i}. {rec['agent']['name']}")
                print(f"   Similarity: {rec['similarity_score']:.2%}")
                print(f"   Reuse Difficulty: {rec['reuse_difficulty']}")
                print(f"   Compatibility: {rec['compatibility_score']:.2%}")
                print(f"   Explanation: {rec['explanation']}")
                if rec['required_modifications']:
                    print(f"   Modifications: {', '.join(rec['required_modifications'])}")
                print()
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

