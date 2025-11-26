"""Goal Specification Agent using ADK."""

from typing import Optional
from google.adk.agents import LlmAgent
from google.adk.models import Gemini as GeminiModel
from ..models.agent_spec import GoalSpecification, AgentCapability, ToolRequirement
from ..observability.logging import get_logger

logger = get_logger(__name__)


def _check_google_generativeai_available() -> bool:
    """
    Check if google.generativeai module is available.
    
    Returns:
        True if available, False otherwise
    """
    try:
        import google.generativeai
        return True
    except ImportError:
        return False


def _ensure_google_generativeai() -> None:
    """
    Ensure google.generativeai is available, attempting installation if needed.
    
    Raises:
        ImportError: If module cannot be imported or installed
    """
    if _check_google_generativeai_available():
        return
    
    # Try to install it
    logger.warning("google.generativeai not found, attempting installation...")
    import subprocess
    import sys
    
    try:
        # Try to install the package
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "google-generativeai>=0.3.0"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            logger.info("Successfully installed google-generativeai")
            # Verify it's now available
            if _check_google_generativeai_available():
                return
            else:
                raise ImportError("google.generativeai still not available after installation attempt")
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            logger.error(f"Failed to install google-generativeai: {error_msg}")
            raise ImportError(
                f"Could not install google-generativeai. Please install manually: "
                f"pip install google-generativeai\n"
                f"Error: {error_msg}"
            )
    except (subprocess.TimeoutExpired, TimeoutError, FileNotFoundError, PermissionError, OSError) as e:
        # Installation failed or not possible
        raise ImportError(
            f"Cannot install google-generativeai automatically. "
            f"Please install manually: pip install google-generativeai\n"
            f"Original error: {e}"
        ) from e


class ADKGoalSpecificationAgent:
    """Goal Specification Agent using ADK LlmAgent."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro", api_key: Optional[str] = None):
        """
        Initialize Goal Specification Agent using ADK.
        
        Args:
            model_name: Name of the Gemini model to use
            api_key: Optional API key (uses environment variable if not provided)
        """
        # Create Gemini model
        model = GeminiModel(model_name=model_name)
        
        # Create ADK agent with specialized instruction
        self.agent = LlmAgent(
            name="goal_specification_agent",
            instruction=self._create_agent_instruction(),
            model=model,
        )
    
    @property
    def adk_agent(self) -> LlmAgent:
        """Expose the underlying ADK agent for use as sub_agent."""
        return self.agent
    
    def specify_goal(self, goal: str) -> GoalSpecification:
        """
        Convert a high-level goal into a structured specification.
        
        Args:
            goal: High-level goal description
            
        Returns:
            Structured goal specification
        """
        logger.info(f"Converting goal to specification: {goal[:100]}...")
        
        # Create prompt for the agent
        prompt = f"""Analyze this goal and extract structured information:

Goal: {goal}

Return a JSON object with:
{{
  "capabilities": ["list", "of", "required", "capabilities"],
  "tool_requirements": [
    {{
      "name": "tool_name",
      "type": "MCP|custom|built-in|OpenAPI",
      "description": "what the tool does",
      "required": true
    }}
  ],
  "input_format": "description of expected input format",
  "output_format": "description of expected output format",
  "domain": "domain or industry context if applicable",
  "constraints": ["list", "of", "constraints"],
  "keywords": ["relevant", "keywords", "for", "search"]
}}

Available capabilities: text_processing, data_analysis, api_integration, code_execution, search, database_access, file_operations, email, calendar, web_scraping, image_processing, sentiment_analysis, translation, summarization

Return ONLY valid JSON."""
        
        try:
            # Use ADK agent to process
            # Note: Work around ADK library bug with parent_context being a string instead of model
            # by ensuring we're in a non-recording span context
            from opentelemetry import trace as otel_trace
            from opentelemetry.trace import NonRecordingSpan
            from opentelemetry.trace.span import INVALID_SPAN_CONTEXT
            
            # Use a non-recording span to avoid the model_copy error
            # This prevents ADK from trying to use parent_context
            with otel_trace.use_span(NonRecordingSpan(INVALID_SPAN_CONTEXT), end_on_exit=False):
                response_gen = self.agent.run_live(prompt)
            
            # Consume async generator
            from ..utils.async_helper import consume_async_generator
            result_text = consume_async_generator(response_gen)
            
            # Log the raw response for debugging
            logger.info(f"Raw response text length: {len(result_text) if result_text else 0}")
            if result_text:
                logger.info(f"Raw response text (first 500 chars): {result_text[:500]}")
            else:
                logger.warning("Empty response text received from consume_async_generator")
            
            if not result_text or not result_text.strip():
                raise ValueError("Empty response from agent - switching to fallback generator")
            
            # Parse and create specification
            import json
            spec_data = self._parse_response(result_text)
            goal_spec = self._create_goal_specification(goal, spec_data)
            
            logger.info(f"Successfully created specification with {len(goal_spec.capabilities)} capabilities")
            return goal_spec
            
        except Exception as e:
            logger.error(f"Error in goal specification: {e}")
            
            # Fallback path: use google.generativeai directly to avoid ADK streaming/context issues
            try:
                logger.warning("Falling back to google.generativeai for goal specification...")
                
                import os
                import json
                import sys
                import importlib
                
                # Check if google.generativeai is available, try to install if needed
                if not _check_google_generativeai_available():
                    logger.info("google.generativeai not found, attempting to ensure it's available...")
                    try:
                        _ensure_google_generativeai()
                    except ImportError as install_err:
                        # Installation failed - provide helpful error and let it fall through to basic fallback
                        logger.error(f"Could not ensure google.generativeai is available: {install_err}")
                        raise install_err
                
                # Try to handle conflicting google packages (cross-platform)
                # If a non-namespace 'google' package was imported earlier (can block subpackages),
                # unload it so native-namespace packages (PEP 420) work.
                if "google" in sys.modules:
                    google_mod = sys.modules["google"]
                    google_file = getattr(google_mod, "__file__", "")
                    # Some conflicting meta 'google' packages ship a real __init__.py; unload them
                    if google_file and google_file.endswith("__init__.py"):
                        # Unload google and its submodules
                        to_unload = [key for key in sys.modules.keys() if key.startswith("google.")]
                        for key in to_unload:
                            del sys.modules[key]
                        del sys.modules["google"]
                        importlib.invalidate_caches()

                # Now import should work
                import google.generativeai as genai
                
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise RuntimeError("GOOGLE_API_KEY not set for fallback path")
                
                genai.configure(api_key=api_key)
                
                # Try multiple model names
                model_names = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-1.5-flash-latest"]
                model = None
                model_error = None
                
                for model_name in model_names:
                    try:
                        model = genai.GenerativeModel(model_name)
                        # Test if model works with a simple request
                        test_response = model.generate_content("test")
                        if getattr(test_response, "text", None):
                            logger.info(f"Using model: {model_name}")
                            break
                    except Exception as model_test_err:
                        model_error = model_test_err
                        logger.debug(f"Model {model_name} failed: {model_test_err}")
                        continue
                
                if not model:
                    raise ValueError(f"None of the models worked. Last error: {model_error}")
                
                # Ask for strict JSON output
                fallback_prompt = f"""{self._create_agent_instruction()}

Return ONLY valid JSON (no markdown, no prose).

Goal: {goal}
"""
                response = model.generate_content(fallback_prompt)
                fallback_text = getattr(response, "text", None) or ""
                
                if not fallback_text:
                    raise ValueError("Fallback model returned empty response")
                
                logger.info(f"Fallback response length: {len(fallback_text)}")
                spec_data = self._parse_response(fallback_text)
                goal_spec = self._create_goal_specification(goal, spec_data)
                logger.info(f"Fallback specification created with {len(goal_spec.capabilities)} capabilities")
                return goal_spec
            
            except Exception as fallback_err:
                logger.error(f"Fallback path failed: {fallback_err}")
                logger.warning("Attempting basic fallback with minimal specification...")
                # Always try basic fallback as last resort
                try:
                    # Create a minimal specification based on the goal text
                    goal_spec = self._create_basic_goal_specification(goal)
                    logger.info(f"Basic fallback specification created with {len(goal_spec.capabilities)} capabilities")
                    return goal_spec
                except Exception as basic_err:
                    logger.error(f"Basic fallback also failed: {basic_err}")
                    # Last resort: create minimal specification with just the goal text
                    logger.warning("Using absolute minimal fallback specification")
                    try:
                        return GoalSpecification(
                            goal=goal,
                            capabilities=[AgentCapability.TEXT_PROCESSING],
                            tool_requirements=[],
                            keywords=goal.lower().split()[:5],
                            domain=None
                        )
                    except Exception as final_err:
                        # If even this fails, something is very wrong with the model
                        logger.critical(f"All fallbacks exhausted. Final error: {final_err}")
                        raise RuntimeError(
                            f"Goal specification completely failed. "
                            f"Primary: {e}. Fallback: {fallback_err}. Basic: {basic_err}. Final: {final_err}"
                        ) from e
    
    def _create_agent_instruction(self) -> str:
        """Create the instruction for the ADK agent."""
        return """You are an expert at analyzing AI agent requirements and converting high-level goals into structured specifications.

Your task is to:
1. Understand the intent behind user goals
2. Extract required capabilities, tools, and constraints
3. Identify input/output formats
4. Generate relevant keywords for search
5. Return structured JSON output

Be precise and thorough in your analysis."""
    
    def _parse_response(self, response_text: str) -> dict:
        """Parse the JSON response from the model."""
        import json
        
        if not response_text:
            raise ValueError("Empty response text")
        
        response_text = response_text.strip()
        
        # Log what we received for debugging
        logger.debug(f"Parsing response (length: {len(response_text)}): {response_text[:200]}...")
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            lines = lines[1:]  # Remove first line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove last line
            response_text = "\n".join(lines)
        
        # Find JSON object
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        
        if start_idx == -1 or end_idx == 0:
            # Log the actual response to help debug
            logger.error(f"No JSON found in response. Response was: {response_text[:500]}")
            raise ValueError(f"No JSON found in response. Response preview: {response_text[:200]}")
        
        json_str = response_text[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}. JSON string was: {json_str[:500]}")
            raise ValueError(f"Invalid JSON in response: {e}")
    
    def _create_goal_specification(self, goal: str, spec_data: dict) -> GoalSpecification:
        """Create GoalSpecification from parsed data."""
        # Convert capabilities
        capabilities = []
        for cap_str in spec_data.get("capabilities", []):
            try:
                # Ensure cap_str is a string, not a model
                if isinstance(cap_str, str):
                    capabilities.append(AgentCapability(cap_str))
                else:
                    capabilities.append(AgentCapability(str(cap_str)))
            except (ValueError, TypeError) as e:
                logger.warning(f"Unknown capability: {cap_str}, error: {e}")
        
        # Convert tool requirements
        tool_requirements = []
        for tool_data in spec_data.get("tool_requirements", []):
            try:
                # Ensure tool_data is a dict, not a string or model
                if isinstance(tool_data, dict):
                    tool_requirements.append(ToolRequirement(**tool_data))
                else:
                    logger.warning(f"Invalid tool_data type: {type(tool_data)}, skipping")
            except Exception as e:
                logger.warning(f"Error creating ToolRequirement from {tool_data}: {e}")
        
        # Ensure all fields are the correct type before creating GoalSpecification
        return GoalSpecification(
            goal=str(goal),  # Ensure goal is a string
            capabilities=capabilities,
            tool_requirements=tool_requirements,
            input_format=str(spec_data.get("input_format")) if spec_data.get("input_format") else None,
            output_format=str(spec_data.get("output_format")) if spec_data.get("output_format") else None,
            domain=str(spec_data.get("domain")) if spec_data.get("domain") else None,
            constraints=[str(c) for c in spec_data.get("constraints", [])],  # Ensure all constraints are strings
            keywords=[str(k) for k in spec_data.get("keywords", [])]  # Ensure all keywords are strings
        )
    
    def _create_basic_goal_specification(self, goal: str) -> GoalSpecification:
        """Create a minimal GoalSpecification from goal text using keyword matching."""
        logger.info("Creating basic goal specification from keywords...")
        
        goal_lower = goal.lower()
        
        # Map keywords to capabilities
        capability_keywords = {
            AgentCapability.TEXT_PROCESSING: ["text", "document", "content", "parse", "extract", "analyze text"],
            AgentCapability.DATA_ANALYSIS: ["data", "analyze", "statistics", "metrics", "dataset", "csv", "json"],
            AgentCapability.API_INTEGRATION: ["api", "rest", "http", "endpoint", "service", "integration"],
            AgentCapability.CODE_EXECUTION: ["code", "execute", "run", "script", "program", "python"],
            AgentCapability.SEARCH: ["search", "find", "query", "lookup", "retrieve"],
            AgentCapability.DATABASE_ACCESS: ["database", "db", "sql", "query", "store", "retrieve data"],
            AgentCapability.FILE_OPERATIONS: ["file", "read", "write", "save", "load", "upload", "download"],
            AgentCapability.EMAIL: ["email", "mail", "send", "message"],
            AgentCapability.CALENDAR: ["calendar", "schedule", "event", "meeting", "appointment"],
            AgentCapability.WEB_SCRAPING: ["scrape", "web", "html", "crawl", "extract from website"],
            AgentCapability.IMAGE_PROCESSING: ["image", "picture", "photo", "visual", "process image"],
            AgentCapability.SENTIMENT_ANALYSIS: ["sentiment", "emotion", "feeling", "opinion", "attitude"],
            AgentCapability.TRANSLATION: ["translate", "translation", "language", "convert language"],
            AgentCapability.SUMMARIZATION: ["summarize", "summary", "abstract", "condense", "brief"]
        }
        
        # Determine capabilities based on keywords
        capabilities = []
        for cap, keywords in capability_keywords.items():
            if any(keyword in goal_lower for keyword in keywords):
                try:
                    capabilities.append(cap)
                except (ValueError, TypeError):
                    pass
        
        # Default to general capabilities if nothing matched
        if not capabilities:
            capabilities = [AgentCapability.TEXT_PROCESSING]
        
        # Extract keywords from goal (simple word extraction)
        words = goal.split()
        keywords = [w.strip('.,!?;:()[]{}"\'').lower() 
                   for w in words 
                   if len(w.strip('.,!?;:()[]{}"\'')) > 3][:10]  # Top 10 keywords
        
        # Try to infer domain from keywords
        domain_keywords = {
            "finance": ["financial", "money", "payment", "transaction", "bank", "currency"],
            "healthcare": ["health", "medical", "patient", "diagnosis", "treatment"],
            "ecommerce": ["product", "order", "cart", "checkout", "purchase", "shopping"],
            "education": ["learn", "course", "student", "education", "tutorial", "lesson"],
            "social": ["social", "user", "profile", "friend", "community", "network"]
        }
        
        domain = None
        for d, keywords_list in domain_keywords.items():
            if any(kw in goal_lower for kw in keywords_list):
                domain = d
                break
        
        # Create minimal specification
        return GoalSpecification(
            goal=str(goal),
            capabilities=capabilities,
            tool_requirements=[],
            input_format=None,
            output_format=None,
            domain=domain,
            constraints=[],
            keywords=keywords
        )

