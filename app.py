"""Flask web application for Agent Recommendation System - Production Ready."""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify, stream_with_context, Response
import traceback
from dotenv import load_dotenv
from src.utils.validators import validate_goal, validate_top_k, sanitize_error_message
from src.utils.exceptions import ValidationError, ConfigurationError

# Try to import CORS (optional for local demo)
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("[WARN] flask-cors not installed. CORS disabled (OK for local demo)")

# Load environment variables
load_dotenv()

# Validate API key is set (never hardcode!)
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is required. "
        "Please set it in your .env file or environment."
    )

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__)

# Production configuration
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(32).hex())
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# CORS configuration for production
if CORS_AVAILABLE:
    CORS(app, resources={
        r"/api/*": {
            "origins": os.getenv('ALLOWED_ORIGINS', '*').split(','),
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

# Global system instance
_system = None
_logs = []

def get_logger():
    """Simple logger that captures logs for UI."""
    class UILogger:
        def __init__(self):
            self.logs = []
        
        def info(self, msg):
            self.logs.append(("info", msg))
            print(f"[INFO] {msg}")
        
        def warning(self, msg):
            self.logs.append(("warning", msg))
            print(f"[WARN] {msg}")
        
        def error(self, msg):
            self.logs.append(("error", msg))
            print(f"[ERROR] {msg}")
        
        def clear(self):
            self.logs = []
    
    return UILogger()

def get_system():
    """Get or create the agent recommendation system."""
    global _system
    if _system is None:
        try:
            # Suppress warnings during import
            import warnings
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*SSL.*')
            warnings.filterwarnings('ignore', message='.*certificate verify failed.*')
            warnings.filterwarnings('ignore', message='.*MaxRetryError.*')
            
            from src.main import AgentRecommendationSystem
            print("[INFO] Initializing Agent Recommendation System...")
            _system = AgentRecommendationSystem()
            print("[INFO] System initialized successfully!")
        except ImportError as e:
            error_msg = str(e)
            print(f"[ERROR] Missing dependency: {error_msg}")
            print("[ERROR] Please install all requirements:")
            print("[ERROR]   C:\\venv_capstone\\Scripts\\pip.exe install -r requirements.txt")
            print(f"[ERROR] Or using system Python:")
            print("[ERROR]   pip install -r requirements.txt")
            raise
        except Exception as e:
            print(f"[ERROR] Error initializing system: {e}")
            traceback.print_exc()
            raise
    return _system

class MockSystem:
    """Mock system for demo when full dependencies aren't available."""
    def __init__(self):
        self.goal_agent = MockGoalAgent()
        self.retrieval_agent = MockRetrievalAgent()
        self.ranking_agent = MockRankingAgent()
        self.explanation_agent = MockExplanationAgent()
    
    def recommend_agents(self, goal, top_k=5):
        """Provide mock recommendations for demo."""
        from src.models.agent_spec import GoalSpecification, AgentCapability, RecommendationResult, Recommendation
        
        # Create basic goal spec
        goal_spec = GoalSpecification(
            goal=goal,
            capabilities=[AgentCapability.TEXT_PROCESSING, AgentCapability.SUMMARIZATION],
            tool_requirements=[],
            keywords=goal.lower().split()[:10],
            domain=None
        )
        
        # Create mock recommendations
        recommendations = []
        for i in range(min(top_k, 3)):
            from src.models.agent_spec import AgentSpecification, ReuseDifficulty, AgentRecommendation
            mock_agent = AgentSpecification(
                id=f"demo_agent_{i+1}",
                name=f"Demo Agent {i+1}",
                description=f"A sample agent for demonstration purposes",
                capabilities=[AgentCapability.TEXT_PROCESSING, AgentCapability.SUMMARIZATION],
                tools=[],
                complexity_score=0.5 + (i * 0.1)
            )
            
            rec = AgentRecommendation(
                agent=mock_agent,
                similarity_score=0.9 - (i * 0.1),
                reuse_difficulty=ReuseDifficulty.EASY if i == 0 else ReuseDifficulty.MODERATE,
                reuse_difficulty_score=0.2 if i == 0 else 0.5,
                compatibility_score=0.85 - (i * 0.1),
                explanation=f"This is a demo recommendation {i+1}. Install all dependencies for real agent recommendations.",
                required_modifications=[]
            )
            recommendations.append(rec)
        
        return RecommendationResult(
            goal_spec=goal_spec,
            recommendations=recommendations,
            explanation="These are demo recommendations. For real agent recommendations, please install all dependencies: pip install -r requirements.txt",
            retrieval_metadata={"demo_mode": True}
        )

class MockGoalAgent:
    def specify_goal(self, goal):
        from src.models.agent_spec import GoalSpecification, AgentCapability
        return GoalSpecification(
            goal=goal,
            capabilities=[AgentCapability.TEXT_PROCESSING],
            tool_requirements=[],
            keywords=goal.lower().split()[:10]
        )

class MockRetrievalAgent:
    def retrieve_similar_agents(self, goal_spec, top_k=20):
        return []

class MockRankingAgent:
    def rank_agents(self, goal_spec, candidates, top_k=5):
        return []

class MockExplanationAgent:
    def explain_recommendations(self, goal_spec, recommendations):
        return recommendations
    
    def generate_overall_explanation(self, goal_spec, recommendations):
        return "Demo mode: Install full dependencies for detailed explanations."

@app.route('/')
def index():
    """Render the main UI page."""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint for production monitoring."""
    try:
        # Check if system can be initialized (lightweight check)
        system = get_system()
        return jsonify({
            "status": "healthy",
            "message": "Agent Recommendation System API",
            "version": os.getenv('APP_VERSION', '1.0.0')
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "message": "Service initialization failed"
        }), 503


@app.route('/api/ready', methods=['GET'])
def ready():
    """Readiness probe endpoint for Kubernetes/Docker."""
    try:
        # Verify API key is configured
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return jsonify({
                "status": "not_ready",
                "message": "GOOGLE_API_KEY not configured"
            }), 503
        
        return jsonify({
            "status": "ready",
            "message": "Service is ready to accept requests"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "not_ready",
            "message": str(e)
        }), 503

@app.route('/api/recommend', methods=['GET', 'POST'])
def recommend():
    """Get agent recommendations with real-time updates."""
    try:
        # Handle both GET (query params) and POST (JSON body)
        if request.method == 'GET':
            goal = request.args.get('goal', '').strip()
            top_k_param = request.args.get('top_k', 5)
        else:
            data = request.get_json() or {}
            goal = data.get('goal', '').strip()
            top_k_param = data.get('top_k', 5)
        
        # Validate input
        is_valid, error_msg = validate_goal(goal)
        if not is_valid:
            return jsonify({
                "error": error_msg,
                "status": "error"
            }), 400
        
        is_valid, error_msg, top_k = validate_top_k(top_k_param)
        if not is_valid:
            return jsonify({
                "error": error_msg,
                "status": "error"
            }), 400
        
        logger = get_logger()
        
        def generate():
            """Generate response with real-time updates."""
            try:
                # Step 1: Initialization - Send initial heartbeat immediately
                try:
                    yield f"data: {json.dumps({'step': 'init', 'message': 'Initializing system...', 'progress': 10})}\n\n"
                except Exception as yield_err:
                    logger.error(f"Error sending initial message: {yield_err}")
                    return
                
                try:
                    system = get_system()
                    yield f"data: {json.dumps({'step': 'init', 'message': 'System ready', 'progress': 20})}\n\n"
                except Exception as init_err:
                    error_msg = f"System initialization failed: {str(init_err)}"
                    logger.error(error_msg)
                    error_details = traceback.format_exc()
                    yield f"data: {json.dumps({'step': 'error', 'message': error_msg, 'progress': 0, 'error': True, 'details': error_details})}\n\n"
                    return
                
                # Step 2: Goal Specification
                yield f"data: {json.dumps({'step': 'goal_spec', 'message': 'Analyzing your goal...', 'progress': 30})}\n\n"
                
                try:
                    goal_spec = system.goal_agent.specify_goal(goal)
                    yield f"data: {json.dumps({'step': 'goal_spec', 'message': f'Goal analyzed: {len(goal_spec.capabilities)} capabilities identified', 'progress': 40, 'goal_spec': {'capabilities': [c.value for c in goal_spec.capabilities], 'keywords': goal_spec.keywords[:5]}})}\n\n"
                except Exception as e:
                    logger.error(f"Goal specification error: {e}")
                    yield f"data: {json.dumps({'step': 'goal_spec', 'message': f'Using fallback goal specification', 'progress': 40, 'warning': str(e)})}\n\n"
                
                # Step 3: Retrieval
                yield f"data: {json.dumps({'step': 'retrieval', 'message': 'Searching for similar agents...', 'progress': 50})}\n\n"
                
                try:
                    candidates = system.retrieval_agent.retrieve_similar_agents(goal_spec, top_k=20)
                    yield f"data: {json.dumps({'step': 'retrieval', 'message': f'Found {len(candidates)} candidate agents', 'progress': 60, 'candidates_count': len(candidates)})}\n\n"
                except Exception as e:
                    logger.error(f"Retrieval error: {e}")
                    candidates = []
                    yield f"data: {json.dumps({'step': 'retrieval', 'message': 'Using alternative retrieval method', 'progress': 60, 'warning': str(e)})}\n\n"
                
                if not candidates:
                    yield f"data: {json.dumps({'step': 'error', 'message': 'No candidate agents found', 'progress': 100, 'error': True})}\n\n"
                    return
                
                # Step 4: Ranking
                yield f"data: {json.dumps({'step': 'ranking', 'message': 'Ranking agents by reuse difficulty...', 'progress': 70})}\n\n"
                
                try:
                    ranked = system.ranking_agent.rank_agents(goal_spec, candidates, top_k=top_k)
                    yield f"data: {json.dumps({'step': 'ranking', 'message': f'Ranked {len(ranked)} agents', 'progress': 80})}\n\n"
                except Exception as e:
                    logger.error(f"Ranking error: {e}")
                    ranked = []
                    yield f"data: {json.dumps({'step': 'ranking', 'message': 'Using alternative ranking', 'progress': 80, 'warning': str(e)})}\n\n"
                
                # Step 5: Explanation
                yield f"data: {json.dumps({'step': 'explanation', 'message': 'Generating explanations...', 'progress': 90})}\n\n"
                
                try:
                    explained = system.explanation_agent.explain_recommendations(goal_spec, ranked)
                    overall_explanation = system.explanation_agent.generate_overall_explanation(goal_spec, explained)
                    yield f"data: {json.dumps({'step': 'explanation', 'message': 'Explanations generated', 'progress': 95})}\n\n"
                except Exception as e:
                    logger.error(f"Explanation error: {e}")
                    explained = ranked
                    overall_explanation = "Generated recommendations based on similarity and compatibility."
                    yield f"data: {json.dumps({'step': 'explanation', 'message': 'Using default explanations', 'progress': 95, 'warning': str(e)})}\n\n"
                
                # Step 6: Final result
                result = {
                    "goal_spec": {
                        "goal": goal_spec.goal if hasattr(goal_spec, 'goal') else goal,
                        "capabilities": [c.value for c in goal_spec.capabilities] if hasattr(goal_spec, 'capabilities') else [],
                        "keywords": goal_spec.keywords[:10] if hasattr(goal_spec, 'keywords') else []
                    },
                    "recommendations": [
                        {
                            "agent": {
                                "name": rec.agent.name if hasattr(rec.agent, 'name') else str(rec.agent),
                                "description": rec.agent.description if hasattr(rec.agent, 'description') else "",
                                "capabilities": [c.value for c in rec.agent.capabilities] if hasattr(rec.agent, 'capabilities') else []
                            },
                            "similarity_score": float(rec.similarity_score) if hasattr(rec, 'similarity_score') else 0.0,
                            "reuse_difficulty": rec.reuse_difficulty.value if hasattr(rec, 'reuse_difficulty') else "unknown",
                            "compatibility_score": float(rec.compatibility_score) if hasattr(rec, 'compatibility_score') else 0.0,
                            "explanation": rec.explanation if hasattr(rec, 'explanation') else "",
                            "required_modifications": rec.required_modifications if hasattr(rec, 'required_modifications') else []
                        }
                        for rec in explained
                    ],
                    "explanation": overall_explanation,
                    "status": "success"
                }
                
                yield f"data: {json.dumps({'step': 'complete', 'message': 'Recommendations ready!', 'progress': 100, 'result': result})}\n\n"
                
            except Exception as e:
                # Log full error details internally
                error_details = traceback.format_exc()
                logger.error(f"Processing error: {error_details}")
                
                # Send sanitized error to client
                safe_error = sanitize_error_message(e, include_details=False)
                try:
                    yield f"data: {json.dumps({'step': 'error', 'message': safe_error, 'progress': 0, 'error': True})}\n\n"
                except Exception as yield_err:
                    logger.error(f"Failed to send error message: {yield_err}")
                    return
        
        try:
            response = Response(stream_with_context(generate()), mimetype='text/event-stream')
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['X-Accel-Buffering'] = 'no'
            response.headers['Connection'] = 'keep-alive'
            return response
        except Exception as e:
            logger.error(f"Error creating response: {e}")
            return jsonify({
                "error": f"Failed to create streaming response: {str(e)}",
                "status": "error",
                "details": traceback.format_exc()
            }), 500
        
    except ValidationError as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return jsonify({
            "error": "Service configuration error",
            "status": "error"
        }), 500
    except Exception as e:
        # Log full error internally
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        # Return sanitized error to client
        safe_error = sanitize_error_message(e, include_details=False)
        return jsonify({
            "error": safe_error,
            "status": "error"
        }), 500

@app.route('/api/init-registry', methods=['POST'])
def init_registry():
    """Initialize the agent registry."""
    try:
        # Only initialize if we can get the system (has all dependencies)
        try:
            system = get_system()
            system.initialize_registry()
            return jsonify({
                "status": "success",
                "message": "Registry initialized successfully"
            })
        except Exception as e:
            error_msg = str(e)
            if "sentence_transformers" in error_msg or "chromadb" in error_msg:
                return jsonify({
                    "status": "error",
                    "error": "Missing dependencies. Please run: pip install -r requirements.txt",
                    "details": error_msg
                }), 500
            raise
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print(f"\n{'='*70}")
    print(" " * 15 + "Agent Recommendation System - Demo UI")
    print(f"{'='*70}")
    print(f"\n✓ Server starting on http://localhost:{port}")
    print(f"\n📝 Next steps:")
    print(f"   1. Open your browser")
    print(f"   2. Navigate to: http://localhost:{port}")
    print(f"   3. Enter a goal and click 'Get Recommendations'")
    print(f"\n💡 Example goals:")
    print(f"   - I need an agent that can summarize email threads")
    print(f"   - Create an agent for analyzing customer feedback")
    print(f"   - I want an agent that can translate documents")
    print(f"\n{'='*70}\n")
    
    try:
        # Warn if not using venv Python (development only)
        if debug_mode:
            python_exe = sys.executable.lower()
            if 'venv' not in python_exe and 'env' not in python_exe:
                print("\n[WARN] Consider using a virtual environment for development")
        
        # Test system initialization before starting server
        print("[INFO] Testing system initialization...")
        try:
            test_system = get_system()
            print("[INFO] System initialization test passed!")
        except Exception as e:
            error_msg = str(e)
            print(f"[WARN] System initialization test failed: {error_msg}")
            
            if "sentence_transformers" in error_msg or "No module named" in error_msg:
                print("\n[ERROR] Missing dependencies detected!")
                print("[ERROR] Please install dependencies:")
                print("[ERROR]   pip install -r requirements.txt")
                print("[WARN] Server will start but will fail when processing requests.\n")
            else:
                print("[WARN] Server will start but may fail when processing requests.")
                print("[WARN] Error details:")
                traceback.print_exc()
        
        # Production vs development mode
        debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        if debug_mode:
            print(f"\n[INFO] Starting Flask server in DEBUG mode on http://0.0.0.0:{port}")
            app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
        else:
            print(f"\n[INFO] Starting Flask server in PRODUCTION mode on http://0.0.0.0:{port}")
            print("[INFO] For production, use a WSGI server like gunicorn:")
            print("[INFO]   gunicorn -w 4 -b 0.0.0.0:5000 app:app")
            app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(" " * 20 + "Server stopped")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"\n\n{'='*70}")
        print(f"ERROR: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check if port {port} is available")
        print(f"  2. Make sure all dependencies are installed: pip install -r requirements.txt")
        print(f"  3. Verify GOOGLE_API_KEY is set")
        print(f"{'='*70}\n")
        raise

