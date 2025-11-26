# Agent Recommendation System

A production-ready AI-powered system that helps developers discover and reuse existing AI agents for their specific needs using natural language processing, semantic search, and intelligent ranking.

## Features

- **Natural Language Goal Processing**: Convert high-level goals into structured specifications
- **Semantic Vector Search**: Find similar agents using advanced embeddings
- **Intelligent Ranking**: Rank agents by reuse difficulty and compatibility
- **Detailed Explanations**: Get explanations for why agents are recommended
- **Production Ready**: Includes Docker, health checks, error handling, and security best practices

## Architecture

- **Google ADK (Agent Development Kit)**: Multi-agent orchestration
- **Gemini 1.5**: Natural language understanding
- **Sentence Transformers**: Semantic embeddings
- **ChromaDB**: Vector database for agent search
- **Flask**: Web API with real-time streaming responses

## Quick Start

### Prerequisites

- Python 3.10+
- Google API Key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Capstone
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

5. **Initialize agent registry**
   ```bash
   python src/main.py --init-registry
   ```

6. **Run the application**
   ```bash
   # Development mode
   python app.py
   
   # Production mode (with gunicorn)
   gunicorn --config gunicorn_config.py app:app
   ```

7. **Access the web UI**
   - Open http://localhost:5000 in your browser

## Production Deployment

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t agent-recommendation-system .
   ```

2. **Run with Docker**
   ```bash
   docker run -p 5000:5000 \
     -e GOOGLE_API_KEY=your_key_here \
     -v $(pwd)/data:/app/data \
     agent-recommendation-system
   ```

3. **Or use Docker Compose**
   ```bash
   # Set environment variables in .env file
   docker-compose up -d
   ```

### Google Cloud Deployment

Deploy to Vertex AI Agent Engine:

```bash
python deploy.py \
  --project-id your-project-id \
  --location us-central1 \
  --agent-id agent-recommendation-system
```

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GOOGLE_API_KEY` | Google API key for Gemini | Yes | - |
| `FLASK_SECRET_KEY` | Flask secret key | No | Auto-generated |
| `FLASK_DEBUG` | Enable debug mode | No | `False` |
| `PORT` | Server port | No | `5000` |
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated) | No | `*` |
| `GUNICORN_WORKERS` | Number of gunicorn workers | No | `4` |
| `LOG_LEVEL` | Logging level | No | `INFO` |

## API Endpoints

### Health Check
```bash
GET /api/health
```

### Readiness Probe
```bash
GET /api/ready
```

### Get Recommendations
```bash
POST /api/recommend
Content-Type: application/json

{
  "goal": "I need an agent that can summarize email threads",
  "top_k": 5
}
```

Response is streamed via Server-Sent Events (SSE) with real-time progress updates.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

```
.
├── app.py                 # Flask web application
├── src/
│   ├── agents/           # ADK agents (goal, retrieval, ranking, explanation)
│   ├── models/           # Pydantic models
│   ├── memory/           # Agent registry and session management
│   ├── tools/            # Tools (vector search, MCP, registry)
│   ├── observability/    # Logging and tracing
│   └── utils/            # Utilities (validators, exceptions)
├── config/               # Configuration files
├── data/                 # Agent registry data
├── templates/            # HTML templates
└── static/              # CSS and JavaScript
```

## Security

- ✅ No hardcoded secrets
- ✅ Input validation and sanitization
- ✅ Error message sanitization
- ✅ CORS configuration
- ✅ Request size limits
- ✅ Non-root Docker user

## Monitoring

- Health check endpoint: `/api/health`
- Readiness probe: `/api/ready`
- Structured logging with configurable levels
- OpenTelemetry tracing support

## Support

For issues and questions, please open an issue in the repository.


