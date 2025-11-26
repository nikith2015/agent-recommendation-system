"""Vector search tool for semantic agent retrieval."""

from typing import List, Tuple, Optional
import numpy as np
import logging
import os
import warnings
import sys
from io import StringIO
from ..models.agent_spec import AgentSpecification

logger = logging.getLogger(__name__)

# Try to import sentence_transformers with error handling
try:
    # Suppress SSL warnings BEFORE importing
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    warnings.filterwarnings('ignore', category=UserWarning, message='.*SSL.*')
    warnings.filterwarnings('ignore', message='.*certificate verify failed.*')
    warnings.filterwarnings('ignore', message='.*MaxRetryError.*')
    
    # Suppress stderr during import
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    finally:
        sys.stderr = old_stderr
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence_transformers not available - using keyword-based search only")
except Exception as e:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(f"Error loading sentence_transformers: {e} - using keyword-based search only")


class VectorSearchTool:
    """Tool for semantic vector search of agents."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize vector search tool.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model = None
        self.model_name = model_name
        self.embeddings_cache: dict = {}
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model with SSL error handling."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info("sentence_transformers not available - using keyword-based search only")
            self.model = None
            return
        
        # Suppress stderr during model loading to hide SSL errors
        old_stderr = sys.stderr
        suppressed_output = StringIO()
        
        try:
            # Suppress stderr temporarily
            sys.stderr = suppressed_output
            
            # Try normal initialization
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Successfully loaded model: {self.model_name}")
            except Exception as e:
                error_str = str(e)
                if "SSL" in error_str or "CERTIFICATE" in error_str or "certificate verify failed" in error_str:
                    # SSL error - try local cache
                    logger.info(f"SSL certificate issue detected, trying local cache...")
                    try:
                        # Try loading from local cache only
                        self.model = SentenceTransformer(self.model_name, local_files_only=True)
                        logger.info("Loaded model from local cache")
                    except Exception as cache_err:
                        # Model not in cache - use keyword fallback
                        logger.info("Model not in local cache - using keyword-based search fallback")
                        self.model = None
                else:
                    logger.warning(f"Error loading model {self.model_name}: {e}")
                    self.model = None
        finally:
            # Restore stderr
            sys.stderr = old_stderr
    
    def compute_embedding(self, text: str) -> List[float]:
        """
        Compute embedding for a text string.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (or zero vector if model not available)
        """
        if self.model is None:
            # Return a dummy embedding if model is not available
            logger.warning("Model not available, returning zero embedding")
            return [0.0] * 384  # Standard size for all-MiniLM-L6-v2
        
        if text not in self.embeddings_cache:
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()
            self.embeddings_cache[text] = embedding
        return self.embeddings_cache[text]
    
    def compute_agent_embedding(self, agent: AgentSpecification) -> List[float]:
        """
        Compute embedding for an agent specification.
        
        Args:
            agent: Agent specification
            
        Returns:
            Embedding vector
        """
        # Create a comprehensive text representation of the agent
        text_parts = [
            agent.name,
            agent.description,
            ", ".join([cap.value for cap in agent.capabilities]),
            agent.domain or "",
            ", ".join([tool.name for tool in agent.tools]),
        ]
        text = " ".join(text_parts)
        return self.compute_embedding(text)
    
    def semantic_search(
        self,
        query: str,
        agents: List[AgentSpecification],
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[AgentSpecification, float]]:
        """
        Perform semantic search to find similar agents.
        Falls back to keyword matching if model is not available.
        
        Args:
            query: Search query text
            agents: List of agents to search
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (agent, similarity_score) tuples, sorted by score
        """
        # Fallback to keyword matching if model is not available
        if self.model is None:
            logger.info("Model not available, using keyword-based fallback search")
            return self._keyword_search(query, agents, top_k, threshold)
        
        # Compute query embedding
        query_embedding = np.array(self.compute_embedding(query))
        
        # Compute similarities
        results = []
        for agent in agents:
            # Get or compute agent embedding
            if agent.embedding:
                agent_embedding = np.array(agent.embedding)
            else:
                agent_embedding = np.array(self.compute_agent_embedding(agent))
                agent.embedding = agent_embedding.tolist()
            
            # Compute cosine similarity
            similarity = np.dot(query_embedding, agent_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(agent_embedding)
            )
            
            if similarity >= threshold:
                results.append((agent, float(similarity)))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _keyword_search(
        self,
        query: str,
        agents: List[AgentSpecification],
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[AgentSpecification, float]]:
        """
        Fallback keyword-based search when vector model is unavailable.
        
        Args:
            query: Search query text
            agents: List of agents to search
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (agent, similarity_score) tuples, sorted by score
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for agent in agents:
            # Build searchable text from agent
            agent_text = " ".join([
                agent.name,
                agent.description,
                ", ".join([cap.value for cap in agent.capabilities]),
                agent.domain or "",
                ", ".join([tool.name for tool in agent.tools]),
            ]).lower()
            
            agent_words = set(agent_text.split())
            
            # Simple keyword matching score
            if query_words:
                matches = len(query_words.intersection(agent_words))
                similarity = matches / len(query_words)
            else:
                similarity = 0.0
            
            if similarity >= threshold:
                results.append((agent, float(similarity)))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def update_agent_embeddings(self, agents: List[AgentSpecification]) -> None:
        """
        Update embeddings for a list of agents.
        
        Args:
            agents: List of agents to update
        """
        for agent in agents:
            if not agent.embedding:
                agent.embedding = self.compute_agent_embedding(agent)


