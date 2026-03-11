"""
Embedding Service for Knowledge Graph Evaluation

This module provides a unified interface for embedding services used in KG evaluation.
It supports both direct embedding service implementations and adapters for existing
LLM services that have embedding capabilities.

还可以集成其他的embedding的模型
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataflow.logger import get_logger


class EmbeddingServingABC(ABC):
    """
    Abstract base class for embedding services.
    
    This interface is designed for single-text embedding retrieval,
    which is commonly used in KG evaluation tasks.
    """
    
    @abstractmethod
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Get embedding vector for a single text.
        
        Args:
            text: Input text string
            model: Model name to use (default: "text-embedding-3-small")
            
        Returns:
            List of float values representing the embedding vector
        """
        raise NotImplementedError


class LLMServiceEmbeddingAdapter(EmbeddingServingABC):
    """
    Adapter class that adapts LLMServingABC services with embedding capabilities
    to the EmbeddingServingABC interface.
    """
    
    def __init__(self, llm_service, model_name: str = "text-embedding-3-small"):
        """
        Initialize the adapter.
        
        Args:
            llm_service: An LLMServingABC instance that has `generate_embedding_from_input` method
            model_name: Default model name to use for embeddings
        """
        self.llm_service = llm_service
        self.model_name = model_name
        self.logger = get_logger()
        
        # Verify that the service has the required method
        if not hasattr(llm_service, 'generate_embedding_from_input'):
            raise ValueError(
                f"LLM service {type(llm_service).__name__} does not have "
                f"`generate_embedding_from_input` method. "
                f"Please use a service that supports embedding generation."
            )
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Get embedding for a single text by calling the underlying LLM service.
        """
        if not text:
            self.logger.warning("Empty text provided for embedding")
            return []
        
        try:
            # Use the service's batch embedding method with a single text
            embeddings = self.llm_service.generate_embedding_from_input([text])
            
            if not embeddings or len(embeddings) == 0:
                self.logger.warning(f"Empty embedding result for text: {text[:50]}...")
                return []
            
            embedding = embeddings[0]
            
            # Ensure it's a list of floats
            if not isinstance(embedding, list):
                self.logger.error(f"Unexpected embedding type: {type(embedding)}")
                return []
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            return []


class OpenAIEmbeddingService(EmbeddingServingABC):
    """
    Direct embedding service implementation for OpenAI-compatible APIs.
    
    This is a standalone implementation that directly calls embedding API endpoints.
    Supports OpenAI API and compatible proxy services (e.g., aihubmix).
    Use this if you want a dedicated embedding service without going through
    the general LLM service interface.
    """
    
    def __init__(
        self,
        api_url: str = "https://api.openai.com/v1/embeddings",
        key_name_of_api_key: str = "DF_API_KEY",
        model_name: str = "text-embedding-3-small",
        max_retries: int = 3
    ):
        """
        Initialize embedding service.
        """
        import os
        from dataflow.serving import APILLMServing_request
        
        self.model_name = model_name
        self.max_retries = max_retries
        self.logger = get_logger()
        
        # Use APILLMServing_request as the underlying service
        # Configure it for embedding endpoint
        self._llm_service = APILLMServing_request(
            api_url=api_url,
            key_name_of_api_key=key_name_of_api_key,
            model_name=model_name,
            max_retries=max_retries
        )
        
        self._adapter = LLMServiceEmbeddingAdapter(self._llm_service, model_name)
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Get embedding for a single text.
        """
        use_model = model if model != "text-embedding-3-small" else self.model_name
        return self._adapter.get_embedding(text, use_model)
    
    def cleanup(self):
        if hasattr(self._llm_service, 'cleanup'):
            self._llm_service.cleanup()


class GoogleCloudVertexAIEmbeddingService(EmbeddingServingABC):
    """
    Direct embedding service implementation for Google Cloud Vertex AI Embeddings API.
    
    This service uses Google Cloud Vertex AI's embedding models, such as:
    - textembedding-gecko@003
    - textembedding-gecko@002
    - textembedding-gecko-multilingual@001
    
    Requires:
    - GOOGLE_APPLICATION_CREDENTIALS environment variable set to service account key file
    - GOOGLE_CLOUD_PROJECT environment variable (or pass project parameter)
    - Vertex AI API enabled in your GCP project
    """
    
    def __init__(
        self,
        project: Optional[str] = None,
        location: str = "us-central1",
        model_name: str = "textembedding-gecko@003",
        max_retries: int = 3
    ):
        """
        Initialize Google Cloud Vertex AI embedding service.
        
        Args:
            project: GCP project ID. If None, will use GOOGLE_CLOUD_PROJECT env var.
            location: GCP region (default: us-central1)
            model_name: Embedding model name (default: textembedding-gecko@003)
            max_retries: Maximum number of retry attempts
        """
        import os
        
        self.logger = get_logger()
        self.model_name = model_name
        self.max_retries = max_retries
        self.location = location
        
        # Get project from parameter or environment variable
        if project is None:
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            if project:
                self.logger.info(f"Using GOOGLE_CLOUD_PROJECT from environment: {project}")
        
        if project is None:
            raise ValueError(
                "Project ID is required. Either pass 'project' parameter or set "
                "GOOGLE_CLOUD_PROJECT environment variable."
            )
        
        self.project = project
        
        # Check for credentials
        google_app_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not google_app_credentials:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. "
                "Please set it to the path of your service account key file."
            )
        
        if not os.path.exists(google_app_credentials):
            raise ValueError(
                f"GOOGLE_APPLICATION_CREDENTIALS file not found: {google_app_credentials}. "
                "Please ensure the path is correct."
            )
        
        # Initialize Vertex AI
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
            
            vertexai.init(project=project, location=location)
            self._model = TextEmbeddingModel.from_pretrained(model_name)
            self.logger.info(
                f"Google Cloud Vertex AI Embedding Service initialized: "
                f"project={project}, location={location}, model={model_name}"
            )
        except ImportError:
            raise ImportError(
                "Google Cloud AI Platform library not found. "
                "Please run: pip install google-cloud-aiplatform"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Get embedding for a single text using Google Cloud Vertex AI.
        
        Args:
            text: Input text string
            model: Model name (ignored, uses instance model_name)
            
        Returns:
            List of float values representing the embedding vector
        """
        if not text:
            self.logger.warning("Empty text provided for embedding")
            return []
        
        try:
            # Vertex AI embedding model returns embeddings directly
            embeddings = self._model.get_embeddings([text])
            
            if not embeddings or len(embeddings) == 0:
                self.logger.warning(f"Empty embedding result for text: {text[:50]}...")
                return []
            
            # Extract the embedding vector
            embedding = embeddings[0].values
            
            # Ensure it's a list of floats
            if not isinstance(embedding, list):
                self.logger.error(f"Unexpected embedding type: {type(embedding)}")
                return []
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            return []
    
    def cleanup(self):
        """Cleanup resources if needed."""
        # Vertex AI doesn't require explicit cleanup
        pass


class HuggingFaceInferenceAPIEmbeddingService(EmbeddingServingABC):
    """
    Direct embedding service implementation for Hugging Face Inference API.
    
    This service uses Hugging Face's Inference API for embedding generation.
    Supports a wide variety of embedding models available on Hugging Face.
    
    Requires:
    - HF_API_TOKEN environment variable set to your Hugging Face API token
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_url: str = "https://api-inference.huggingface.co/pipeline/feature-extraction",
        key_name_of_api_key: str = "HF_API_TOKEN",
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        """
        Initialize Hugging Face Inference API embedding service.
        
        Args:
            model_name: Hugging Face model identifier (default: sentence-transformers/all-MiniLM-L6-v2)
            api_url: Base URL for Hugging Face Inference API
            key_name_of_api_key: Environment variable name for API token (default: HF_API_TOKEN)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        import os
        
        self.logger = get_logger()
        self.model_name = model_name
        self.api_url = api_url
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Get API token from environment
        self.api_token = os.environ.get(key_name_of_api_key)
        if self.api_token is None:
            raise ValueError(
                f"Lack of `{key_name_of_api_key}` in environment variables. "
                f"Please set `{key_name_of_api_key}` as your Hugging Face API token."
            )
        
        # Construct full API URL with model
        self.full_api_url = f"{api_url}/{model_name}"
        
        # Setup session with retry logic
        import requests
        self.requests = requests  # Store for use in exception handling
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        })
        
        self.logger.info(
            f"Hugging Face Inference API Embedding Service initialized: "
            f"model={model_name}, url={self.full_api_url}"
        )
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Get embedding for a single text using Hugging Face Inference API.
        
        Args:
            text: Input text string
            model: Model name (ignored, uses instance model_name)
            
        Returns:
            List of float values representing the embedding vector
        """
        if not text:
            self.logger.warning("Empty text provided for embedding")
            return []
        
        # Prepare request payload
        payload = {
            "inputs": text,
            "options": {
                "wait_for_model": True  # Wait if model is loading
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.full_api_url,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    embedding = response.json()
                    
                    # Handle different response formats
                    if isinstance(embedding, list):
                        # If it's a list, take the first element (for single input)
                        if len(embedding) > 0:
                            embedding = embedding[0]
                        else:
                            self.logger.warning("Empty embedding list in response")
                            return []
                    
                    # Ensure it's a list of floats
                    if not isinstance(embedding, list):
                        self.logger.error(f"Unexpected embedding type: {type(embedding)}")
                        return []
                    
                    return embedding
                
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    wait_time = 2 ** attempt
                    self.logger.info(
                        f"Model is loading, waiting {wait_time}s before retry "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    import time
                    time.sleep(wait_time)
                    continue
                
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text[:200]}"
                    self.logger.error(error_msg)
                    if attempt == self.max_retries - 1:
                        return []
                    continue
                    
            except self.requests.exceptions.Timeout:
                self.logger.warning(
                    f"Request timeout (attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt == self.max_retries - 1:
                    return []
                continue
                
            except Exception as e:
                self.logger.error(
                    f"Failed to generate embedding (attempt {attempt + 1}/{self.max_retries}): {e}",
                    exc_info=True
                )
                if attempt == self.max_retries - 1:
                    return []
                continue
        
        return []
    
    def cleanup(self):
        """Cleanup resources if needed."""
        if hasattr(self, 'session'):
            self.session.close()
