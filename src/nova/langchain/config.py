import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
# from langchain.embeddings import HuggingFaceEmbeddings # Keep for potential fallback? No, replace fully for now
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOllama
# from langchain.llms import Ollama # Unused import

# Import NVIDIA classes
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# Import HuggingFaceEmbeddings for fallback or alternative
from langchain_community.embeddings import HuggingFaceEmbeddings


# Function to check NVIDIA API Key (avoids running getpass at import time)
def get_nvidia_api_key() -> Optional[str]:
    key = os.environ.get("NVIDIA_API_KEY")
    if key and key.startswith("nvapi-"):
        return key
    return None

class LangChainConfig(BaseModel):
    """Configuration for LangChain components, supporting Ollama and NVIDIA NIM."""
    
    # Provider Selection
    llm_provider: str = Field(default="ollama", description="LLM provider ('ollama' or 'nvidia')")
    
    # LLM Configuration (Common)
    llm_temperature: float = Field(default=0.1, description="Temperature for LLM generation")
    llm_max_tokens: int = Field(default=2048, description="Maximum tokens for LLM generation")

    # Ollama Specific Configuration
    ollama_llm_model: str = Field(default="llama2", description="Base model name for Ollama LLM")
    
    # NVIDIA Specific Configuration
    # Check for API key at runtime in get_llm/get_embeddings if provider is nvidia
    nvidia_llm_model: str = Field(default="meta/llama3-8b-instruct", description="Model name for NVIDIA LLM (used if provider is 'nvidia')")
    nvidia_nim_base_url: Optional[str] = Field(default=None, description="Base URL for self-hosted NVIDIA NIM (e.g., 'http://localhost:8000/v1')")
    # nvidia_api_key: Optional[str] = Field(default_factory=get_nvidia_api_key, description="NVIDIA API Key (fetched from env)") # Store it if needed, but use check in methods

    # Embedding Configuration
    embedding_provider: str = Field(default="huggingface", description="Embedding provider ('huggingface' or 'nvidia')")
    hf_embedding_model: str = Field(default="sentence-transformers/all-mpnet-base-v2", 
                                description="Model for HuggingFace text embeddings")
    hf_embedding_device: str = Field(default="cpu", description="Device for HuggingFace embedding model")
    nvidia_embedding_model: str = Field(default="nvidia/nv-embed-v1", description="Model name for NVIDIA Embeddings") # Example, user might need to change
    # nvidia_embedding_nim_base_url: Optional[str] = Field(default=None, description="Base URL for self-hosted NVIDIA Embedding NIM (if different from LLM NIM)") # Use nvidia_nim_base_url for simplicity for now


    # Vector Store Configuration
    vector_store_path: str = Field(default="data/vector_store", 
                                  description="Path to store/load FAISS vector database")
    
    # RAG Configuration
    rag_chunk_size: int = Field(default=1000, description="Size of text chunks for RAG")
    rag_chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    
    # Web Search Configuration
    web_search_enabled: bool = Field(default=True, description="Whether to enable web search")
    web_search_max_results: int = Field(default=5, description="Maximum number of search results")
    
    def get_llm(self, streaming: bool = False) -> Any:
        """Get configured LLM instance based on the provider."""
        if self.llm_provider == "nvidia":
            api_key = get_nvidia_api_key()
            if not api_key and not self.nvidia_nim_base_url:
                raise ValueError("NVIDIA provider selected, but NVIDIA_API_KEY env var is not set or invalid, and no nvidia_nim_base_url provided.")
            
            # Prioritize NIM base URL if provided
            if self.nvidia_nim_base_url:
                 print(f"Using NVIDIA NIM at: {self.nvidia_nim_base_url} with model {self.nvidia_llm_model}")
                 return ChatNVIDIA(
                    model=self.nvidia_llm_model,
                    nvidia_api_key=api_key, # Pass key even for NIM if available, might be needed
                    base_url=self.nvidia_nim_base_url,
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens,
                    streaming=streaming,
                    # top_p, etc. can be added if needed
                 )
            else:
                 # Use NVIDIA API Catalog
                 print(f"Using NVIDIA API Catalog with model {self.nvidia_llm_model}")
                 return ChatNVIDIA(
                    model=self.nvidia_llm_model,
                    nvidia_api_key=api_key, # Required for API catalog
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens,
                    streaming=streaming,
                 )
        elif self.llm_provider == "ollama":
            print(f"Using Ollama with model {self.ollama_llm_model}")
            return ChatOllama(
                model=self.ollama_llm_model,
                temperature=self.llm_temperature,
                # Ollama doesn't directly use max_tokens in constructor, it's often a generation param
                # max_tokens=self.llm_max_tokens, 
                streaming=streaming
            )
        else:
            raise ValueError(f"Unsupported llm_provider: {self.llm_provider}")
    
    def get_embeddings(self) -> Any: # Return type depends on provider
        """Get configured embeddings model based on the provider."""
        if self.embedding_provider == "nvidia":
            api_key = get_nvidia_api_key()
            if not api_key and not self.nvidia_nim_base_url:
                 raise ValueError("NVIDIA embedding provider selected, but NVIDIA_API_KEY env var is not set or invalid, and no nvidia_nim_base_url provided.")

            # Assume embedding NIM is at the same base URL as LLM NIM if nvidia_nim_base_url is set
            base_url = self.nvidia_nim_base_url # Use the main NIM URL

            if base_url:
                print(f"Using NVIDIA Embedding NIM at: {base_url} with model {self.nvidia_embedding_model}")
                return NVIDIAEmbeddings(
                    model=self.nvidia_embedding_model, 
                    nvidia_api_key=api_key, # Pass key if available
                    base_url=base_url
                )
            else:
                # Use NVIDIA API Catalog for embeddings
                print(f"Using NVIDIA API Catalog Embeddings with model {self.nvidia_embedding_model}")
                return NVIDIAEmbeddings(
                    model=self.nvidia_embedding_model, 
                    nvidia_api_key=api_key # Required for API catalog
                )

        elif self.embedding_provider == "huggingface":
             print(f"Using HuggingFace Embeddings with model {self.hf_embedding_model}")
             return HuggingFaceEmbeddings(
                 model_name=self.hf_embedding_model,
                 model_kwargs={"device": self.hf_embedding_device}
             )
        else:
            raise ValueError(f"Unsupported embedding_provider: {self.embedding_provider}")

    # Vector store now needs to handle different embedding types
    def get_vector_store(self, embeddings: Optional[Any] = None) -> FAISS:
        """Get configured vector store. Loads from path if exists, otherwise initializes empty.
        Requires appropriate embeddings instance."
        if embeddings is None:
            embeddings = self.get_embeddings()
        
        # Check if path exists and load, otherwise create empty? Langchain FAISS loader expects existing index.
        # For now, stick to load_local, assuming index is pre-built or managed elsewhere.
        # Need to ensure the index was built with the SAME embedding model being loaded.
        try:
            # FAISS.load_local requires allow_dangerous_deserialization=True for loading HF embeddings pickle
            allow_dangerous = isinstance(embeddings, HuggingFaceEmbeddings)
            print(f"Loading FAISS vector store from {self.vector_store_path} with {'HuggingFace' if allow_dangerous else 'NVIDIA'} embeddings.")
            return FAISS.load_local(
                self.vector_store_path, 
                embeddings, 
                allow_dangerous_deserialization=allow_dangerous # Set based on embedding type
            )
        except Exception as e:
            # Handle case where the vector store doesn't exist or loading fails
            # Option 1: Raise error
            # raise FileNotFoundError(f"Failed to load FAISS index from {self.vector_store_path}. Ensure it exists and was created with the correct embedding model. Error: {e}")
            # Option 2: Return None or an empty store (requires different handling downstream)
            print(f"Warning: Could not load FAISS index from {self.vector_store_path}. Returning None. Error: {e}")
            # You might need to adjust how the vector store is used if it can be None.
            # Or, implement logic to create a new empty store here if needed.
            return None # Adjust downstream code to handle None vector_store"""