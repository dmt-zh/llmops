#!.venv/bin/python3.12

from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Configuration settings for the service.

    This class uses Pydantic's BaseSettings to manage configuration through environment
    variables and `.env` file. It includes settings for OpenAI API, Qdrant vector store,
    Opik monitoring, RAG system, and file paths.
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        extra='ignore',
    )

    # --- OpenAI Configuration ---
    LLM_MODEL_NAME: str
    LLM_API_KEY: SecretStr = Field(
        default='random_string',
        description='API secret key to get access to the LLM.',
    )
    LLM_BASE_URL: str = Field(
        default='http://127.0.0.1:8080',
        description='Connection URI for the LLM server.'
    )

    # --- Cache storage Configuration ---
    CACHE_STORAGE_DIR: str

    # --- Qdrant Configuration ---
    QDRANT_URL: str = Field(
        default='http://localhost:6333',
        description='Connection URI for the Qdrant DB.',
    )
    COLLECTION_NAME: str = Field(
        default='domain_knowledge',
        description='Named set of points (vectors).',
    )

    # --- Text splitter Configuration ---
    ENCODING_NAME: str = Field(
        default='cl100k_base',
        description='Tiktoken model name.',
    )
    CHUNK_SIZE: int = Field(
        default=300,
        description='Maximum size of text chunks.',
    )
    CHUNK_OVERLAP: int = Field(
        default=50,
        description='Overlap in characters between chunks.',
    )

    # --- HuggingFace Embeddings Configuration ---
    ENCODER_MODEL_NAME: str = Field(
        default='intfloat/multilingual-e5-large',
        description='Model name to use.',
    )
    EMBEDDING_SIZE: int = Field(
        default=1024,
        description='Embedding size model produces.',
    )
    MODEL_KWARGS: dict[str, Any] = Field(
        default={'trust_remote_code': True},
        description='Keyword arguments to pass to the Sentence Transformer model.',
    )
    ENCODE_KWARGS: dict[str, Any] = Field(
        default={
            'normalize_embeddings': True,
            'prompt': 'passage: ',
        },
        description='Keyword arguments to pass to the Sentence Transformer model.',
    )

    # --- Graph Configuration ---
    K_SEARCH_RESULTS: int = Field(
        default=5,
        description='How many results to provide after online searching.',
    )
    RECURSION_LIMIT: int = Field(
        default=10,
        description='Maximum number of times a call can recurse.',
    )
