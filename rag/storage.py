import gc
import logging
from pathlib import Path

import orjson
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic_settings import BaseSettings
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm

##############################################################################################

def _configure_splitter(settings: BaseSettings) -> RecursiveCharacterTextSplitter:
    """
    Configure a text splitter for document chunking.

    Creates a RecursiveCharacterTextSplitter that uses tiktoken encoding
    to split documents into appropriately sized chunks with overlap.
    """

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=settings.ENCODING_NAME,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

##############################################################################################

def _configure_embedder(settings: BaseSettings) -> HuggingFaceEmbeddings:
    """
    Configure a HuggingFace embedding model for text vectorization.

    Sets up an embedding model with specified model name, cache directory,
    and encoding parameters for converting text to vector representations.
    """

    return HuggingFaceEmbeddings(
        model_name=settings.ENCODER_MODEL_NAME,
        cache_folder=str(Path(settings.CACHE_STORAGE_DIR).resolve().joinpath('embeddings')),
        model_kwargs=settings.MODEL_KWARGS,
        encode_kwargs=settings.ENCODE_KWARGS,
    )

##############################################################################################

class QdrantStorage:
    """
    Wrapper class for managing Qdrant vector store operations.

    Provides a simplified interface for working with Qdrant vector store,
    including collection management and vector store operations using
    LangChain integration.
    """

    def __init__(self, settings: BaseSettings) -> None:
        self._qdrant_client = QdrantClient(url=settings.QDRANT_URL)
        self._text_splitter = _configure_splitter(settings)
        self._vector_store = self._init_vector_store(settings)
        self._settings = settings

    ##########################################################################################

    def _configure_vector_store(self, settings: BaseSettings) -> None:
        """
        Create or verify the existence of a Qdrant collection.
        Attempts to create a new collection with the specified configuration.
        If the collection already exists, logs a warning and continues.
        """

        try:
            self._qdrant_client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=VectorParams(size=settings.EMBEDDING_SIZE, distance=Distance.COSINE)
            )
            logging.info(f'Qdrant collection {settings.COLLECTION_NAME} is created!')
        except UnexpectedResponse as err:
            if f'`{settings.COLLECTION_NAME}` already exists' in str(err):
                logging.warning(
                    f'Qdrant collection `{settings.COLLECTION_NAME}` already exists. Skipping collection creation.'
                )
            else:
                logging.error(f'Error initializing Qdrant client: {err}')

    ##########################################################################################

    def _init_vector_store(self, settings: BaseSettings) -> QdrantVectorStore:
        """
        Creates a QdrantVectorStore with the specified collection, embedding model,
        and dense retrieval mode.
        """

        logging.info('Initialized Qdrant storage!')
        return QdrantVectorStore(
            client=self._qdrant_client,
            collection_name=settings.COLLECTION_NAME,
            embedding=_configure_embedder(settings),
            retrieval_mode=RetrievalMode.DENSE,
        )

    ##########################################################################################

    @property
    def get_retriever(self) -> QdrantVectorStore:
        """Get a configured retriever for similarity search."""

        return self._vector_store.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 4},
        )

    ##########################################################################################

    def create_collection(self) -> None:
        """
        Create a new Qdrant collection with the configured parameters.

        Creates a collection using the configured name and vector parameters from settings.
        The collection is created from documents loaded from JSON file.
        """

        self._configure_vector_store(self._settings)
        curr_dir = Path(__file__).resolve().parent
        save_path = curr_dir.parent.joinpath('data/main_dataset.json')
        with open(save_path) as fin:
            documents = orjson.loads(fin.read())
            for _, doc_content in tqdm(documents.items(), desc='Uploading docs to Qdrant', ncols=110, leave=False):
                document_chunks = self._text_splitter.create_documents(doc_content)
                self._vector_store.add_documents(document_chunks)

            logging.info('Uploaded vectores to DB!')
            gc.collect()

    ##########################################################################################

    def clear_collection(self) -> None:
        """
        Clear all data from the Qdrant collection.

        Deletes the existing collection and creates a new empty one with the same parameters.
        This is useful for resetting the vector store state.
        """

        logging.info(f'Clearing Qdrant collection {self._settings.COLLECTION_NAME}...')
        self.client.delete_collection(collection_name=self._settings.COLLECTION_NAME)

##############################################################################################
