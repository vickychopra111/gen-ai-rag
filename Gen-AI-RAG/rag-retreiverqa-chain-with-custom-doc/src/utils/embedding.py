from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger

class EmbeddingModel:
    """Singleton class for the embedding model."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            try:
                cls._instance.model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
            except Exception as e:
                logger.exception(f"Error loading embedding model: {e}")
                raise
        return cls._instance

    def get_model(self):
        """Returns the embedding model."""
        return self.model