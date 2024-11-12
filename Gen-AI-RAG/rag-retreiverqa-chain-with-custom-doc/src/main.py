from utils.data_loading import DocumentLoaderFactory
from utils.embedding import EmbeddingModel
from utils.qa_chain import QaChainBuilder
from utils.elt import DocumentProcessor
from utils.vector_store_manager import VectorStoreManager  # Import the new class
from loguru import logger
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ... (Other imports remain the same) ...

 
       

def add_documents_to_vector_store(vector_store: Chroma, documents: List[Document]):
    """Adds new documents to the existing vector store."""
    try:
        vector_store.add_documents(documents)  # Directly add documents
        logger.info(f"Added {len(documents)} documents to the vector store.")
    except Exception as e:
        logger.exception(f"Error adding documents to vector store: {e}")
        raise

def main():
    try:
        # Initialize components
        loader_factory = DocumentLoaderFactory()
        embedding_model = EmbeddingModel().get_model()
        processor = DocumentProcessor()
        vector_store_manager = VectorStoreManager(embedding_model)

        # Load initial documents and create vector store
        initial_docs = loader_factory.load_documents(
            'https://lilianweng.github.io/posts/2017-06-21-overview/'
        )
        vector_store = Chroma.from_documents(
            documents=initial_docs, embedding=embedding_model
        )

        # Create QA chain
        qa_chain = QaChainBuilder().with_llm(
            HuggingFacePipeline(pipeline=pipe)
        ).with_retriever(
            vector_store.as_retriever()
        ).build()

        # Example usage of adding documents:
        new_docs = loader_factory.load_documents("path/to/your/document.pdf")
        processed_docs = processor.process_documents(new_docs)
        add_documents_to_vector_store(vector_store, processed_docs)

        # ... (Rest of your code for querying and using the QA chain) ...

    except Exception as e:
        logger.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    logger.add("app.log", rotation="500 MB")
    main()