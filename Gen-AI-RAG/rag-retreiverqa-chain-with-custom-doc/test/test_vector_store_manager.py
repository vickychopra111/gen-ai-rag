import unittest
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.store.vector_store_manager import VectorStoreManager  # Replace your_module

class TestVectorStoreManager(unittest.TestCase):

    def test_create_vector_store(self):
        embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        manager = VectorStoreManager(embedding_model)
        documents = [Document(page_content="Test document")]
        manager.create_vector_store(documents)
        self.assertIsNotNone(manager.vector_store)

    def test_add_documents(self):
        embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        manager = VectorStoreManager(embedding_model)
        documents = [Document(page_content="Test document")]
        manager.create_vector_store(documents)
        new_documents = [Document(page_content="Another document")]
        manager.add_documents(new_documents)
        # Add assertions to check if documents were added

    def test_get_retriever(self):
        embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        manager = VectorStoreManager(embedding_model)
        documents = [Document(page_content="Test document")]
        manager.create_vector_store(documents)
        retriever = manager.get_retriever()
        self.assertIsNotNone(retriever)