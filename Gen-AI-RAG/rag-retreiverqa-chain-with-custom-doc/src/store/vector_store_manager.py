from langchain_community.vectorstores import FAISS
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

class VectorStoreManager:
    """
        Manages the vector store for the model.
    """
    
    def __init__(self, embedding_model) -> None:
        self.embedding_model=embedding_model
        self.vector_store=None
        self.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        
    def __create_vector_store(self,documents: List[Document]):
        """ Creates a new vector store."""
        if self.vector_store is not None:
            return self.vector_store
        try:
            self.vector_store=FAISS(
                documents=documents,                
                embedding_model=self.embedding_model)
        
        except Exception as ex:
            logger.error(f"Error creating vector store: {ex}")
            raise ex
        return self.vector_store
    
    def add_documents(self,documents: List[Document]):
        """ Adds documents to the vector store."""
        if self.vector_store is None:
            self.vector_store= self.__create_vector_store(self, documents=documents)
        
        try:
            splitted_docs=self.text_splitter.split_documents(documents=documents)
            self.vector_store.add_documents(splitted_docs)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
        
    def get_retriever(self):
        if self.vector_store is None:
            raise ValueError(" Vector store has not been created")
        return self.vector_store.as_retriever()
    