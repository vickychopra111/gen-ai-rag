from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from typing import List
from langchain.schema import Document
import bs4
from loguru import logger

class DocumentLoaderFactory:
    
    """
    """
    def load_documents(self, file_path:str)-> List[Document]:
        if len(file_path) == 0:
            raise ValueError("File name cannot be empty")
        
        try:
            loader=None
            if file_path.endswith(".pdf"):
                loader=WebBaseLoader(
                    web_path=file_path,
                    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    ))
                )
            elif file_path.startswith("http"):
                loader=PyPDFLoader(file_path)
            else:
                raise ValueError("File path must be a pdf file or a url")
            
            return loader.load()
        except Exception as e:
            logger.exception(f"Error loading documents from {file_path}: {e}")
            
    
    