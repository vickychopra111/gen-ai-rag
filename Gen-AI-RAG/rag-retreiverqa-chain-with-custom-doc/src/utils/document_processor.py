from typing import List
from langchain.schema import Document
import re

class DocumentProcessor:
    """
    Class for performing ELT operations on documents before adding them to the vector store.
    """

    def __init__(self):
        pass  # You can add initialization logic here if needed

    def extract(self, document: Document) -> str:
        """
        Extracts the relevant content from the document.
        """
        return document.page_content

    def transform(self, text: str) -> str:
        """
        Transforms the extracted text.
        """
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        return text

    def load(self, document: Document, transformed_text: str) -> Document:
        """
        Loads the transformed text back into the document object.
        """
        document.page_content = transformed_text  # Update the page content
        return document

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Processes a list of documents by applying the ELT pipeline.
        """
        processed_documents = []
        for document in documents:
            extracted_text = self.extract(document)
            transformed_text = self.transform(extracted_text)
            processed_document = self.load(document, transformed_text)
            processed_documents.append(processed_document)
        return processed_documents