import unittest
from src.utils.document_loader import DocumentLoaderFactory  # Replace your_module

class TestDocumentLoaderFactory(unittest.TestCase):

    def test_create_loader_web(self):
        factory = DocumentLoaderFactory()
        loader = factory.create_loader("https://example.com")
        self.assertIsInstance(loader, WebBaseLoader)

    def test_create_loader_pdf(self):
        factory = DocumentLoaderFactory()
        loader = factory.create_loader("path/to/document.pdf")
        self.assertIsInstance(loader, PyPDFLoader)

    def test_create_loader_invalid(self):
        factory = DocumentLoaderFactory()
        with self.assertRaises(ValueError):
            factory.create_loader("invalid_file.txt")