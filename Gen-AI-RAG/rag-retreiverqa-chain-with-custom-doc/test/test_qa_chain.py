import unittest
from src.utils.qa_chain import QaChainBuilder  # Replace your_module
# ... (Import necessary classes for HuggingFacePipeline and Chroma.as_retriever) ...

class TestQaChainBuilder(unittest.TestCase):

    def test_build(self):
        # ... (Create mock HuggingFacePipeline and Chroma.as_retriever objects) ...
        builder = QaChainBuilder()
        qa_chain = builder.with_llm(mock_llm).with_retriever(mock_retriever).build()
        self.assertIsInstance(qa_chain, RetrievalQA)

    def test_build_missing_llm(self):
        builder = QaChainBuilder()
        with self.assertRaises(ValueError):
            builder.with_retriever(mock_retriever).build()  # Missing LLM

    def test_build_missing_retriever(self):
        builder = QaChainBuilder()
        with self.assertRaises(ValueError):
            builder.with_llm(mock_llm).build()  # Missing retriever