from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from loguru import logger

class QaChainBuilder:
    """Builder class for creating the QA chain."""

    def __init__(self):
        self.llm = None
        self.retriever = None

    def with_llm(self, llm: HuggingFacePipeline):
        """Sets the language model."""
        self.llm = llm
        return self

    def with_retriever(self, retriever: Chroma.as_retriever):
        """Sets the retriever."""
        self.retriever = retriever
        return self

    def build(self) -> RetrievalQA:
        """Builds and returns the QA chain."""
        if self.llm is None or self.retriever is None:
            raise ValueError("LLM and retriever must be set before building the QA chain.")
        try:
            return RetrievalQA.from_chain_type(
                llm=self.llm, chain_type='stuff', retriever=self.retriever
            )
        except Exception as e:
            logger.exception(f"Error creating QA chain: {e}")
            raise