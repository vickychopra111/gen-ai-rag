# gen-ai-rag

Testable--> Deployable--> Scalable 

This is the project structure 
rag_project/
├── data/ # Data files (PDFs, etc.)
├── model/  # Model checkpoints  
│   ├── embedding_model.py
│   ├── llm_model.py
├── utils/ # Utility functions
│   ├── document_loader.py # Factory Pattern 
│   ├── document_processor.py
│   ├── qa_chain.py  # Chain builder pattern
│   └── elt.py  #   module for ELT
├── tests/  #  test files
│   ├── test_data_loading.py
│   ├── test_embedding.py
│   ├── test_qa_chain.py
│   └── test_elt.py  #  ELT test file
├── main.py
├── Dockerfile
└── requirements.txt
