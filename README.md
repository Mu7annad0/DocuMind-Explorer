# DocuMind-Explorer
DocuMind-Explorer is an application designed to interact with your documents locally. By leveraging the power of RAG (Retrieval Augmented Generation), DocuMind-Explorer allows you to ask questions about your documents and receive accurate and relevant answers.

# Advanced RAG Techniques Used:
### 1. Query Decomposition (Multi-Query):
 Breaks down complex queries into smaller sub-queries to retrieve more precise and comprehensive information.  
### 2. Hybrid Search: 
Integrates keyword-based and semantic search to enhance document retrieval accuracy.  
### 3. Hybrid Chunking: 
Applies multiple chunking strategies to optimize document segmentation for better retrieval and generation.  
### 4. Rerancking and Filtering: 
Ranks retrieved results by relevance and applies filters to remove the unrelated context.

# Installation
### 1. Clone the Repository
```sh
git colne git@github.com:Mu7annad0/DocuMind-Explorer.git
cd DocuMind-Explorer
```

### 2. Install the dependencies
```sh
pip install -r requirements.txt
```

### 3. Get GROQ Api Key, and then
```sh
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

### 4. Get Cohere Api Key for the embedding and reranking
```sh
export COHERE_API_KEY="YOUR_COHERE_API_KEY"
```

### 5. Run the application
```sh
streamlit run app.py
```
