# DocuMind-Explorer
DocuMind-Explorer is an application designed to interact with your documents locally. By leveraging the power of RAG (Retrieval Augmented Generation) and the Gemma 2 language model, DocuMind-Explorer allows you to ask questions about your documents and receive accurate and relevant answers.

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

### 3. Install Ollama:

* https://ollama.com/


### 4. Download the model
```sh
ollama pull gemma2
```

### 5. Run the model
```sh
ollama serve
```

### 6. Run the application
```sh
streamlit run app.py
```
