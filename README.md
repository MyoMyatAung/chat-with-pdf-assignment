# Generative AI Question-Answering System

This project implements a backend system for intelligent question-answering over a corpus of academic PDF papers, with web search fallback, as per the assignment requirements. It uses a multi-agent architecture with LangGraph, Retrieval-Augmented Generation (RAG) with FAISS and Google Gemini embeddings, and a FastAPI server.

## How It Works

### Architecture Overview
The system is a modular, Python-based backend designed to answer questions grounded in provided PDF documents, handle follow-up questions with session-based memory, and perform web searches for out-of-scope queries. It uses:

- **FastAPI**: Provides RESTful API endpoints (`/ask` for questions, `/clear_memory` for session reset).
- **LangChain**: Manages document ingestion, RAG, and session memory.
- **LangGraph**: Orchestrates a multi-agent workflow for routing, clarification, retrieval, and web search.
- **FAISS**: Stores document embeddings for efficient retrieval.
- **Google Gemini**: LLM (`gemini-2.5-flash`) for answer generation.
- **HuggingFace**: Powers embeddings (`sentence-transformers/all-mpnet-base-v2`) for query processing
- **Tavily**: Handles web searches for queries not answerable from PDFs.

### Agent Descriptions
The system uses a LangGraph-based multi-agent architecture with the following nodes:
- **Router Agent**: Classifies queries as:
  - `pdf`: Answerable from the PDF corpus (e.g., "Which prompt template gave the highest zero-shot accuracy?").
  - `web`: Requires external info (e.g., "What did OpenAI release this month?").
  - `clarify`: Ambiguous queries needing clarification (e.g., "How many examples are enough?").
- **Clarification Agent**: Generates a clarification response for vague queries, using PDF context if available.
- **RAG Agent**: Retrieves relevant PDF chunks from FAISS and generates answers using Gemini.
- **Web Search Agent**: Performs a Tavily search and summarizes results with Gemini.
- **Final Answer**: All agents (except router) produce the final response, stored in session memory.

### Workflow
1. PDFs are ingested (`ingest.py`) by chunking, embedding with Gemini, and storing in FAISS.
2. A user query is sent via the `/ask` endpoint with a `session_id`.
3. The router classifies the query and directs it to the appropriate agent.
4. The selected agent processes the query, using session history for context, and returns an answer.
5. Session memory is maintained in-memory (per `session_id`) and can be cleared via `/clear_memory`.

### Trade-offs
- **In-memory FAISS**: Fast and simple but not scalable for large datasets.
- **Single-user Memory**: In-memory session store (dict) is sufficient for the assignment but not production-ready.
- **Basic Clarification**: Returns a clarification message but doesn’t loop for user input due to API focus.

## How to Run Locally Using Docker-Compose

### Prerequisites
- **Docker and Docker Compose**: Install from [docker.com](https://www.docker.com/).
- **API Keys**:
  - Google API Key (Gemini): Get from [Google Makersuite](https://makersuite.google.com/).
  - Tavily API Key: Get from [tavily.com](https://tavily.com/) (free tier available).
- **PDFs**: Place academic papers (PDFs) in the `papers/` folder. For testing, use the assignment PDF.

### Setup
1. **Clone the Repository**:
   ```bash
   git clone git@github.com:MyoMyatAung/chat-with-pdf-assignment.git
   cd generative-ai-assignment
   ```
2. **Create `.env` File**:
   Create a `.env` file in the project root with:
   ```
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```
3. **Add PDFs**:
   Copy provided PDF papers to the `papers/` folder. Ensure at least one PDF exists.
4. **Ingest PDF**:
    ```bash
   python ingest.py
   ```
5. **Build and Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```
   - This builds the Docker image, runs `ingest.py` to process PDFs, and starts the FastAPI server on `http://localhost:8000`.
   - The FAISS index is created in `faiss_index/` on first run.
6. **Test the API**:
   - Ask a PDF-based question:
     ```bash
     curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"Which prompt template gave the highest zero-shot accuracy on Spider in Zhang et al. (2024)?","session_id":"user1"}'
     ```
     Expected: `{"answer": "SimpleDDL-MD-Chat gave the highest zero-shot accuracy (65–72% EX across models)."}`
   - Ask a web-based question:
     ```bash
     curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What did OpenAI release this month?","session_id":"user1"}'
     ```
   - Clear session memory:
     ```bash
     curl -X POST http://localhost:8000/clear_memory -H "Content-Type: application/json" -d '"user1"'
     ```
7. **Stop the Server**:
   ```bash
   docker-compose down
   ```

### Running Without Docker (Optional)
1. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/macOS
   # or: env\Scripts\activate (Windows)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ingest PDFs:
   ```bash
   python ingest.py
   ```
4. Start the server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
5. Test as above.

### Troubleshooting
- **FAISS Dimension Error**: Ensure `ingest.py` and `agents.py` use the same embedding model HuggingFaceEmbeddings(`sentence-transformers/all-mpnet-base-v2`). Delete `faiss_index/` and re-run `python ingest.py`.
- **Empty Index**: Verify `papers/` contains valid PDFs and `ingest.py` runs without errors.
- **API Key Issues**: Check `.env` for valid `GOOGLE_API_KEY` and `TAVILY_API_KEY`.
- **Port Conflicts**: If port 8000 is in use, change to another (e.g., `--port 8001`).

## How to Improve in the Future
1. **Scalable Vector Store**: Replace in-memory FAISS with a persistent vector database like Pinecone or Weaviate for handling larger corpora.
2. **Interactive Clarification**: Implement a conversational loop for ambiguous queries, prompting users for clarification via the API.
3. **Evaluation System**: Add a script with golden Q&A pairs (e.g., from assignment examples) and compute metrics like BLEU or semantic similarity.
4. **Hybrid Search**: Combine keyword-based search (e.g., BM25) with vector search for better retrieval.
5. **Persistent Memory**: Use a database (e.g., Redis) for session memory to support multiple users and persistence.

## Docker and docker-compose.yml Files

### Dockerfile
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./pdfs:/app/pdfs
      - ./faiss_index:/app/faiss_index
```
