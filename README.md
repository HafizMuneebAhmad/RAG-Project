<p align="center">
  <h1 align="center">🤖 RAG Assistant</h1>
  <p align="center">
    <strong>A production-grade Retrieval-Augmented Generation pipeline with advanced hybrid search, cross-encoder reranking, and real-time streaming.</strong>
  </p>
  <p align="center">
    <a href="#-features">Features</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-getting-started">Getting Started</a> •
    <a href="#-api-reference">API Reference</a> •
    <a href="#-project-structure">Project Structure</a> •
    <a href="#-evaluation">Evaluation</a>
  </p>
</p>

---

## 📋 Overview

**RAG Assistant** is a modular, end-to-end Retrieval-Augmented Generation system built with LangChain, FastAPI, and Streamlit. It ingests PDF documents, chunks and embeds them into a ChromaDB vector store, and answers user questions using a multi-stage retrieval pipeline that combines semantic search, BM25 lexical search, Reciprocal Rank Fusion, and cross-encoder reranking — all served through a streaming REST API and an interactive chat UI.

### Why This Project?

| Traditional RAG | This Project |
|---|---|
| Single-pass vector similarity search | Hybrid retrieval (vector + BM25) with RRF fusion |
| No result refinement | Cross-encoder reranking for precision |
| Static query matching | LLM-powered query transformation |
| No conversation memory | Chat history-aware responses |
| Basic evaluation | Built-in Precision, Recall & MRR metrics |

---

## ✨ Features

- **🔍 Advanced Hybrid Retrieval** — Combines dense vector search (ChromaDB) with sparse lexical search (BM25) using Reciprocal Rank Fusion (RRF) for superior document recall.
- **🎯 Cross-Encoder Reranking** — Applies `ms-marco-MiniLM-L-6-v2` cross-encoder to rerank candidate documents for maximum relevance.
- **🧠 Query Transformation** — Rewrites user queries via LLM to be more specific and retrieval-friendly before searching.
- **💬 Conversational Memory** — Maintains chat history across turns for context-aware multi-turn conversations.
- **⚡ Real-Time Streaming** — Streams LLM responses token-by-token through the API and Streamlit UI for instant feedback.
- **📄 PDF Ingestion Pipeline** — Loads, splits, and indexes PDF documents into ChromaDB with metadata tracking.
- **📊 Evaluation Framework** — Built-in retrieval evaluation with Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
- **🏗️ Modular Architecture** — Clean separation of concerns with pluggable components for loaders, splitters, retrievers, rerankers, and LLMs.

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Chat UI                        │
│                   (streamlit_app.py — Port 8501)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP (REST)
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI Backend (Port 8000)                  │
│              /ask (JSON)  •  /ask/stream (SSE)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                       RAG Service Layer                          │
│            Orchestrates ingestion & query answering              │
└──────┬───────────────────────────────────────┬──────────────────┘
       │                                       │
┌──────▼──────────┐                   ┌────────▼─────────────────┐
│  Ingestion       │                   │  Advanced RAG Chain       │
│  Pipeline        │                   │                          │
│  ┌────────────┐  │                   │  ┌────────────────────┐  │
│  │ PDF Loader │  │                   │  │ Query Transformer  │  │
│  └─────┬──────┘  │                   │  └────────┬───────────┘  │
│  ┌─────▼──────┐  │                   │  ┌────────▼───────────┐  │
│  │ Text       │  │                   │  │ Advanced Retriever │  │
│  │ Splitter   │  │                   │  │ ┌───────┐ ┌──────┐ │  │
│  └─────┬──────┘  │                   │  │ │Vector │ │ BM25 │ │  │
│  ┌─────▼──────┐  │                   │  │ └───┬───┘ └──┬───┘ │  │
│  │ ChromaDB   │  │                   │  │     └──┬─────┘     │  │
│  │ Indexing   │  │                   │  │   ┌────▼─────┐     │  │
│  └────────────┘  │                   │  │   │ RRF      │     │  │
└──────────────────┘                   │  │   │ Fusion   │     │  │
                                       │  │   └────┬─────┘     │  │
                                       │  │   ┌────▼─────┐     │  │
                                       │  │   │Cross-Enc.│     │  │
                                       │  │   │Reranker  │     │  │
                                       │  │   └──────────┘     │  │
                                       │  └────────────────────┘  │
                                       │  ┌────────────────────┐  │
                                       │  │ LLM (Groq)        │  │
                                       │  │ + Chat History     │  │
                                       │  └────────────────────┘  │
                                       └──────────────────────────┘
```

### Retrieval Pipeline Flow

```
User Question
     │
     ▼
Query Transformer (LLM rewrites query)
     │
     ▼
┌────┴────┐
│         │
▼         ▼
Vector    BM25
Search    Search
│         │
└────┬────┘
     ▼
RRF Fusion (merges & scores)
     │
     ▼
Cross-Encoder Reranker (top-K)
     │
     ▼
Context + Prompt → LLM → Answer (streamed)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Groq API (configurable model, default: `openai/gpt-oss-120b`) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` via `sentence-transformers` |
| **Vector Store** | ChromaDB (persistent, local) |
| **BM25 Search** | `rank_bm25` (BM25Okapi) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` |
| **Framework** | LangChain (chains, prompts, output parsers) |
| **API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit (dark-themed chat interface) |
| **Config** | python-dotenv + Pydantic |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Groq API Key** — Get one at [console.groq.com](https://console.groq.com)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/RAG.git
   cd RAG
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install fastapi uvicorn streamlit langchain langchain-groq langchain-google-genai \
               langchain-huggingface langchain-chroma langchain-community langchain-text-splitters \
               sentence-transformers rank-bm25 chromadb pypdf python-dotenv pydantic
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   LLM_MODEL=openai/gpt-oss-120b
   CHROMA_PATH=./chroma_db
   COLLECTION_NAME=rag_documents
   ```

5. **Add your PDF documents**

   Place your PDF files in the `data/` directory. Update the file path in `services/rag_service.py` if needed:

   ```python
   ingestion = DocumentIngestion(r"path/to/your/document.pdf")
   ```

### Running the Application

**Start the FastAPI backend:**

```bash
python main.py
```

The API will be available at `http://127.0.0.1:8000`.

**Start the Streamlit frontend** (in a separate terminal):

```bash
streamlit run streamlit_app.py
```

The chat UI will open at `http://localhost:8501`.

---

## 📡 API Reference

### `GET /`

Health check endpoint.

**Response:**

```json
{
  "message": "RAG api is running"
}
```

---

### `POST /ask`

Submit a question and receive a complete JSON response.

**Request Body:**

```json
{
  "question": "What are the 7 layers of the OSI model?"
}
```

**Response:**

```json
{
  "question": "What are the 7 layers of the OSI model?",
  "answer": "The 7 layers of the OSI model are..."
}
```

---

### `POST /ask/stream`

Submit a question and receive a streaming `text/plain` response (token-by-token).

**Request Body:**

```json
{
  "question": "Explain the transport layer."
}
```

**Response:** Chunked `text/plain` stream.

---

## 📂 Project Structure

```
RAG/
│
├── main.py                      # Application entry point (Uvicorn server)
├── streamlit_app.py             # Streamlit chat UI with dark theme
├── .env                         # Environment variables (API keys, config)
│
├── api/                         # FastAPI REST API layer
│   ├── app.py                   # Route definitions (/ask, /ask/stream)
│   ├── schemas.py               # Pydantic request/response models
│   └── dependencies.py          # Dependency injection (RAG service)
│
├── services/                    # Business logic / orchestration
│   └── rag_service.py           # RAGService — initializes pipeline & handles queries
│
├── chains/                      # LangChain chain implementations
│   ├── rag_chain.py             # Basic RAG chain (vector search only)
│   ├── advanced_rag_chain.py    # Advanced RAG chain (hybrid + reranking + streaming)
│   ├── query_transformer.py     # LLM-based query rewriting
│   ├── query_retriever.py       # Combined transform + retrieve step
│   └── chat_history.py          # In-memory conversation history manager
│
├── retriever/                   # Document retrieval strategies
│   ├── base_retriever.py        # Abstract base class (ABC)
│   ├── retriever.py             # Basic vector retriever (MMR search)
│   ├── bm25_retriever.py        # BM25 lexical retriever (BM25Okapi)
│   ├── hybrid_retriever.py      # Hybrid retriever (vector + BM25 + RRF)
│   ├── advanced_retriever.py    # Full pipeline (hybrid + reranking)
│   └── rrf.py                   # Reciprocal Rank Fusion implementation
│
├── reranker/                    # Cross-encoder reranking
│   └── reranker.py              # DocumentReranker (ms-marco-MiniLM)
│
├── embeddings/                  # Embedding models
│   └── embeddings.py            # HuggingFace all-MiniLM-L6-v2 wrapper
│
├── vectorstore/                 # Vector database
│   └── chroma.py                # ChromaDB wrapper (add, search, filter)
│
├── loaders/                     # Document loaders
│   └── pdf_loader.py            # PDF loading via PyPDFLoader
│
├── splitters/                   # Text chunking
│   └── text_splitters.py        # RecursiveCharacterTextSplitter (500 chars, 50 overlap)
│
├── ingestion/                   # Document ingestion pipeline
│   └── ingestion.py             # Load → Split → Index (end-to-end)
│
├── llm/                         # LLM wrapper
│   └── model.py                 # AIModel (Groq ChatGroq)
│
├── config/                      # Configuration management
│   └── settings.py              # Settings class (env vars, validation)
│
├── utils/                       # Shared utilities
│   ├── logger.py                # Structured logging setup
│   └── exception.py             # Custom exceptions (RAGError hierarchy)
│
├── evaluation/                  # Retrieval quality evaluation
│   ├── retrieval_evaluator.py   # Precision@K, Recall@K, MRR metrics
│   ├── evaluation_runner.py     # Automated evaluation runner
│   ├── comparison.py            # Retriever comparison framework
│   └── test_dataset.py          # Test question/answer dataset
│
├── data/                        # PDF documents for ingestion
│   ├── OSIModel.pdf
│   ├── OSI Reference model.pdf
│   └── LAB REPORT.pdf
│
└── chroma_db/                   # Persistent ChromaDB storage (auto-generated)
```

---

## 📊 Evaluation

The project includes a built-in evaluation framework to measure retrieval quality across different retriever configurations.

### Metrics

| Metric | Description |
|---|---|
| **Precision@K** | Fraction of retrieved documents that are relevant |
| **Recall@K** | Fraction of relevant documents that were retrieved |
| **MRR** | Mean Reciprocal Rank — how early the first relevant result appears |

### Running Evaluation

```python
from evaluation.evaluation_runner import EvaluationRunner
from retriever.advanced_retriever import AdvancedRetriever

runner = EvaluationRunner(retriever=your_retriever)
runner.run()
```

### Comparing Retrievers

```python
from evaluation.comparison import RetrievelComparison
from evaluation.retrieval_evaluator import RetrievalEvaluator

comparison = RetrievelComparison(evaluator=RetrievalEvaluator())
results = comparison.evaluate(retriever=your_retriever, tests=test_data)
print(results)  # {'precision': 0.67, 'recall': 0.80, 'mrr': 0.83}
```

---

## ⚙️ Configuration

All configuration is managed through environment variables in the `.env` file:

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key | *required* |
| `LLM_MODEL` | LLM model identifier | `openai/gpt-oss-120b` |
| `CHROMA_PATH` | ChromaDB persistence directory | `./chroma_db` |
| `COLLECTION_NAME` | ChromaDB collection name | `rag_documents` |

---

## 🧩 How It Works

1. **Ingestion** — PDF documents are loaded via `PyPDFLoader`, split into 500-character chunks (50-char overlap) using `RecursiveCharacterTextSplitter`, and indexed into ChromaDB with `all-MiniLM-L6-v2` embeddings.

2. **Query Transformation** — The user's raw question is rewritten by the LLM into a more specific, retrieval-optimized query.

3. **Hybrid Retrieval** — The transformed query is searched against both the vector store (semantic similarity) and a BM25 index (lexical matching) to maximize recall.

4. **RRF Fusion** — Results from both retrievers are merged using Reciprocal Rank Fusion (constant = 60), producing a unified ranked list.

5. **Cross-Encoder Reranking** — The top candidates are re-scored by a `ms-marco-MiniLM-L-6-v2` cross-encoder, and only the top-K most relevant documents are kept.

6. **Answer Generation** — The reranked documents are formatted as context, combined with the question and chat history, and passed to the Groq LLM which generates a response (optionally streamed token-by-token).

---

## 🗺️ Roadmap

- [ ] Multi-format document support (DOCX, TXT, Markdown, HTML)
- [ ] File upload through the Streamlit UI
- [ ] Persistent chat history (database-backed)
- [ ] Configurable chunking strategies
- [ ] Docker deployment with `docker-compose`
- [ ] Automatic evaluation on document ingestion
- [ ] Support for additional LLM providers (OpenAI, Anthropic, Ollama)

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

---

<p align="center">
  Built with ❤️ using LangChain, FastAPI, ChromaDB & Groq
</p>
