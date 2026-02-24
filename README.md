# Beer RAG 🍻

A high-performance Retrieval-Augmented Generation (RAG) system for beer recipes, leveraging structured enrichment, contextual chunking, and multi-stage retrieval.

This project serves as a showcase for advanced RAG techniques applied to a specialized domain (homebrewing), demonstrating how to transform messy, technical raw data into a highly searchable and informative knowledge base.

## 🚀 The Approach: HyPE Enrichment

The core of this system is the **HyPE (Hypothetical Questions for Enrichment)** approach, inspired by recent research in RAG optimization ([Hypothetical Questions for Enrichment](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5139335)).

### 1. Structured Data Enrichment

Raw beer recipes are often difficult to query. They contain technical metrics (ABV, IBU, SRM), community slang, and messy user notes. We use **Gemini 2.5 Flash Lite** with structured output to "interrogate" each recipe using a standardized set of 15+ sommelier-style questions covering:

- **Appearance**: Color, clarity, and head retention.
- **Aroma**: Malt profile, hop characteristics, and yeast esters.
- **Flavor & Mouthfeel**: Balance, bitterness quality, body, and finish.
- **Overall Impression**: Style accuracy and drinkability.

By integrating user comments and translating technical metrics into descriptive language, the RAG can answer nuanced queries like _"Find me a stout that feels creamy and has chocolate notes but isn't too boozy."_

### 2. Contextual Chunking

Standard character-based chunking often loses the relationship between a fragment of text and its source. Our `ChunkingService` implements **Contextual Chunk Headers**:

- **Strict Splitting**: Documents are split by logical sections (e.g., "Aroma", "Mouthfeel").
- **Header Injection**: Each chunk is prepended with its source metadata: `Recipe: [Name] | Style: [Style] | Section: [Category] | Text: ...`
- **Impact**: This ensures that even small text fragments carry enough context for the embedding model and the reranker to understand exactly what they refer to.

### 3. Two-Stage Retrieval

To balance speed and precision, we use a two-stage pipeline:

1.  **Vector Search**: Candidate retrieval using **Qwen3-Embedding-0.6B** (optimized for low latency and high recall).
2.  **Reranking**: Precision refinement using **Qwen3-Reranker-0.6B**. This cross-encoder step ensures that the most relevant chunks are promoted, significantly reducing hallucinations.

---

## 🛠️ Models & Technical Stack

- **Enrichment LLM**: `gemini-2.5-flash-lite` (via Vertex AI)
- **Embedding Model**: `Qwen/Qwen3-Embedding-0.6B`
- **Reranker**: `Qwen/Qwen3-Reranker-0.6B`
- **Vector Store**: PGVector (PostgreSQL)
- **Orchestration**: LangChain, Pydantic, BeautifulSoup4

---

## 🏗️ Architecture

The project is organized as a decoupled Full-Stack application:

- **`backend/`**: FastAPI server providing RAG capabilities.
  - `api/`: REST endpoints for chat and recipe retrieval.
  - `services/`: Core RAG logic, embedding, and reranking.
  - `utilities/`: CLI tools for data preparation.
- **`frontend/`**: React + Vite + Tailwind CSS dashboard.
  - ChatGPT-like interface optimized for beer expertise.
  - Real-time streaming responses.

---

## 🚦 Quick Start

### 1. Environment Setup

Copy the sample environment file and fill in your API keys:

```bash
cp .env.sample .env
```

The `.env` file contains configuration for both local development and Docker:

- **LLM Keys**: Required for the RAG enrichment and chat.
- **Database**: Credentials for PostgreSQL/PGVector.

### 2. Local Development

#### Backend

Ensure you have a local Postgres instance running or use Docker for the database.

```bash
cd backend
uv sync
source .venv/bin/activate
uvicorn main:app --reload
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. Docker Compose (Full Stack)

```bash
docker compose up
```

The application will be available at:

- **Frontend**: `http://localhost:3001`
- **Backend API**: `http://localhost:8002`

---

## 🛠️ Data Preparation Pipeline

Before chatting, ensure your database is populated:

1. **Enrich**: `hype-enrichment -o enriched_recipes.csv`
2. **Populate**: `populate-db enriched_recipes.csv`

## 📈 Performance Insights

Transitioning from small local models (Qwen2.5-1.5B) to **Gemini 2.5 Flash Lite** for enrichment drastically improved result quality. The structured output eliminated common parsing errors, while the inclusion of contextual headers in chunks solved early issues where the reranker was performing poorly due to lost context.
