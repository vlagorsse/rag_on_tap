# RAG-on-Tap: Agent Guidelines

This document provides essential context and standards for AI agents and developers working on the RAG-on-Tap project.

## Tech Stack
- **Backend:** FastAPI, LangChain, `langchain-google-genai` (Gemini 2.5 Flash Lite).
- **Frontend:** React, Vite, Tailwind CSS.
- **Database:** PostgreSQL with PGVector.
- **Orchestration:** Docker Compose.

## Code Standards & Formatting
Consistency is mandatory. Always format code according to the following rules:

### Environment & Tooling
- **direnv:** Use `direnv` to manage environment variables and virtual environments. The `backend/.envrc` should automatically activate the virtual environment and load `.env` variables.
- **Test Setup:** Ensure `direnv allow` is run in the `backend` directory to load the environment before running tests.

### Python (Backend)
- **Formatter:** `black`
- **Import Sorting:** `isort`
- Use type hints wherever possible.

### TypeScript/JavaScript (Frontend)
- **Formatter:** `prettier`
- Use functional components and hooks for React.

## Key Architectures
- **Monorepo:** Backend and Frontend are decoupled in their respective directories.
- **API Prefix:** All backend routes are prefixed with `/api`.
- **Nginx:** Acts as a reverse proxy for both static assets and API forwarding.
