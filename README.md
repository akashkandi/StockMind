# StockMind

**Multi-Agent Investment Research System — research any company in 60 seconds**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agents-FF6B35?style=for-the-badge)](https://langchain-ai.github.io/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

**[Live Demo](https://stockmind-3.onrender.com)** · **[API Docs](https://stockmind-2.onrender.com/docs)**

---

## Overview

StockMind is a production multi-agent AI system that researches any company by running 4 specialized agents simultaneously — news retrieval, financial analysis, SEC filing review, and sentiment scoring — then synthesizes everything into a professional investment report with a BUY/HOLD/SELL recommendation.

Type a company name. Watch 4 agents work in parallel. Get a full research report in under 60 seconds.

---

## Architecture

```
User Input: "Apple"
        ↓
┌──────────────┬─────────────────┬─────────────┬──────────────────┐
│  News Agent  │Financials Agent │  SEC Agent  │ Sentiment Agent  │
│              │                 │             │                  │
│ Tavily web   │ yfinance API    │ SEC EDGAR   │ GPT-4o-mini      │
│ search for   │ live stock      │ 10-K/10-Q   │ financial        │
│ recent news  │ price, P/E,     │ risk factor │ sentiment        │
│ + headlines  │ revenue, margins│ extraction  │ scoring          │
└──────┬───────┴────────┬────────┴──────┬──────┴──────────┬───────┘
       │                │               │                 │
       └────────────────┴───────────────┴─────────────────┘
                                ↓
                     LangGraph Supervisor Agent
                                ↓
                    GPT-4o-mini Synthesis
                                ↓
              Professional Report + BUY/HOLD/SELL
                                ↓
                    Saved to PostgreSQL
                    Streamed via WebSocket
```

---

## Features

- **4 Parallel Agents** — News, Financials, SEC, and Sentiment agents run simultaneously using LangGraph + ThreadPoolExecutor
- **Live Agent Dashboard** — WebSocket streams each agent's status in real time as it completes
- **Professional Reports** — Executive summary, financial analysis, risk assessment, sentiment scoring, and recommendation
- **Report History** — Every report saved to PostgreSQL, retrievable anytime
- **Index/ETF Detection** — Automatically detects market indices and adapts analysis to macro factors
- **Dockerized** — Full stack runs with `docker-compose up`

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph (same framework used at JPMorgan, Uber, LinkedIn) |
| Backend API | Python 3.11 + FastAPI |
| Frontend | React 18 |
| Real-time Streaming | WebSockets |
| Database | PostgreSQL + SQLAlchemy ORM |
| LLM | GPT-4o-mini |
| News Search | Tavily API |
| Financial Data | yfinance (live stock data, no API key needed) |
| SEC Filings | SEC EDGAR API (free government database) |
| Sentiment (local) | FinBERT — BERT fine-tuned on financial text |
| Containerization | Docker + docker-compose |

---

## Notable Technical Decisions

**LangGraph for orchestration**
LangGraph was chosen over CrewAI because it provides production-grade stateful graph execution — the same framework JPMorgan and Uber use in production. The supervisor/worker pattern enables clean parallel execution and deterministic state management.

**Parallel agent execution**
All 4 agents run simultaneously using ThreadPoolExecutor with asyncio. Total research time is bounded by the slowest agent (~20-30s) rather than the sum of all agents (~60-90s sequential).

**FinBERT for sentiment**
Rather than using a general-purpose LLM for sentiment, FinBERT (BERT fine-tuned on financial news, earnings calls, and analyst reports) provides domain-specific sentiment scoring. In production deployment, the API-based fallback maintains equivalent results without the memory overhead of running PyTorch locally.

**WebSocket streaming**
Agent status updates stream in real time via WebSocket — each agent sends `running` → `complete` events as it finishes. This gives users live visibility into the research pipeline rather than a blank loading screen.

---

## Getting Started

### Prerequisites
- Python 3.11+, Node.js 18+
- [OpenAI API key](https://platform.openai.com)
- [Tavily API key](https://tavily.com) (free tier)

### Installation

```bash
git clone https://github.com/akashkandi/StockMind.git
cd StockMind/backend

python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Create backend/.env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
DATABASE_URL=postgresql://stockmind:stockmind123@localhost:5432/stockmind
```

```bash
# Start PostgreSQL
docker run --name stockmind-db -e POSTGRES_USER=stockmind \
  -e POSTGRES_PASSWORD=stockmind123 -e POSTGRES_DB=stockmind \
  -p 5432:5432 -d postgres:15

# Start backend
uvicorn main:app --reload

# Start frontend (new terminal)
cd ../frontend && npm install && npm start
```

### Docker

```bash
docker-compose up
# Frontend → http://localhost:3000
# Backend  → http://localhost:8000
# API Docs → http://localhost:8000/docs
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/research` | Start research on a company |
| `WS` | `/ws/{research_id}` | Stream live agent progress |
| `GET` | `/reports` | List all past reports |
| `GET` | `/reports/{id}` | Get specific report |
| `GET` | `/reports/company/{name}` | Search reports by company |
| `GET` | `/docs` | Swagger UI |

---

## Project Structure

```
StockMind/
├── backend/
│   ├── main.py            # FastAPI + WebSocket endpoints
│   ├── graph.py           # LangGraph orchestration
│   ├── news_agent.py      # News research agent
│   ├── financials_agent.py # Financial data agent
│   ├── sec_agent.py       # SEC filing agent
│   ├── sentiment_agent.py # Sentiment analysis agent
│   ├── database.py        # PostgreSQL models + connection
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/App.js         # React UI + WebSocket client
│   └── Dockerfile
└── docker-compose.yml
```

---

Built by **Akash Kandi** — MS Computer Science

[GitHub](https://github.com/akashkandi) · [LinkedIn](https://linkedin.com/in/akashkandi)
