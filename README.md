# Arize Corp Travel Assistant

A policy-aware travel agent built with LangGraph. It helps employees find flights, hotels and things to do and only recommends options that comply with company travel policy. The agent is traced with Arize Phoenix and evaluated for user frustration.

## Setup

Requires Python 3.11+, Poetry and an OpenAI API key.

```bash
poetry install
cp .env.example .env          # add your OPENAI_API_KEY
pipx install --python python3.12 arize-phoenix
```

## Run

```bash
phoenix serve                                  # Phoenix UI: http://localhost:6006
poetry run uvicorn api:app --reload            # Chat UI: http://localhost:8000
```

Scripts (with both running):

```bash
poetry run python scripts/run_queries.py       # send test queries
poetry run python scripts/export_spans.py      # export spans to exports/
poetry run python scripts/run_evals.py         # run evals and create the frustrated-interactions dataset
```

## Files

| File | Description |
|---|---|
| `agent.py` | LangGraph agent: LLM ⇄ tools loop, system prompt, conversation memory |
| `api.py` | FastAPI app: `/chat` endpoint, chat UI, Phoenix tracing |
| `flight_tool.py` | `search_flights` tool: structured flight options (mock data) |
| `policy_tool.py` | `check_flight_policy` and `check_hotel_policy` tools |
| `policy.json` | Company travel policy rules |
| `static/index.html` | Chat UI |
| `scripts/run_queries.py` | Sends test conversations to the agent |
| `scripts/export_spans.py` | Exports all spans from Phoenix |
| `scripts/run_evals.py` | User frustration and policy evals; logs results to Phoenix and creates a dataset |
| `exports/` | Exported spans and eval results |
