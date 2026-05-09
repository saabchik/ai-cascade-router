# AI Cascade Router

[![EN](https://img.shields.io/badge/lang-en-red.svg)](README.en.md) [![RU](https://img.shields.io/badge/lang-ru-blue.svg)](README.md)

A smart LLM proxy that **saves up to 70% tokens** on simple queries by routing them to local models, while complex ones go to the cloud.

> Actual savings depend on your use case: chatbot / translation → 60-80%, mixed workload → 35-55%, analysis / refactoring → 5-15%.

---

## Core Idea

### Problem

Cloud LLMs (GPT, Claude, Gemini) cost $0.15 to $15 per million tokens. Under heavy load, bills add up fast.

### Solution

Most LLM queries (60-80%) are simple: short questions, basic code, greetings, translations. These can be handled by a **local model** (Qwen, Llama, Mistral via LM Studio) — fast and free. Complex tasks (data analysis, refactoring, strategy) go to a **cloud model** (any model via OpenRouter) — high quality but paid.

**AI Cascade Router** is a proxy server that works in real-time:

1. **Classifies** the query by domain (code, chat, analysis...) and complexity
2. **Checks cache** — similar queries may already have an answer ready
3. **Makes a decision** — LOCAL (free), CLOUD (paid), or HYBRID (local extraction + cloud)
4. For multi-step queries — **decomposes** into subtasks (CASCADE), each routed separately
5. **Self-assessment** — the model rates its own confidence; if low — fallback to cloud
6. **Saves session** — with `session_id`, dialog history is passed as context to the model
7. **Tracks savings** — every request logs tokens saved

```
                        ┌─────────────────┐
                        │  Client request  │
                        └────────┬────────┘
                                 │
                        ┌────────v────────┐
                        │  Semantic cache │
                        │     (hit?)      │
                        └───┬────────┬───┘
                       hit  │        │ miss
                      ┌─────v──┐     │
                      │ cache  │     │
                      │response│     │
                      └────┬───┘     │
                           │        ┌─v──────────────┐
                           │        │ Classification │
                           │        │  (ML + rules)   │
                           │        └──┬──┬───┬──┬───┘
                           │       local│  │hybrid│cascade
                           │     ┌──────v──┐ │  ┌──v───────┐
                           │     │  LOCAL  │ │  │ CASCADE  │
                           │     │  model  │ │  │decompose │
                           │     │on LMStr │ │  │→ N tasks │
                           │     └──┬──────┘ │  └──────────┘
                           │       ┌───v┐    │
                           │       │conf< →    │
                           │       │thresh│   │
                           │       └─┬─┬─┘    │
                           │    ok   │ │ low  │
                           │  ┌──────v │  ┌──v──────────┐
                           │  │return  │  │  CLOUD      │
                           │  │response│  │ (OpenRouter)│
                           │  └────────┘  └─────────────┘
                           │
                        ┌──v──────────────┐
                        │  Metrics:       │
                        │  tokens_saved,  │
                        │  cost, time     │
                        └─────────────────┘
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **ML Classifier** | Sentence-Transformers (all-MiniLM-L6-v2) + reference queries, auto-fallback to rules |
| **Session Support** | Multi-turn dialogs — history saved by `session_id`, auto-context trimming |
| **Semantic Cache** | Embedding-based cache — finds similar queries even with different wording |
| **Self-Assessment** | The model rates its own confidence in its response |
| **Domain Thresholds** | Different thresholds for code, analysis, chat — code stays local more often, analysis goes to cloud |
| **Cascade Decomposition** | Complex multi-step queries broken into subtasks |
| **404 Fallback** | If a cloud model is unavailable, automatically tries the next one |
| **Universal Language** | Works with any language — response in the same language as the query |
| **Smart Token Savings** | Mathematical savings tracking: `(1 - cloud_used/baseline) × 100%` |
| **OpenAI-Compatible API** | `/v1/chat/completions` endpoint — plug into VS Code, Cursor, Continue.dev, Cline with zero plugins |

---

## How It Works (Detailed)

### 1. Query Classification

The system determines:

**Domain** (task type):
- `code` — code generation, debugging
- `analysis` — data analysis, comparison
- `chat` — simple chat
- `question` — questions
- `creative` — creative writing
- `technical` — technical tasks

**Complexity:**
- Simple (< 200 chars, no keywords)
- Medium (complex keywords present)
- Complex (multi-step)

**Multi-step?** (needs decomposition):
- Queries with "and", "then", "also" → decomposed into subtasks

### 2. ML Classifier (v0.3)

Uses ML instead of simple rules:
- Model: `all-MiniLM-L6-v2` (80MB, ~20ms per query)
- Reference queries for each complexity class
- Cosine similarity of embeddings → complexity classification
- Auto-fallback to rule-based when ML is unavailable

### 3. Route Decision

Each domain has a confidence threshold:

```yaml
domain_thresholds:
  code: 0.65        # Code — more local (lower threshold)
  analysis: 0.80    # Analysis — more cloud (higher threshold)
  chat: 0.60        # Chat — almost always local
  question: 0.55    # Questions — minimum threshold
  creative: 0.70    # Creative — medium
  technical: 0.75   # Technical — above medium
```

Logic:
- If model confidence ≥ threshold → **LOCAL**
- If confidence < threshold → **CLOUD**
- For complex keywords → **HYBRID**

### 4. Local Mode

```
Query → LM Studio (your model) → Response + Self-Assessment
```

**Self-Assessment** — the model rates its own response:
```json
{
  "confidence": 0.85,
  "flags": ["unknown_term", "insufficient_context"]
}
```

If critical flags are present → increased chance of cloud delegation.

### 5. Hybrid Mode

```
Query → Local model (context extraction) → Cloud (final answer)
```

The local model extracts key information → the cloud model uses this context for a more accurate response.

### 6. Cascade Mode

For multi-step queries:
```
"Analyze data and create a report with charts"
          ↓
Decomposition:
  1. "Analyze data" → LOCAL
  2. "Create report" → CLOUD
  3. "Add charts" → HYBRID
```

Each subtask is routed independently.

### 7. Semantic Cache

1. Exact hash of the query is checked
2. If no match → semantic search (cosine similarity > 0.8)
3. On cache hit → returns saved response without calling any model

Embeddings: `all-MiniLM-L6-v2` (80MB, ~20ms per query).

### 8. Language Detection

The system automatically detects the query language and adds the appropriate system prompt:
- Any non-ASCII language (Russian, Chinese, Arabic...) → "Please respond in the same language as the user's question."
- Response always in the query language

---

## Savings Calculation

**Mathematical model:**
```
baseline_tokens = prompt_tokens + estimated_output_tokens (rolling average)
actual_cloud_tokens = actual cloud tokens used
tokens_saved = max(0, baseline_tokens - actual_cloud_tokens)
tokens_saved_percent = max(0, (1 - actual_cloud_tokens / baseline_tokens) × 100)
```

| Mode | Calculation |
|------|------------|
| **local** | 100% savings (all local) |
| **hybrid** | savings = baseline - cloud_tokens |
| **cascade** | Sum of savings from all subtasks |
| **cloud** | 0% (paid for cloud) |

Each request updates the cumulative metric.

---

## Dashboard & Metrics

### GET /metrics

```bash
curl http://localhost:8000/metrics
```

**Response:**
```json
{
  "total_requests": 150,
  "local_requests": 95,
  "cloud_requests": 30,
  "hybrid_requests": 25,
  "tokens_saved": 4200,
  "total_cost_usd": 0.12,
  "avg_response_time_ms": 890.0,
  "delegation_rate": 0.37,
  "roi_percent": 320.0
}
```

### UI Dashboard (planned)

Shows not just percentages but a bundle of metrics:

🦜 **Local model handled 73% of requests**  
💰 **Saved 1.2M tokens (~$4.80)**  
⚡ **Average latency reduced by 40%**  
📊 **ROI: 320%** (savings vs maintenance cost)

---

## Comparison

| Project | Savings | Features |
|---------|---------|----------|
| **RouteLLM** | up to 85% | ML classifier (DistilBERT), trained on Arena data, stateless |
| **LiteLLM** | — | AI Gateway for 100+ providers, no built-in routing, stateless |
| **This project** | up to 70% | LM Studio + cloud model + ML classifier + semantic cache + sessions + multi-language |

---

## Installation

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd ai-cascade-router
pip install -r requirements.txt
```

> **Windows:** if `pip install` fails, try `py -m pip install -r requirements.txt`

### 2. Set up LM Studio (required for local mode and savings)

1. Download [LM Studio](https://lmstudio.ai/)
2. Download any model (Qwen, Mistral, Gemma recommended)
3. Launch LM Studio and load the model
4. Make sure the API is available at http://localhost:1234/v1
5. In `config.yaml`, keep `model_name: "auto"` — the router will use whatever is loaded

### 3. Choose a cloud model (optional)

Default is `openai/gpt-4o-mini` ($0.15/1M tokens) — cheap and fast. To change it, edit `cloud_model.model_name` in `config.yaml`.

**Finding the model name:**
1. Go to https://openrouter.ai/models
2. Pick a model (Claude Sonnet, Gemini Pro, DeepSeek, etc.)
3. Copy its **slug** from the browser address bar, e.g.:
   - `https://openrouter.ai/anthropic/claude-sonnet-4` → `anthropic/claude-sonnet-4`
   - `https://openrouter.ai/google/gemini-2.5-pro` → `google/gemini-2.5-pro`
4. Paste it into `config.yaml`:
   ```yaml
   cloud_model:
     model_name: "anthropic/claude-sonnet-4"
   ```

### 4. Set up .env

```bash
cp .env.example .env
```

Edit `.env` and add:
```
OPENROUTER_API_KEY=your_openrouter_api_key
```

### 5. Run

```bash
python main.py
```

Server starts at http://localhost:8000

### 5. CLI client (interactive chat)

```bash
python cli.py
```

Just type your queries — no quotes, no curl, no JSON. Session context is preserved automatically.

```
============================================================
  AI Cascade Router CLI
  Type your query and press Enter.
  /exit  — quit
  /new   — reset session (start fresh)
============================================================
write a Python factorial function
...
  [local->cloud, saved 0 tok, 2300ms]

add error handling
...
  [local->cloud, saved 0 tok, 1450ms]
```

### 6. Docker (optional)

```bash
docker build -t ai-cascade-router .
docker run -p 8000:8000 --env-file .env ai-cascade-router
```

---

## Configuration (config.yaml)

### Local model

```yaml
local_model:
  base_url: "http://localhost:1234/v1"
  model_name: "auto"  # "auto" = uses whatever is loaded in LM Studio
  timeout_seconds: 120
```

### Cloud model (OpenRouter)

Change `model_name` to any slug from https://openrouter.ai/models

```yaml
cloud_model:
  provider: "openrouter"
  model_name: "openai/gpt-4o-mini"  # openai/gpt-4o-mini, anthropic/claude-sonnet-4, google/gemini-2.5-pro, deepseek/deepseek-chat...
  base_url: "https://openrouter.ai/api/v1"
  # api_key loaded from OPENROUTER_API_KEY env var
```

> **How to find the slug:** go to OpenRouter → pick a model → copy the identifier from the URL.  
> Example: URL `https://openrouter.ai/anthropic/claude-sonnet-4` → slug `anthropic/claude-sonnet-4`.

### Semantic cache

```yaml
cache:
  max_size_mb: 100
  ttl_responses: 3600       # 1 hour for responses
  ttl_computations: 86400   # 24 hours for computations
```

### Metrics

```yaml
metrics:
  maintenance_cost_per_hour: 0.01
  cloud_cost_per_1k_tokens: 0.001
```

### Sessions

```yaml
session:
  enabled: true
  ttl_minutes: 30       # Session lifetime
  max_context_tokens: 8192  # 8K — optimal for code. For analysis lower to 4K, for complex projects raise to 16K
```

---

## API Endpoints

### POST /route

Main routing endpoint.

```bash
# Linux/macOS
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, how are you?", "task_type": "chat"}'

# Windows PowerShell
curl.exe -X POST http://localhost:8000/route -H "Content-Type: application/json" -d '{"query": "Hello, how are you?", "task_type": "chat"}'

# Windows PowerShell (alternative)
Invoke-RestMethod -Uri "http://localhost:8000/route" -Method Post -ContentType "application/json" -Body '{"query": "Hello, how are you?", "task_type": "chat"}'
```

**Parameters:**
- `query` (required) — request text
- `task_type` (opt.) — `chat`, `code`, `simple`, `question`, `analysis`, `creative`, `technical`
- `session_id` (opt.) — session identifier for multi-turn dialog
- `context` (opt.) — additional context (dict)

**Response:**
```json
{
  "response": "Hello! I'm doing well, thank you!",
  "source": "local",
  "confidence": 0.92,
  "tokens_saved": 38,
  "tokens_saved_percent": 100.0,
  "response_time_ms": 1245.6
}
```

### GET /metrics

Session metrics.

```bash
curl http://localhost:8000/metrics
```

### GET /health

Health check.

### POST /v1/chat/completions (OpenAI-compatible)

For integration with VS Code, Cursor, Continue.dev, Cline and any OpenAI-compatible tools.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "router", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Setup in tools:**

- **Continue.dev:** `config.json` → `"models": [{"title": "Cascade Router", "provider": "openai", "apiBase": "http://localhost:8000/v1"}]`
- **Cline:** Settings → API Provider → OpenAI Compatible → API Base URL: `http://localhost:8000/v1`
- **Cursor:** Settings → Models → Add custom model → `http://localhost:8000/v1`

**Response:**
```json
{
  "id": "chatcmpl-1746789012",
  "object": "chat.completion",
  "model": "router",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
  "cascade_meta": {"source": "local", "response_time_ms": 1234.56}
}
```

The `cascade_meta` field contains routing info: `source` (where the answer came from), `response_time_ms`.

---

## Project Structure

```
ai-cascade-router/
├── main.py                 # FastAPI application (entry point)
├── config.yaml             # Configuration
├── README.md               # This file
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker configuration
├── cli.py                  # Interactive REPL client (chat without curl)
├── .gitignore             # Git ignore rules
├── .env.example           # .env template
├── LICENSE                # MIT license
│
├── session/                # Sessions (multi-turn dialogs)
│   └── manager.py          # SessionManager — history, TTL, trimming
│
├── router/                 # Routing core
│   ├── engine.py           # RouterEngine — routing logic
│   ├── semantic.py         # SemanticClassifier — ML + rule-based classification
│   ├── decomposer.py       # TaskDecomposition — multi-step decomposition
│   └── protocol.py        # RouteDecision, RouteResponse
│
├── models/                 # Local models
│   └── local_engine.py     # LocalEngine — LM Studio wrapper + self-assessment + retry
│
├── cloud/                  # Cloud providers
│   └── client.py           # CloudClient — OpenRouter + 404 fallback + language detection
│
├── cache/                  # Caching
│   └── manager.py          # SemanticCache with embeddings
│
├── metrics/                # Metrics
│   └── logger.py           # Token savings calculation
│
├── utils/                  # Utilities
│   ├── model_cache.py      # Shared embedding model (singleton)
│   └── prompt_optimizer.py # Prompt optimization for cloud
│
├── cli.py                  # Interactive REPL client
├── test_local_model.py     # LM Studio diagnostic
└── tests/                  # 21 tests
    ├── test_cascade.py     # 14 tests (ML + rule-based + cascade)
    └── test_session.py     # 7 tests (sessions, TTL, trimming)
```

---

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Specific test
python -m pytest tests/test_cascade.py::TestRouterEngine -v -s

# With coverage
python -m pytest tests/ --cov=. --cov-report=html

# Windows (if python is not found)
py -m pytest tests/ -v
```

**21 tests:**
- SemanticClassifier (ML + rule-based, multi-step, Russian keywords)
- TaskDecomposer (query decomposition)
- RouterEngine (routing, cascade)
- CascadeFlow (subtask processing)
- SessionManager (session creation, messages, TTL, context trimming)

---

## Requirements

- Python 3.10+
- LM Studio with a fast model on your machine
- OpenRouter API key (for cloud models)

---

## Quick Start (cloud only, no LM Studio)

Just want to try the router without a local model?

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`: `cp .env.example .env` (Windows: `copy .env.example .env`)
3. Open `.env` and add your `OPENROUTER_API_KEY`
4. Run: `python main.py`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `sentence-transformers` won't install | `pip install --upgrade pip`; Windows may need Microsoft C++ Build Tools |
| LM Studio timeout | Make sure a model is loaded in LM Studio, `model_name: "auto"` in config.yaml |
| OpenRouter returns errors | Check `OPENROUTER_API_KEY` in `.env`; make sure you have credits |
| All queries go to cloud (0% savings) | LM Studio is unavailable or running a reasoning model (deepseek-r1, o1); load qwen/llama/mistral |
| `curl.exe` not found on Windows | Use `Invoke-RestMethod` in PowerShell (see examples above) |

---

## Roadmap

- [x] v0.1: Basic routing (local/cloud)
- [x] v0.2: Self-assessment, domain thresholds, 404 fallback
- [x] v0.3: ML classifier (sentence-transformers), universal language detection
- [x] v0.4: Refactoring — unified _call_cloud(), ML model singleton, auto-cost metrics
- [x] v0.5: Session support — multi-turn dialogs (session_id + TTL + trimming)
- [x] v0.6: CLI REPL client — interactive chat without curl
- [x] v0.7: OpenAI-compatible endpoint (/v1/chat/completions) — integrate with VS Code, Cursor, Continue
- [ ] v0.8: UI Dashboard (metrics, savings visualization)
- [ ] v0.9: Adaptive routing — router analyzes local model speed and confidence, adjusts thresholds dynamically
- [ ] v0.10: In-agent mode — agent calls router at each loop step
- [ ] v0.10: Ensemble refinement — local generates draft, cloud improves (quality ensemble)
- [ ] v1.0: Docker compose + Prometheus metrics

---

## License

[MIT](LICENSE)