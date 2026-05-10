# AI Cascade Router

[![EN](https://img.shields.io/badge/lang-en-red.svg)](README.en.md) [![RU](https://img.shields.io/badge/lang-ru-blue.svg)](README.md)

Умный прокси-маршрутизатор для LLM, который **экономит до 70% токенов** на простых запросах, направляя их в локальные модели, а сложные — в облако.

> Реальная экономия зависит от сценария: чат-бот / перевод → 60-80%, смешанная нагрузка → 35-55%, анализ и рефакторинг → 5-15%.

---

## Суть технологии

### Проблема

Облачные LLM (GPT, Claude, Gemini) стоят от $0.15 до $15 за миллион токенов. При высокой нагрузке счета растут очень быстро.

### Решение

Большинство запросов к LLM (60-80%) — простые: короткие вопросы, базовый код, приветствия, перевод фраз. Их можно обработать **локальной моделью** (Qwen, Llama, Mistral через LM Studio) — быстро и бесплатно. Сложные задачи (анализ данных, рефакторинг, стратегия) уходят в **облачную модель** (любую но через OpenRouter) — качественно, но платно.

**AI Cascade Router** — это прокси-сервер, который в реальном времени:

1. **Классифицирует** запрос по домену (код, чат, анализ...) и сложности
2. **Проверяет кэш** — возможно, похожий запрос уже был, ответ готов
3. **Принимает решение** — LOCAL (локально, бесплатно), CLOUD (облако, платно) или HYBRID (локальная выжимка + облако)
4. Если запрос многоступенчатый — **декомпозирует** на подзадачи (CASCADE) и каждую маршрутизирует отдельно
5. **Self-assessment** — модель оценивает свою уверенность; если низкая — fallback в облако
6. **Сохраняет сессию** — при передаче `session_id` история диалога передаётся модели как контекст
7. **Считает экономию** — каждый запрос фиксирует сколько токенов сэкономлено

```
                        ┌─────────────────┐
                        │  Запрос клиента  │
                        └────────┬────────┘
                                 │
                        ┌────────v────────┐
                        │  Семантический  │
                        │  кэш (hit?)     │
                        └───┬────────┬───┘
                       hit  │        │ miss
                      ┌─────v──┐     │
                      │ cache  │     │
                      │ответ   │     │
                      └────┬───┘     │
                           │        ┌─v──────────────┐
                           │        │  Классификация  │
                           │        │  (ML + rules)   │
                           │        └──┬──┬───┬──┬───┘
                           │       local│  │hybrid│cascade
                           │     ┌──────v──┐ │  ┌──v───────┐
                           │     │  LOCAL  │ │  │ CASCADE  │
                           │     │модель на│ │  │декомпози- │
                           │     │ LMSTudio│ │  │ция → N    │
                           │     └──┬──────┘ │  │подзадач   │
                           │       ┌───v┐    │  └──────────┘
                           │       │conf< →    │
                           │       │thresh│   │
                           │       └─┬─┬─┘    │
                           │    ok   │ │ low  │
                           │  ┌──────v │  ┌──v──────────┐
                           │  │return  │  │  CLOUD      │
                           │  │ответ   │  │  (OpenRouter)│
                           │  └────────┘  └─────────────┘
                           │
                        ┌──v──────────────┐
                        │  Метрики:       │
                        │  tokens_saved,  │
                        │  cost, время    │
                        └─────────────────┘
```

---

## Ключевые фичи

| Фича | Описание |
|------|----------|
| **ML-классификатор** | Sentence-Transformers (all-MiniLM-L6-v2) + референсные запросы, авто-fallback на правила |
| **Session Support** | Многотурные диалоги — история сохраняется по `session_id`, авто-обрезка контекста |
| **Semantic Cache** | Кэш с эмбеддингами — находит похожие запросы, даже если формулировка другая |
| **Self-Assessment** | Модель сама оценивает свою уверенность в ответе |
| **Domain Thresholds** | Разные пороги для кода, анализа, чата — код чаще локально, анализ — в облако |
| **Cascade Decomposition** | Сложные многоступенчатые запросы разбиваются на подзадачи |
| **404 Fallback** | Если облачная модель недоступна — автоматически пробует следующую |
| **Universal Language** | Работает с любым языком (русский, английский, китайский...) — ответ на языке запроса |
| **Smart Token Savings** | Математический подсчёт экономии: `(1 - cloud_used/baseline) × 100%` |
| **OpenAI-совместимый API** | Endpoint `/v1/chat/completions` — подключается к VS Code, Cursor, Continue.dev, Cline без плагинов |

> **⚠️ Важно:** Cascade-режим (декомпозиция многоступенчатых запросов) и Hybrid-режим требуют локальной модели со скоростью генерации **от 25 токенов/с**. При более медленных моделях возможны таймауты (дефолтный timeout — 120с). Так что подбирайте более менее быструю модель под свою машину.

---

## Как это работает (подробно)

### 1. Классификация запроса

При поступлении запроса система определяет:

**Домен** (тип задачи):
- `code` — генерация кода, отладка
- `analysis` — анализ данных, сравнение
- `chat` — простой чат
- `question` — вопросы
- `creative` — креативное письмо
- `technical` — технические задачи

**Сложность**:
- Простой (< 200 символов, без ключевых слов)
- Средний (сложные ключевые слова)
- Сложный (многоступенчатый)

**Multi-step?** (нужна декомпозиция):
- Запросы с "и", "затем", "также", "и потом" → декомпозиция на подзадачи

### 2. ML-классификатор (v0.3)

Вместо простых правил используется ML:
- Модель: `all-MiniLM-L6-v2` (80MB, ~20ms на запрос)
- Референсные запросы для каждого класса сложности
- Косинусное сходство эмбеддингов → определение класса
- Авто-fallback на rule-based при недоступности ML

### 3. Определение маршрута

По домену выбирается порог уверенности:

```yaml
domain_thresholds:
  code: 0.65        # Код — чаще локально (ниже порог)
  analysis: 0.80    # Анализ — чаще в облако (выше порог)
  chat: 0.60        # Чат — почти всегда локально
  question: 0.55    # Вопросы — минимум порог
  creative: 0.70     # Креатив — средний
  technical: 0.75     # Технические — выше среднего
```

Логика:
- Если уверенность модели ≥ порог → **LOCAL**
- Если уверенность < порог → **CLOUD**
- Для сложных ключевых слов → **HYBRID**

### 4. Обработка в локальном режиме

```
Запрос → LM Studio (ваша модель) → Ответ + Self-Assessment
```

**Self-Assessment** — модель сама оценивает свой ответ:
```json
{
  "confidence": 0.85,
  "flags": ["неизвестный_термин", "недостаточно_контекста"]
}
```

Если запрос содержит критичные флаги → повышается шанс делегирования в облако.

### 5. Обработка в гибридном режиме

```
Запрос → Локальная модель (извлечение контекста) → Облако (финальный ответ)
```

Локальная модель извлекает ключевую информацию → облачная модель использует этот контекст для более точного ответа.

Экономия: токены затраченные на локальный preprocessing.

### 6. Каскадный режим (Cascade)

Для многоступенчатых запросов:
```
"Проанализируй данные и создай отчет с графиками"
          ↓
Декомпозиция:
  1. "Проанализируй данные" → LOCAL
  2. "Создай отчет" → CLOUD
  3. "Добавь графики" → HYBRID
```

Каждая подзадача маршрутизируется отдельно.

### 7. Семантический кэш

При поступлении запроса:
1. Проверяется точный хэш запроса
2. Если нет совпадения → семантический поиск (cosine similarity > 0.8)
3. При cache hit → возвращается сохранённый ответ без обращения к модели

Эмбеддинги: `all-MiniLM-L6-v2` (80MB, ~20ms на запрос).

### 8. Language Detection

Система автоматически определяет язык запроса и добавляет соответствующий system prompt:
- Любой не-ASCII язык (русский, китайский, арабский...) → "Please respond in the same language as the user's question."
- Ответ всегда на языке запроса

---

## Подсчет экономии

**Математическая модель:**
```
baseline_tokens = prompt_tokens + estimated_output_tokens (rolling average)
actual_cloud_tokens = фактические токены облака
tokens_saved = max(0, baseline_tokens - actual_cloud_tokens)
tokens_saved_percent = max(0, (1 - actual_cloud_tokens / baseline_tokens) × 100)
```

| Режим | Что считается |
|-------|----------------|
| **local** | 100% экономии (все локально) |
| **hybrid** | Экономия = baseline - cloud_tokens |
| **cascade** | Сумма экономии всех подзадач |
| **cloud** | 0% (платили облаку) |

Каждый запрос обновляет накопительную метрику.

---

## Dashboard & Метрики

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

### UI Dashboard (планируется)

Показывает не только проценты, а связку метрик:

🦜 **Локальная модель обработала 73% запросов**  
💰 **Сэкономлено 1.2M токенов (~$4.80)**  
⚡ **Средняя задержка снижена на 40%**  
📊 **ROI: 320%** (экономия vs затраты на поддержку)

---

## Сравнение с аналогами

| Проект | Экономия | Особенности |
|--------|----------|-------------|
| **RouteLLM** | до 85% | ML классификатор (DistilBERT), обучение на Arena данных, stateless |
| **LiteLLM** | — | AI Gateway для 100+ провайдеров, без встроенного роутинга, stateless |
| **Наш проект** | до 70% | LM Studio + cloud model + ML-классификатор + semantic cache + сессии + мультиязычность |

---

## Установка

### 1. Клонировать и установить зависимости

```bash
git clone <repo-url>
cd ai-cascade-router
pip install -r requirements.txt
```

> **Windows:** если `pip install` не сработал, попробуй `py -m pip install -r requirements.txt`

### 2. Настроить LM Studio (необходимо для локального режима и экономии)

1. Скачать [LM Studio](https://lmstudio.ai/)
2. Скачать любую модель (рекомендуется Qwen, Mistral, Gemma и т.д)
3. Запустить LM Studio и загрузить модель
4. Убедиться что API доступен на http://localhost:1234/v1
5. В `config.yaml` поле `model_name` можно оставить `"auto"` — роутер сам возьмёт загруженную модель

### 3. Выбрать облачную модель (опционально)

По умолчанию стоит `openai/gpt-4o-mini` ($0.15/1M токенов) — дешёвая и быстрая. Хочешь другую — открой `config.yaml` и измени `model_name` в секции `cloud_model`.

**Где взять имя модели:**
1. Зайди на https://openrouter.ai/models
2. Выбери модель (например Claude Sonnet, Gemini Pro, DeepSeek)
3. Скопируй её **slug** — это строка из адресной строки браузера, например:
   - `https://openrouter.ai/anthropic/claude-sonnet-4` → `anthropic/claude-sonnet-4`
   - `https://openrouter.ai/google/gemini-2.5-pro` → `google/gemini-2.5-pro`
4. Впиши этот slug в `config.yaml`:
   ```yaml
   cloud_model:
     model_name: "anthropic/claude-sonnet-4"
   ```

### 4. Настроить .env

```bash
cp .env.example .env
```

Добавить в `.env`:
```
OPENROUTER_API_KEY=ваш_ключ_openrouter
```

### 5. Запуск

```bash
python main.py
```

Сервер запустится на http://localhost:8000

### 6. CLI-клиент (интерактивный чат)

```bash
python cli.py
```

Просто печатай запросы — без кавычек, без curl, без JSON. Контекст диалога сохраняется автоматически.

```
============================================================
  AI Cascade Router CLI
  Type your query and press Enter.
  /exit  — quit
  /new   — reset session (start fresh)
============================================================
напиши функцию факториала на Python
...
  [local->cloud, saved 0 tok, 2300ms]

добавь обработку ошибок
...
  [local->cloud, saved 0 tok, 1450ms]
```

### 7. Docker (опционально)

```bash
docker build -t ai-cascade-router .
docker run -p 8000:8000 --env-file .env ai-cascade-router
```

---

## Конфигурация (config.yaml)

### Локальная модель

```yaml
local_model:
  base_url: "http://localhost:1234/v1"
  model_name: "auto"  # "auto" = использует то, что загружено в LM Studio
  timeout_seconds: 120
```

### Облачная модель (OpenRouter)

Меняй `model_name` на любой slug с https://openrouter.ai/models

```yaml
cloud_model:
  provider: "openrouter"
  model_name: "openai/gpt-4o-mini"  # openai/gpt-4o-mini, anthropic/claude-sonnet-4, google/gemini-2.5-pro, deepseek/deepseek-chat...
  base_url: "https://openrouter.ai/api/v1"
  # api_key loaded from OPENROUTER_API_KEY env var
```

> **Как узнать slug модели:** зайди на OpenRouter → выбери модель → скопируй её идентификатор из URL.  
> Пример: URL `https://openrouter.ai/anthropic/claude-sonnet-4` → slug `anthropic/claude-sonnet-4`.

### Семантический кэш

```yaml
cache:
  max_size_mb: 100
  ttl_responses: 3600       # 1 час для ответов
  ttl_computations: 86400   # 24 часа для вычислений
```

### Метрики

```yaml
metrics:
  maintenance_cost_per_hour: 0.01
  cloud_cost_per_1k_tokens: 0.001
```

### Сессии

```yaml
session:
  enabled: true
  ttl_minutes: 30       # Время жизни сессии
  max_context_tokens: 8192  # 8K — оптимум для кода. Для анализа можно снизить до 4K, для сложных проектов — поднять до 16K
```

---

## API Endpoints

### POST /route

Основной endpoint для маршрутизации.

```bash
# Linux/macOS
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Привет, как дела?", "task_type": "chat"}'

# Windows PowerShell
curl.exe -X POST http://localhost:8000/route -H "Content-Type: application/json" -d '{"query": "Привет, как дела?", "task_type": "chat"}'

# Windows PowerShell (альтернатива)
Invoke-RestMethod -Uri "http://localhost:8000/route" -Method Post -ContentType "application/json" -Body '{"query": "Привет, как дела?", "task_type": "chat"}'
```

**Параметры:**
- `query` (обяз.) — текст запроса
- `task_type` (опц.) — `chat`, `code`, `simple`, `question`, `analysis`, `creative`, `technical`
- `session_id` (опц.) — идентификатор сессии для многотурного диалога
- `context` (опц.) — дополнительный контекст (dict)

**Response:**
```json
{
  "response": "Привет! У меня всё хорошо, спасибо!",
  "source": "local",
  "confidence": 0.92,
  "tokens_saved": 38,
  "tokens_saved_percent": 100.0,
  "response_time_ms": 1245.6
}
```

### GET /metrics

Метрики сессии.

```bash
curl http://localhost:8000/metrics
```

### GET /health

Проверка здоровья.

### POST /v1/chat/completions (OpenAI-совместимый)

Для интеграции с VS Code, Cursor, Continue.dev, Cline и любыми инструментами, поддерживающими OpenAI API.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "router", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Настройка в инструментах:**

- **Continue.dev:** в `config.json` → `"models": [{"title": "Cascade Router", "provider": "openai", "apiBase": "http://localhost:8000/v1"}]`
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

Поле `cascade_meta` содержит служебную информацию: `source` (откуда ответ), `response_time_ms`.

---

## Структура проекта

```
ai-cascade-router/
├── main.py                 # FastAPI приложение (точка входа)
├── config.yaml             # Конфигурация
├── README.md               # Этот файл
├── requirements.txt        # Зависимости
├── Dockerfile              # Docker конфигурация
├── cli.py                  # Интерактивный REPL-клиент (чат без curl)
├── .gitignore             # Git ignore файлы
├── .env.example           # Шаблон .env
├── LICENSE                # MIT лицензия
│
├── session/                # Сессии (многотурные диалоги)
│   └── manager.py          # SessionManager — хранение истории, TTL, обрезка
│
├── cli.py                  # Интерактивный REPL-клиент (чат без curl)
├── test_local_model.py     # Диагностика LM Studio (быстрый тест)
└── tests/                  # 21 тест
    ├── test_cascade.py     # 14 тестов (ML + rule-based + cascade)
    └── test_session.py     # 7 тестов (сессии, TTL, обрезка контекста)
```

---

## Тестирование

```bash
# Все тесты
python -m pytest tests/ -v

# Конкретный тест
python -m pytest tests/test_cascade.py::TestRouterEngine -v -s

# С покрытием
python -m pytest tests/ --cov=. --cov-report=html

# Windows (если python не находит)
py -m pytest tests/ -v
```

**21 тестов:**
- SemanticClassifier (ML + rule-based, multi-step, русские ключевые слова)
- TaskDecomposer (декомпозиция запросов)
- RouterEngine (роутинг, каскад)
- CascadeFlow (обработка подзадач)
- SessionManager (создание сессий, сообщения, TTL, обрезка контекста)

---

## Требования

- Python 3.10+
- LM Studio с быстро работающей моделью на вашей машине
- OpenRouter API ключ (для облачных моделей)

---

## Быстрый старт (только облако, без LM Studio)

Если хочешь просто попробовать роутер без локальной модели:

1. Установи зависимости: `pip install -r requirements.txt`
2. Скопируй `.env.example` в `.env`: `cp .env.example .env` (или в Windows: `copy .env.example .env`)
3. Открой `.env` и впиши свой `OPENROUTER_API_KEY`
4. Запусти: `python main.py`

---

## Troubleshooting

| Проблема | Решение |
|----------|---------|
| `sentence-transformers` не устанавливается | `pip install --upgrade pip`; на Windows может потребоваться Microsoft C++ Build Tools |
| LM Studio не отвечает (таймаут) | Проверь что модель загружена в LM Studio, `model_name: "auto"` в config.yaml |
| OpenRouter возвращает ошибку | Проверь `OPENROUTER_API_KEY` в `.env`; убедись что на счету есть средства |
| Все запросы идут в облако (0% экономии) | LM Studio недоступен или загружена reasoning-модель (deepseek-r1, o1) — они медленные; загрузи qwen/llama/mistral |
| `curl.exe` не найден на Windows | Используй `Invoke-RestMethod` в PowerShell (см. примеры выше) |

---

## Roadmap

- [x] v0.1: Базовый роутинг (local/cloud)
- [x] v0.2: Self-assessment, domain thresholds, 404 fallback
- [x] v0.3: ML-классификатор (sentence-transformers), universal language detection
- [x] v0.4: Рефакторинг — единый _call_cloud(), синглтон ML-модели, авто-расчёт метрик
- [x] v0.5: Session support — многотурные диалоги с сохранением контекста (session_id + TTL + обрезка)
- [x] v0.6: CLI REPL-клиент — интерактивный чат без curl
- [x] v0.7: OpenAI-совместимый endpoint (/v1/chat/completions) — интеграция с VS Code, Cursor, Continue
- [ ] v0.8: UI Dashboard (метрики, визуализация экономии)
- [ ] v0.9: Адаптивный роутинг — роутер анализирует скорость и уверенность локальной модели, подстраивает пороги динамически
- [ ] v0.10: In-agent mode — агент сам вызывает роутер на каждом шаге loop-а
- [ ] v0.11: Ensemble refinement — local генерирует черновик, cloud улучшает (ансамбль для качества)
- [ ] v1.0: Docker compose для продакшена

---

## Лицензия

[MIT](LICENSE)
