# Session / Context Support — v0.6

## Цель

Добавить поддержку многотурных диалогов (сессий) в AI Cascade Router.  
Клиент может передавать `session_id` — роутер хранит историю сообщений и передаёт её модели как контекст при каждом запросе.

## Мотивация

Текущая версия (v0.3) работает только как single-turn прокси. Для реального использования (чат-боты, ассистенты, iter-запросы) нужна память диалога.

## Архитектура

### 1. Новый модуль `session/manager.py`

Исполнение: **in-memory** (Redis опционально позже)

```python
class SessionManager:
    def __init__(self, ttl_minutes=30, max_context_tokens=4096)
    
    def get_or_create(self, session_id: str) -> Session
    def add_message(self, session_id: str, role: str, content: str)
    def get_context(self, session_id: str) -> list[dict]
    def _trim_context(self, session)  # старые сообщения удаляются при превышении max_context_tokens
    def _cleanup_expired(self)        # lazy eviction при каждом запросе
```

**Session** — dataclass:
- `id: str`
- `messages: list[dict]`  — `[{"role": "system", "content": ...}, {"role": "user", "content": ...}, ...]`
- `created_at: float`
- `updated_at: float`

**Обрезка контекста:** при превышении `max_context_tokens` удаляются самые старые user/assistant пары, system-prompt остаётся всегда.

### 2. Изменение `main.py`

- `RouteRequest` получает поле `session_id: Optional[str] = None`
- Если `session_id` передан:
  - Создаём/получаем сессию
  - Добавляем user message
  - Передаём `messages=история` в generate()
  - После ответа добавляем assistant message
  - Кэш пропускается для сессионных запросов
- Если не передан — поведение не меняется (stateless)

### 3. Изменение `models/local_engine.py`

`generate()` принимает опциональный параметр `messages: Optional[list[dict]] = None`:
- Если `messages` передан — использует их как `messages` в API-запросе к LM Studio
- Если не передан — собирает из `system_prompt` + `prompt` как сейчас

### 4. Изменение `cloud/client.py`

Аналогично local_engine: `generate()` принимает опциональный `messages`.

### 5. Изменение `cache/manager.py`

Если запрос пришёл с `session_id` — кэш не применяется (ответ всегда уникален в контексте диалога).

### 6. Конфиг `config.yaml`

```yaml
session:
  enabled: true
  ttl_minutes: 30
  max_context_tokens: 4096
```

### 7. Обновление README

- Документация сессий.
- Пример многотурного запроса.
- Columns в таблице API.

## Поток данных

```
Клиент → POST /route {query: "Привет", session_id: "abc-123"}
→ SessionManager.get_or_create("abc-123")
→ SessionManager.add_message("abc-123", "user", "Привет")
→ messages = SessionManager.get_context("abc-123")
→ LocalEngine.generate(messages=messages)  или  CloudClient.generate(messages=messages)
→ SessionManager.add_message("abc-123", "assistant", response)
→ Ответ клиенту
```

## Тестирование

- `tests/test_session.py`:
  - Создание сессии
  - Добавление сообщений
  - Контекст возвращает всю историю
  - Обрезка контекста при превышении max_tokens
  - TTL очистка
  - Stateless запрос (без session_id) не создаёт сессию
