[![ruff](https://img.shields.io/badge/ruff-checked-red)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
[![tests](https://img.shields.io/github/actions/workflow/status/xor2003/llm_router/main.yml?branch=main)](https://github.com/xor2003/llm_router/actions)
[![coverage](https://img.shields.io/codecov/c/github/xor2003/llm_router/main)](https://codecov.io/gh/xor2003/llm_router)

# LLM Proxy Router

A LLM proxy server that intelligently routes requests to various LLM providers (OpenAI, Gemini) with failover handling, rate limiting, and tool call support. Compatibel to LiteLLM config file.

The application will automatically capture and report all unhandled exceptions to a local file.

## Features
- **Model Routing**: Routes requests to backend models with automatic failover
- **Rate Limiting**: Handles provider rate limits with cooldown tracking
- **XML Tool Workaround**: Converts tool calls for models without native support
- **Streaming Support**: Full streaming response compatibility
- **State Management**: Tracks model health and availability
- **Local Logging**: All exceptions are logged to `logs/exceptions.json` for local analysis.
- **PII Scrubbing**: Automatically scrubs PII from logs to protect user privacy.

## Getting Started

### Prerequisites
- Python 3.10+
- Poetry

### Installation
```bash
git clone https://github.com/your-repo/llm-proxy-router.git
cd llm-proxy-router
poetry install
```

### Configuration
Create `config.yaml` with your model configurations:
```yaml
model_list:
  - id: "gemini-1"
    group_name: "creative"
    model_name: "gemini/gemini-2.5-pro"
    api_key: ${GEMINI_API_KEY}
    provider: "gemini"

  - id: "gpt-4o"
    group_name: "analytical"
    model_name: "openrouter/openai/gpt-4o"
    api_key: ${OPENROUTER_KEY}
    api_base: "https://openrouter.ai/api/v1"
    provider: "openai"
```

### Running the Server
```bash
poetry run python main.py
```

## Architecture Overview

### Core Components
1. **Router (`app/router.py`)**: 
   - Manages model groups and failover logic
   - Implements smart routing using `get_active_or_failover_model()`
2. **Client (`app/client.py`)**:
   - Handles communication with LLM providers
   - Translates requests/responses between formats
3. **State Manager (`app/state.py`)**:
   - Tracks model health and cooldown states
4. **Dependencies (`app/dependencies.py`)**:
   - Provides dependency injection for components

### Request Flow
1. Request received at `/v1/chat/completions`
2. Router selects appropriate model
3. Client sends request to backend provider
4. If tool call needed and unsupported, applies XML workaround
5. Response streamed back to client

## API Endpoints

### `GET /v1/models`
Lists available model groups.

### `POST /v1/chat/completions`
Main endpoint for chat completions. Supports:
- Standard OpenAI-compatible requests
- Streaming responses
- Tool calls (native or via XML workaround)

**Request Format**:
```json
{
  "model": "group-name",
  "messages": [...],
  "tools": [...],
  "stream": true/false
}
```

## XML Tool Workaround
For models without native tool call support:
1. Inject XML-formatted prompt
2. Parse model response for XML tags
3. Convert XML to OpenAI-style tool call format

## Configuration Reference
`config.yaml` options:
- `model_list`: List of backend models
- `proxy_server_config`: Host/port settings
- `mcp_tool_use_prompt_template`: Template for XML workaround

## Development
Run tests:
```bash
poetry run pytest tests/