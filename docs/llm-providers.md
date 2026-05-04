# LLM Provider Configuration

MeetingBro uses any **OpenAI-compatible API** for meeting summaries and translation.
The three environment variables that control it are:

```env
MEETINGBRO_LLM_API_KEY=your_api_key_here
MEETINGBRO_LLM_BASE_URL=https://api.openai.com/v1
MEETINGBRO_LLM_MODEL=gpt-4o-mini
```

**If you leave these unset**, MeetingBro still works: transcription runs locally via Whisper,
and summaries fall back to local heuristics (bullet-point extraction without LLM quality).

---

## Provider Options

### 1. OpenAI

The reference provider. All other providers listed here are compatible with the same API format.

| | |
|---|---|
| Sign up | https://platform.openai.com/signup |
| API keys | https://platform.openai.com/api-keys |
| Pricing | https://openai.com/api/pricing |
| Data policy | https://openai.com/policies/api-data-usage-policies |

**Recommended model:** `gpt-4o-mini` — good quality at low cost. Use `gpt-4o` for better summarization of long or complex meetings.

```env
MEETINGBRO_LLM_API_KEY=sk-...
MEETINGBRO_LLM_BASE_URL=https://api.openai.com/v1
MEETINGBRO_LLM_MODEL=gpt-4o-mini
```

**Notes:**
- Pay-per-use pricing. New accounts may receive trial credits — check the dashboard.
- Meeting summaries and translations involve relatively short text prompts; typical session cost is in the low cents range with `gpt-4o-mini`.
- Audio and transcription never leave your machine; only the text transcript is sent to the LLM.

---

### 2. OpenRouter

An aggregator that routes to 200+ models from OpenAI, Anthropic, Meta, Mistral, and others through a single API. Useful if you want to switch models without changing infrastructure.

| | |
|---|---|
| Sign up | https://openrouter.ai |
| API keys | https://openrouter.ai/keys |
| Pricing | https://openrouter.ai/models (per-model) |
| Data policy | https://openrouter.ai/privacy |

```env
MEETINGBRO_LLM_API_KEY=sk-or-v1-...
MEETINGBRO_LLM_BASE_URL=https://openrouter.ai/api/v1
MEETINGBRO_LLM_MODEL=openai/gpt-4o-mini
```

**Model name format:** `provider/model-name`, e.g.:
- `openai/gpt-4o-mini`
- `anthropic/claude-3-haiku`
- `meta-llama/llama-3.1-8b-instruct`
- `mistralai/mistral-small`

**Notes:**
- Some models on OpenRouter are available with a free rate-limited tier — see https://openrouter.ai/models?q=free for the current list.
- Pricing and availability change frequently; verify before choosing a model for production use.

---

### 3. Groq

A cloud inference service focused on very fast response times (low-latency LLM inference via custom hardware). Good for meeting use cases where you want summaries to appear quickly.

| | |
|---|---|
| Sign up | https://console.groq.com |
| API keys | https://console.groq.com/keys |
| Pricing | https://console.groq.com/settings/billing |
| Supported models | https://console.groq.com/docs/models |

```env
MEETINGBRO_LLM_API_KEY=gsk_...
MEETINGBRO_LLM_BASE_URL=https://api.groq.com/openai/v1
MEETINGBRO_LLM_MODEL=llama-3.1-8b-instant
```

**Notes:**
- Groq has historically offered a free tier with rate limits. Check current plan details at the link above.
- Models available through Groq are open-weight models (Llama, Mixtral, Gemma); the quality for structured summaries is generally good.
- Very fast generation speed compared to other providers.

---

### 4. Mistral AI

Mistral's own hosted inference for their models. Good for European users who prefer EU-based data processing.

| | |
|---|---|
| Sign up | https://console.mistral.ai |
| API keys | https://console.mistral.ai/api-keys |
| Pricing | https://mistral.ai/technology/#pricing |
| Data policy | https://mistral.ai/terms |

```env
MEETINGBRO_LLM_API_KEY=...
MEETINGBRO_LLM_BASE_URL=https://api.mistral.ai/v1
MEETINGBRO_LLM_MODEL=mistral-small-latest
```

**Notes:**
- Models are hosted on EU servers, which may be relevant for GDPR compliance.
- `mistral-small-latest` offers a good price-to-quality trade-off for summarization.
- Check the console for any available free-tier or trial credits.

---

### 5. Ollama (fully local, no API key required)

Run an LLM entirely on your own machine. No data leaves your device and no API key is needed. Performance depends on your hardware (CPU-only is slow for large models; GPU recommended).

| | |
|---|---|
| Install | https://ollama.com/download |
| Model library | https://ollama.com/library |

**Setup:**
```bash
# Install Ollama, then pull a model
ollama pull llama3.2
# Ollama starts a local server on port 11434
ollama serve
```

```env
MEETINGBRO_LLM_API_KEY=ollama
MEETINGBRO_LLM_BASE_URL=http://localhost:11434/v1
MEETINGBRO_LLM_MODEL=llama3.2
```

**Recommended models by hardware:**

| Hardware | Recommended model |
|---|---|
| 8 GB RAM (CPU) | `llama3.2:3b` or `qwen2.5:3b` |
| 16 GB RAM (CPU) | `llama3.2` (8B) or `qwen2.5:7b` |
| GPU with 8 GB VRAM | `llama3.1:8b` or `mistral` |
| GPU with 16+ GB VRAM | `llama3.1:70b-q4` |

**Notes:**
- Summary and translation quality with small local models (3B–8B) is noticeably lower than GPT-4o-mini class models.
- Suitable for privacy-sensitive environments or offline use.
- The `MEETINGBRO_LLM_API_KEY` value is not checked by Ollama but the variable must be set to a non-empty string.

---

### 6. Together AI

A cloud inference platform with a wide model selection and competitive pricing.

| | |
|---|---|
| Sign up | https://api.together.xyz |
| API keys | https://api.together.xyz/settings/api-keys |
| Pricing | https://www.together.ai/pricing |

```env
MEETINGBRO_LLM_API_KEY=...
MEETINGBRO_LLM_BASE_URL=https://api.together.xyz/v1
MEETINGBRO_LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
```

**Notes:**
- Check the website for any current trial credits for new accounts.
- Good selection of open-weight models at low cost.

---

### 7. AWS Bedrock

Amazon's managed LLM service. It does not expose a native OpenAI-compatible endpoint, but you can front it with a proxy.

**Option A — LiteLLM proxy (recommended for teams):**
```bash
pip install litellm
litellm --model bedrock/anthropic.claude-3-haiku-20240307-v1:0 --port 8080
```
```env
MEETINGBRO_LLM_API_KEY=any_value
MEETINGBRO_LLM_BASE_URL=http://localhost:8080/v1
MEETINGBRO_LLM_MODEL=bedrock/anthropic.claude-3-haiku-20240307-v1:0
```

**Notes:**
- Requires AWS credentials configured via `~/.aws/credentials` or environment variables.
- Useful for enterprise deployments that already use AWS and need to keep data within a VPC.
- LiteLLM documentation: https://docs.litellm.ai/docs/providers/bedrock

---

## Choosing a provider

| Priority | Recommendation |
|---|---|
| Lowest cost | Groq free tier or OpenRouter free models (check current availability) |
| Best quality | OpenAI `gpt-4o` or `gpt-4o-mini` |
| Privacy / no data leaving machine | Ollama (local) |
| EU data residency | Mistral AI |
| Enterprise / existing AWS | AWS Bedrock via LiteLLM |
| Flexible model switching | OpenRouter |

---

## Privacy reminder

Only the **text transcript** is sent to the LLM provider you configure. Audio capture and Whisper transcription run entirely on your machine. If you use a local LLM such as Ollama, no data leaves your device at all.

To verify this, see [app/backend/meetingbro/llm/openai_compatible.py](../app/backend/meetingbro/llm/openai_compatible.py).
