# 🧠 AI Agent Cache Utilities

This repository provides **lightweight caching and memory management utilities** for AI and LLM applications.
It includes tools for managing **in-memory caches**, **model lifecycle (LRU) management**, and **result caching with TTL and LRU eviction**.

---

## Overview

When building AI agents or applications that frequently interact with LLMs, model embeddings, or other expensive-to-compute results, caching can **significantly reduce latency and cost**.

This library includes:

| Class               | Description                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **`MemoryCache`**   | Generic LRU memory cache with max size (in MB). Ideal for storing large data objects like LLM responses.               |
| **`ModelLRUStore`** | Keeps a limited number of model instances in memory (LRU-based). Prevents repeated model loading.                      |
| **`ResultCache`**   | Stores computation results with time-to-live (TTL) and LRU eviction. Great for caching embeddings, API responses, etc. |

---

## Installation

You can copy this module directly into your project, or install dependencies manually if you want enhanced memory tracking:

```bash
pip install pympler
```

> `pympler` is optional — it allows for more accurate memory size estimation.

---

##  Usage Examples

### 1. MemoryCache

```python
from cache_utils import MemoryCache

# Create a cache with a 100MB memory limit
cache = MemoryCache(max_size_mb=100)

# Store responses
cache.set("response:123", {"text": "Hello, world!"})

# Retrieve them later
value = cache.get("response:123")
print(value)  # {"text": "Hello, world!"}

# Check stats
print(cache.stats())
# Output: {'entries': 1, 'used_MB': 0.01, 'max_MB': 100.0}
```

Automatically removes least-recently-used (LRU) entries when exceeding memory limit.
Thread-safe for concurrent agent access.

---

### 2. ModelLRUStore

```python
from cache_utils import ModelLRUStore

# Keep up to 3 loaded models in memory
model_store = ModelLRUStore(max_models=3)

# Simulate models (could be HuggingFace, OpenAI, etc.)
model_store.set("bert-base", "Loaded BERT model")
model_store.set("gpt-small", "Loaded GPT model")

# Access a model (refreshes its LRU position)
model = model_store.get("bert-base")

# Add more models — oldest one will be evicted when full
model_store.set("mistral", "Loaded Mistral model")
model_store.set("llama", "Loaded Llama model")

print(model_store.list_loaded())
# Output: ['gpt-small', 'mistral', 'llama']
```

Prevents repeated loading of large models.
Thread-safe and lightweight.

---

### 3. ResultCache

```python
from cache_utils import ResultCache
import time

# Cache up to 500 results, each valid for 1 hour (3600s)
results = ResultCache(max_entries=500, ttl=3600)

# Store a model or tool result
results.set("embedding:query1", [0.1, 0.2, 0.3])

# Retrieve it
print(results.get("embedding:query1"))  # [0.1, 0.2, 0.3]

# Wait for expiration (TTL)
time.sleep(2)
print(results.get("embedding:query1"))  # Still valid within TTL

# Inspect cache stats
print(results.stats())
# Output: {'entries': 1, 'max_entries': 500}
```

Supports **time-to-live (TTL)** expiration.
Evicts oldest items when reaching `max_entries`.

---

## 🧰 API Reference

### `MemoryCache(max_size_mb=50)`

* `set(key, value)` — Store a value.
* `get(key)` — Retrieve value, update LRU order.
* `clear()` — Remove all items.
* `stats()` — Return dict with usage stats.

### `ModelLRUStore(max_models=3)`

* `set(model_id, model)` — Add or update a model.
* `get(model_id)` — Retrieve model and update LRU.
* `list_loaded()` — Return loaded model IDs.
* `unload(model_id)` — Remove a model manually.

### `ResultCache(max_entries=500, ttl=3600)`

* `set(key, value)` — Store a cached result.
* `get(key)` — Return result if valid and not expired.
* `clear()` — Empty the cache.
* `stats()` — Return current entry count.

---

## Logging Support

You can enable debug-level logging to see cache eviction activity:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Notes

* Thread-safe for concurrent agent execution.
* Optional dependency: `pympler` for detailed object size measurement.
* Designed to integrate with **AI agent frameworks**, **retrieval systems**, and **LLM pipelines**.

