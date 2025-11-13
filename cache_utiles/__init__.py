#####################################
# Note:
# This code was created while trying to build an AI agents library.
#####################################

import threading, sys, time, logging
from collections import OrderedDict
try:
    from pympler import asizeof
except ImportError:
    asizeof = None

log = logging.getLogger(__name__)

class MemoryCache:
    """Generic LRU cache with max memory size in MB.
    Examples to use: Storing and sorting responses from the LLM model
    """

    def __init__(self, max_size_mb=50):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.total_size = 0
        self.lock = threading.Lock()

    def _estimate_size(self, obj):
        if asizeof:
            try: return asizeof.asizeof(obj)
            except Exception: pass
        try: return sys.getsizeof(obj)
        except Exception: return 1024

    def get(self, key):
        with self.lock:
            if key not in self.cache: return None
            value, size = self.cache.pop(key)
            self.cache[key] = (value, size)
            return value

    def set(self, key, value):
        with self.lock:
            if key in self.cache:
                _, old_size = self.cache.pop(key)
                self.total_size -= old_size
            size = self._estimate_size(value)
            self.cache[key] = (value, size)
            self.total_size += size
            while self.total_size > self.max_size_bytes and self.cache:
                old_key, (_, old_size) = self.cache.popitem(last=False)
                self.total_size -= old_size
                log.debug(f"Evicted {old_key} ({old_size/1024:.1f} KB)")
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.total_size = 0

    def stats(self):
        return {
            "entries": len(self.cache),
            "used_MB": round(self.total_size / (1024 * 1024), 2),
            "max_MB": self.max_size_bytes / (1024 * 1024),
        }

    def __contains__(self, key): return key in self.cache
    def __len__(self): return len(self.cache)


# ======================================================
# ⚙️ Model LRU Manager
# ======================================================

class ModelLRUStore:
    """
    Keeps loaded HuggingFace (or other) model instances alive in memory.
    Prevents repeated loading, with LRU eviction.
    """

    def __init__(self, max_models: int = 3):
        self.max_models = max_models
        self.models = OrderedDict()
        self.lock = threading.Lock()

    def get(self, model_id):
        with self.lock:
            if model_id not in self.models:
                return None
            model = self.models.pop(model_id)
            self.models[model_id] = model  # move to end
            return model

    def set(self, model_id, model):
        with self.lock:
            if model_id in self.models:
                self.models.pop(model_id)
            elif len(self.models) >= self.max_models:
                self.models.popitem(last=False)
            self.models[model_id] = model

    def list_loaded(self):
        with self.lock:
            return list(self.models.keys())

    def unload(self, model_id):
        with self.lock:
            if model_id in self.models:
                self.models.pop(model_id)

# ======================================================
# 🧩 Result Cache (for LLM responses or tool results)
# ======================================================
class ResultCache:
    """
    Caches model/tool results to avoid recomputation.
    Supports:
    - Auto-expiry by TTL
    - LRU eviction by max_entries

    Example: Storage of embedded and vectors, etc.
    """

    def __init__(self, max_entries: int = 500, ttl: int = 3600):
        """
        :param max_entries: Maximum number of cached results.
        :param ttl: Time-to-live for each cached item (seconds).
        """
        self.cache = OrderedDict()
        self.max_entries = max_entries
        self.ttl = ttl
        self.lock = threading.Lock()

    def _is_expired(self, item):
        return (time.time() - item["timestamp"]) > self.ttl

    def get(self, key):
        """Return cached value if valid."""
        with self.lock:
            if key not in self.cache:
                return None

            item = self.cache.pop(key)
            if self._is_expired(item):
                return None

            # move to end (LRU)
            self.cache[key] = item
            return item["value"]

    def set(self, key, value):
        """Store result with timestamp and evict if needed."""
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)

            self.cache[key] = {
                "value": value,
                "timestamp": time.time()
            }

            # Evict if full
            if len(self.cache) > self.max_entries:
                self.cache.popitem(last=False)

    def clear(self):
        with self.lock:
            self.cache.clear()

    def stats(self):
        with self.lock:
            return {
                "entries": len(self.cache),
                "max_entries": self.max_entries,
            }