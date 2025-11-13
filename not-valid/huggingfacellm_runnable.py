##############################
# Note: 
# This file based on langchain runnables, 
# It was built to try and make LLM work 
# and replace the library's default, but it is unusable.
##############################

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from langchain_core.runnables import Runnable
from typing import Optional, Dict, Any, List, Union
import torch

from core.config.logging_config import log_info, log_error, log_debug, log_warning
from core.utils.cache_utils import ResultCache, ModelLRUStore

# -----------------------------
# Base class: shared logic
# -----------------------------
class HuggingFaceBase:
    """
    Base class with common init logic:
    - tokenizer loading
    - quantization config helper
    - device detection
    - basic utilities (tokenize)
    This class DOES NOT load a specific model type by itself.
    """
    def __init__(
        self,
        model_id: str,
        device_map: str = "cpu",  # جعلته optional مع افتراضي "cpu" عشان مرونة وتجنب أخطاء
        use_quantization: bool = False,
        quantization_bits: int = 4,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_memory: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        self.model_id = model_id
        self.device = self._get_device()
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.device_map = device_map  # حفظ device_map (cuda أو cpu فقط، بدون auto)
        self.torch_dtype = torch_dtype or (torch.float16 if self.device != "cpu" else torch.float32)
        self.kwargs = kwargs
        self.max_memory = max_memory

        log_info(f"[Base] Loading tokenizer for {model_id}")
        # tokenizer load
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            # set pad token to eos if missing
            try:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception:
                # some tokenizers might not have eos_token either
                self.tokenizer.pad_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token or "")
        # placeholder for model and pipe
        self.model = None
        self.pipe = None

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        # mps may not be stable for large models, but we report it
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_quant_config(self) -> Optional[BitsAndBytesConfig]:
        if not (self.use_quantization or self.kwargs.get("load_in_4bit") or self.kwargs.get("load_in_8bit")):
            return None
        # prefer explicit flags
        load4 = self.kwargs.get("load_in_4bit", False) or (self.use_quantization and self.quantization_bits == 4)
        load8 = self.kwargs.get("load_in_8bit", False) or (self.use_quantization and self.quantization_bits == 8)
        return BitsAndBytesConfig(
            load_in_4bit=load4,
            load_in_8bit=load8,
            bnb_4bit_compute_dtype=torch.float16 if load4 else None,
            bnb_4bit_use_double_quant=True if load4 else False,
            bnb_4bit_quant_type="nf4" if load4 else None
        )

    def tokenize(self, text: Union[str, List[str]], **kwargs):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, **kwargs)

    def clear_cache(self):
        """Clear GPU memory if possible"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        log_info("[Base] Memory cache cleared")

    def unload(self):
        """
        Unload model weights (free memory) and delete references.
        Each subclass should override if necessary to do more precise cleanup.
        """
        try:
            # try to move model to cpu then delete
            if self.model is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
            self.model = None
            self.pipe = None
            self.clear_cache()
            log_info(f"[Base] Unloaded model {self.model_id}")
        except Exception as e:
            log_warning(f"[Base] Error during unload: {e}")


# -----------------------------
# CausalLM class (concrete)
# -----------------------------
class HuggingFaceCausalLM(HuggingFaceBase, Runnable):
    """
    Standalone Causal LM wrapper.
    Handles loading, pipeline creation, and generation logic.
    Automatically uses GPU if available, else CPU.
    """
    def __init__(self, model_id: str, **kwargs):
        # 🔹 تحديد device_map: auto لو GPU متاح (لـ accelerate)، غير كده cpu
        device_map = kwargs.get("device_map")  # لو مرر يدوياً، استخدمه
        if device_map is None:
            device_map = "auto" if torch.cuda.is_available() else "cpu"

        # نداء super مع تمرير device_map صراحة
        super().__init__(model_id, device_map=device_map, **kwargs)

        log_info(f"[CausalLM] Loading model weights for {model_id}")

        quant_config = self._get_quant_config()

        # إعدادات التحميل الأساسية
        model_load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "cache_dir": self.cache_dir,
        }

        # 🔹 استخدام device_map المحدد (auto أو cpu)
        model_load_kwargs["device_map"] = self.device_map  # يستخدم accelerate إذا auto

        # 🔹 تحديد نوع الـ dtype أو quantization
        if quant_config:
            model_load_kwargs["quantization_config"] = quant_config
        else:
            model_load_kwargs["dtype"] = self.torch_dtype

        if self.max_memory:
            model_load_kwargs["max_memory"] = self.max_memory

        # دمج أي إعدادات إضافية من kwargs
        model_load_kwargs.update(self.kwargs or {})

        # 🧠 تحميل الموديل
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_load_kwargs)

        # 🔹 إعداد الـ pipeline بدون تحديد device (لأن accelerate يدير لو device_map="auto")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        log_info(f"[CausalLM] Model {model_id} ready (device={self.device})")

    # -------------------------
    # توليد النصوص (generation)
    # -------------------------
    def _normalize_gen_kwargs(self, **kwargs) -> Dict[str, Any]:
        defaults = dict(
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        defaults.update(kwargs)
        return defaults

    def invoke(
        self,
        prompt: Union[str, List[str]],
        remove_prompt: bool = True,
        use_cache: bool = True,
        result_cache: Optional[ResultCache] = None,
        **generation_kwargs
    ) -> Union[str, List[str]]:
        key = None
        if use_cache and result_cache is not None:
            k_items = tuple(sorted(generation_kwargs.items()))
            key = (self.model_id, prompt if isinstance(prompt, str) else tuple(prompt), k_items)
            cached = result_cache.get(key)
            if cached is not None:
                log_debug("[CausalLM] Returning cached result")
                return cached

        gen_kwargs = self._normalize_gen_kwargs(**generation_kwargs)
        try:
            results = self.pipe(prompt, **gen_kwargs)
            if isinstance(prompt, str):
                outs = self._process_pipeline_result(results, prompt if remove_prompt else None)
            else:
                outs = [self._process_pipeline_result(res, pr if remove_prompt else None)
                        for res, pr in zip(results, prompt)]
            if key is not None:
                result_cache.set(key, outs)
            return outs
        except Exception as e:
            log_error(f"[CausalLM] Generation error: {e}")
            raise

    def _process_pipeline_result(self, result: Union[List[Dict], Dict], prompt: Optional[str] = None) -> Union[str, List[str]]:
        if isinstance(result, dict):
            text = result.get("generated_text", "")
            if prompt and text.startswith(prompt):
                text = text[len(prompt):]
            return text.strip()
        elif isinstance(result, list):
            texts = []
            for r in result:
                t = r.get("generated_text", "")
                if prompt and t.startswith(prompt):
                    t = t[len(prompt):]
                texts.append(t.strip())
            return texts[0] if len(texts) == 1 else texts
        else:
            return str(result)

    def __call__(self, prompt: str, **kwargs):
        return self.invoke(prompt, **kwargs)

    def unload(self):
        try:
            if self.model is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
            self.model = None
            self.pipe = None
            self.clear_cache()
            log_info(f"[CausalLM] Unloaded {self.model_id}")
        except Exception as e:
            log_warning(f"[CausalLM] Unload error: {e}")

# -----------------------------
# ModelManager: single entrypoint to manage multiple models (LRU eviction)
# -----------------------------
class ModelManager:
    """
    Manage multiple HuggingFace model instances with LRU eviction.
    - load_model(model_id, type="causal", **props) -> returns instance
    - get_model(model_id)
    - unload_model(model_id)
    - list_loaded()
    """
    def __init__(self, max_models_loaded: int = 2, result_cache_size: int = 1024, result_ttl: Optional[int] = 60 * 60):
        self.store = ModelLRUStore(max_models=max_models_loaded)
        self.result_cache = ResultCache(maxsize=result_cache_size, ttl=result_ttl)

    def load_model(self, model_id: str, model_type: str = "causal", **props) -> Any:
        """
        Load or return cached model instance.
        model_type currently supports: 'causal'
        props are forwarded to the model constructor (quantization flags, cache_dir, etc.)
        """
        inst = self.store.get(model_id)
        if inst:
            log_info(f"[Manager] Model {model_id} already loaded. Returning cached instance.")
            return inst

        log_info(f"[Manager] Loading model {model_id} as type={model_type}")
        if model_type == "causal":
            inst = HuggingFaceCausalLM(model_id, **props)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.store.put(model_id, inst)
        return inst

    def get_model(self, model_id: str):
        return self.store.get(model_id)

    def unload_model(self, model_id: str):
        inst = self.store.remove(model_id)
        if inst:
            try:
                inst.unload()
                log_info(f"[Manager] Unloaded model {model_id}")
            except Exception as e:
                log_warning(f"[Manager] Error unloading model {model_id}: {e}")
            return True
        return False

    def list_loaded(self) -> List[str]:
        return self.store.list_models()

    def clear_all(self):
        self.store.clear_all()
        self.result_cache.clear()
        log_info("[Manager] Cleared all models and caches.")

    def generate(
        self,
        model_id: str,
        prompt: Union[str, List[str]],
        model_type: str = "causal",
        use_result_cache: bool = True,
        **generation_kwargs
    ) -> Union[str, List[str]]:
        """
        Convenience wrapper:
         - ensure model is loaded
         - call model.invoke with manager's result cache
        """
        model = self.get_model(model_id)
        if not model:
            model = self.load_model(model_id, model_type=model_type, **generation_kwargs.get("loader_props", {}))
        return model.invoke(prompt, use_cache=use_result_cache, result_cache=self.result_cache, **generation_kwargs)

    def get_result_cache(self) -> ResultCache:
        return self.result_cache


# -----------------------------
# Example usage (for testing)
# -----------------------------
# if __name__ == "__main__":
    # # NOTE: in real usage, use local cached models or HF tokens if private models.
    # manager = ModelManager(max_models_loaded=2, result_cache_size=512, result_ttl=3600)

    # # load small model for quick testing (use a small HF model name you have)
    # model_id = "gpt2"  # replace with a model you have/allow
    # try:
    #     # load model (first time)
    #     m = manager.load_model(model_id, model_type="causal", use_quantization=False, cache_dir=None)

    #     # single generation
    #     out = manager.generate(model_id, "Hello, how are you?", max_new_tokens=40)
    #     print("OUT:", out)

    #     # batch generation
    #     prompts = ["Artificial intelligence is", "Programming means", "The future will be"]
    #     outs = manager.generate(model_id, prompts, max_new_tokens=20, num_return_sequences=1)
    #     print("BATCH OUTS:", outs)

    #     # list loaded
    #     print("Loaded models:", manager.list_loaded())

    #     # use result cache: same prompt should be cached
    #     out2 = manager.generate(model_id, "Hello, how are you?", max_new_tokens=40)
    #     print("OUT (cached):", out2)

    #     # unload
    #     manager.unload_model(model_id)
    #     print("After unload, loaded:", manager.list_loaded())

    # except Exception as e:
    #     log_error(f"Example run failed: {e}")