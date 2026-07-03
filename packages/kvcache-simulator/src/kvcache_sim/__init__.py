"""Local KV cache hit-rate simulator aligned with KVCache.AI web tools."""

from .calculator import BYTES_PER_GIB, calculate_cache_size, load_models_data
from .simulator import run_sweep

__version__ = "0.1.2"

__all__ = [
    "BYTES_PER_GIB",
    "__version__",
    "calculate_cache_size",
    "load_models_data",
    "run_sweep",
]
