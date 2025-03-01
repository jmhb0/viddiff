import logging
from typing import Optional
import redis
import numpy as np

def get_from_cache(key: str, redis_env: redis.Redis) -> Optional[str]:
    try:
        value = redis_env.get(key.encode())
        if value:
            # logging.warning(f"Cache Hit: {key} -> {value.decode()}")
            return value.decode()
    except Exception as e:
        logging.warning(f"Error getting from cache: {e}")
    return None


def save_to_cache(key: str, value: str, redis_env: redis.Redis):
    try:
        redis_env.set(key.encode(), value.encode())
        # logging.warning(f"Cache Saved: {key} -> {value}")
    except Exception as e:
        logging.warning(f"Error saving to cache: {e}")
    return None

def save_to_cache_np(key: str, numpy_array: np.ndarray, redis_env: redis.Redis):
    """Special case of save_to_cache for numpy arrays."""
    hashed_key = hash_key(key)

    buffer = io.BytesIO()
    np.save(buffer, numpy_array)
    value_bytes = buffer.getvalue()

    try:
        redis_env.set(hashed_key.encode(), value_bytes)
    except Exception as e:
        logging.warning(f"Error saving numpy array to cache: {e}")

def get_from_cache_np(key: str, redis_env: redis.Redis) -> Optional[np.ndarray]:
    """Retrieve a cached numpy array from Redis."""
    hashed_key = hash_key(key)

    try:
        value_bytes = redis_env.get(hashed_key.encode())
        if value_bytes:
            buffer = io.BytesIO(value_bytes)
            buffer.seek(0)
            numpy_array = np.load(buffer)
            return numpy_array
    except Exception as e:
        logging.warning(f"Error getting numpy array from cache: {e}")

    return None

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def hash_key_32(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest()

def hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()
