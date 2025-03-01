import ipdb
import hashlib
import lmdb
from typing import Dict, List, Optional
import json
import numpy as np
import io

def save_to_cache(key: str, value: str, env: lmdb.Environment):
    with env.begin(write=True) as txn:
        hashed_key = hash_key(key)
        txn.put(hashed_key.encode(), value.encode())

def get_from_cache(key: str, env: lmdb.Environment) -> Optional[str]:
    with env.begin(write=False) as txn:
        hashed_key = hash_key(key)
        value = txn.get(hashed_key.encode())
    if value:
        return value.decode()
    return None

def save_to_cache_np(key: str, numpy_array: np.ndarray, env: lmdb.Environment):
    """special case of save_to_cache for numpy arrays"""
    hashed_key = hash_key(key)

    buffer = io.BytesIO()
    np.save(buffer, numpy_array)
    value_bytes = buffer.getvalue()

    with env.begin(write=True) as txn:
        txn.put(hashed_key.encode(), value_bytes)

def get_from_cache_np(key, env):
    hashed_key = hash_key(key)
    with env.begin() as txn:
        value_bytes = txn.get(hashed_key.encode())
    
    if value_bytes:
        buffer = io.BytesIO(value_bytes)
        buffer.seek(0)
        numpy_array = np.load(buffer)
        return numpy_array

    else:
        return None

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def hash_key_32(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest()

def hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()

