import threading
from typing import Any, Dict, Type


class SingletonMeta(type):
    """
    Thread-safe Singleton Metaclass with double-checked locking.
    """

    _instances: Dict[Type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # First check (no lock)
        if cls not in cls._instances:
            with cls._lock:
                # Second check (with lock)
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]