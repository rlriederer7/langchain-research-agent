from abc import ABC, abstractmethod
from typing import Optional


class StorageAdapter(ABC):
    @abstractmethod
    def save(self, session_id: str, data: str):
        pass

    @abstractmethod
    def load(self, session_id: str) -> Optional[str]:
        pass
