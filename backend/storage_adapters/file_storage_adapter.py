import os
from typing import Optional

from storage_adapters.storage_adapter import StorageAdapter


class FileStorageAdapter(StorageAdapter):
    def __init__(self, storage_dir: str = "./chat_histories"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save(self, session_id: str, data: str):
        filepath = os.path.join(self.storage_dir, f"{session_id}.json")
        with open(filepath, 'w') as f:
            f.write(data)

    def load(self, session_id: str) -> Optional[str]:
        filepath = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return f.read()
        return None
