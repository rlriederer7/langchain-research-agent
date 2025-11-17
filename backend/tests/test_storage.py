import pytest
import json
import os
import tempfile
from storage_adapters.file_storage_adapter import FileStorageAdapter


class TestFileStorageAdapter:
    @pytest.fixture
    def temp_storage_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def storage(self, temp_storage_dir):
        return FileStorageAdapter(storage_dir=temp_storage_dir)

    def test_save_and_load(self, storage):
        session_id = "test_session"
        test_data = json.dumps([{"role": "user", "content": "hello"}])

        storage.save(session_id, test_data)

        loaded = storage.load(session_id)

        assert loaded == test_data

    def test_load_nonexistent_session(self, storage):
        result = storage.load("nonexistent_session_is8u4hjo89r")
        assert result is None

    def test_overwrite_session(self, storage):
        session_id = "test_session"

        storage.save(session_id, "first_data")
        storage.save(session_id, "second_data")

        loaded = storage.load(session_id)
        assert loaded == "second_data"

    def test_multiple_sessions(self, storage):
        storage.save("session_1", "data_1")
        storage.save("session_2", "data_2")

        assert storage.load("session_1") == "data_1"
        assert storage.load("session_2") == "data_2"
