from fastapi import APIRouter
from core.database import get_checkpointer

router = APIRouter()


@router.get("/test-checkpoint")
def test_checkpoint():
    try:
        checkpointer = get_checkpointer()

        config = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "",
                "checkpoint_id": None
            }
        }

        updated_config = checkpointer.put(config, {
            "v": 1,
            "ts": "2024-12-11T00:00:00Z",
            "id": "test-2",
            "channel_values": {"test": "works!", "timestamp": "just now"},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": []
        }, {
                                              "source": "update",
                                              "step": 1,
                                              "writes": None,
                                              "parents": {}
                                          }, {})

        print(f"Put returned: {updated_config}")

        result = checkpointer.get(updated_config)
        print(f"Get returned: {result}")

        checkpoints = list(checkpointer.list(config))
        print(f"List found: {len(checkpoints)} checkpoints")
        if checkpoints:
            print("there are checkpoints")
            for i in checkpoints:
                print(i)

        return {
            "status": "success",
            "put_result": str(updated_config),
            "get_result": str(result),
            "list_count": len(checkpoints)
        }
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}
