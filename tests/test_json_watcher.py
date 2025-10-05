from __future__ import annotations

import runtime_utils


def test_json_watcher_receives_agent(monkeypatch):
    received: dict | None = None

    def watcher(payload: dict) -> None:
        nonlocal received
        received = payload

    runtime_utils.JSON_WATCHERS.clear()
    runtime_utils.add_json_watcher(watcher)

    class FakeResp:
        status_code = 200
        text = "{}"

        def json(self) -> dict:
            return {"response": "ok"}

    def fake_post(url, json, timeout):
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)

    runtime_utils.generate_with_watchdog(
        {"model": "test", "prompt": "hi"}, agent_name="AgentX"
    )

    assert received is not None
    assert received.get("__agent") == "AgentX"
    assert received.get("prompt") == "hi"

    runtime_utils.JSON_WATCHERS.clear()
