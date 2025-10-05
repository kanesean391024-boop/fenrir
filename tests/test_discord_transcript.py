import types
import sys

from conductor import _discord_transcript


def test_discord_transcript(monkeypatch):
    fake_msgs = [
        {"author": "b", "text": "hi", "timestamp": "2024-01-02"},
        {"author": "a", "text": "hey", "timestamp": "2024-01-01"},
    ]

    fake_mod = types.SimpleNamespace(fetch_recent_discord_messages=lambda n: fake_msgs)
    monkeypatch.setitem(sys.modules, "fenrir_ui", fake_mod)

    text = _discord_transcript(2)
    assert text.splitlines()[0].startswith("[2024-01-01] a:")
