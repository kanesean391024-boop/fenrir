import os
from unittest import mock
from conductor import post_to_discord_via_webhook


def test_discord_noop(monkeypatch):
    if "DISCORD_WEBHOOK_URL" in os.environ:
        del os.environ["DISCORD_WEBHOOK_URL"]
    called = False

    def fake_post(*a, **k):
        nonlocal called
        called = True

    monkeypatch.setattr("requests.post", fake_post)
    post_to_discord_via_webhook("hi")
    assert called is False
