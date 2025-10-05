import conductor


def test_trim_budget(monkeypatch):
    monkeypatch.setattr(conductor, "tokenize_text", lambda m, t: t.split())
    msg = "\n".join(f"line{i}" for i in range(50))
    trimmed = conductor.trim_message_for_budget("m", "sys", "pre", msg, "post", 20)
    lines = trimmed.splitlines()
    assert len(lines) >= 10
    assert lines[-1] == "line49"
    assert "line0" not in lines
