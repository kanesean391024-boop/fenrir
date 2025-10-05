import runtime_utils


def test_unicode_and_punctuation():
    text = "Hello 世界 😄!"
    tokens = runtime_utils.tokenize_text("m", text)
    assert "Hello" in tokens
    assert "世界" in tokens
    assert "😄" in tokens
    assert "!" in tokens


def test_fallback(monkeypatch):
    def boom(model, snippet):
        raise RuntimeError
    monkeypatch.setattr(runtime_utils, "_tokenize_cached", boom)
    tokens = runtime_utils.tokenize_text("m", "a b")
    assert tokens == ["a", "b"]


def test_caching(monkeypatch):
    text = "hello"
    runtime_utils.tokenize_text("m", text)
    class Boom:
        def findall(self, _):
            raise RuntimeError
    monkeypatch.setattr(runtime_utils, "_WORDLIKE_RE", Boom())
    assert runtime_utils.tokenize_text("m", text) == ["hello"]
