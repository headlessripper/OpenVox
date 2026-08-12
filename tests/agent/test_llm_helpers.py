import io, json
import pytest
from openvox.agent.llm import ollama, openai_compatible
from openvox.agent.turn import Turn

class _FakeResp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False

def test_ollama_streams_content(monkeypatch):
    lines = [json.dumps({"message": {"content": c}}) for c in ["Hel", "lo", "!"]]
    body = ("\n".join(lines) + "\n").encode()
    def fake_urlopen(req, timeout=None):
        assert req.full_url.endswith("/api/chat")
        payload = json.loads(req.data)
        assert payload["model"] == "m" and payload["stream"] is True
        assert payload["messages"][-1] == {"role": "user", "content": "hi"}
        return _FakeResp(body)
    monkeypatch.setattr("openvox.agent.llm.urllib.request.urlopen", fake_urlopen)
    respond = ollama(model="m")
    assert "".join(respond("hi", [Turn("user", "prev"), Turn("assistant", "yo")])) == "Hello!"

def test_openai_compatible_streams_sse(monkeypatch):
    chunks = ["Hi", " there"]
    sse = "".join("data: " + json.dumps({"choices":[{"delta":{"content":c}}]}) + "\n\n" for c in chunks)
    sse += "data: [DONE]\n\n"
    def fake_urlopen(req, timeout=None):
        assert req.full_url.endswith("/chat/completions")
        return _FakeResp(sse.encode())
    monkeypatch.setattr("openvox.agent.llm.urllib.request.urlopen", fake_urlopen)
    respond = openai_compatible(base_url="http://localhost:8080/v1")
    assert "".join(respond("hi", [])) == "Hi there"

def test_unreachable_endpoint_raises(monkeypatch):
    def boom(req, timeout=None): raise OSError("connection refused")
    monkeypatch.setattr("openvox.agent.llm.urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError, match="reach"):
        list(ollama(model="m")("hi", []))
