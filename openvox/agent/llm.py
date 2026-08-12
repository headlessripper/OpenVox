import json
import urllib.request

def _messages(history, user_text):
    msgs = []
    for turn in history:
        role = "assistant" if turn.role == "assistant" else "user"
        msgs.append({"role": role, "content": turn.text})
    msgs.append({"role": "user", "content": user_text})
    return msgs

def _post(url, payload, headers, who):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    try:
        return urllib.request.urlopen(req, timeout=120)
    except OSError as exc:
        raise RuntimeError(f"could not reach {who} at {url}: {exc}. Is the local server running?") from exc

def ollama(model: str, host: str = "http://localhost:11434"):
    url = host.rstrip("/") + "/api/chat"
    def respond(user_text, history):
        payload = {"model": model, "stream": True, "messages": _messages(history, user_text)}
        with _post(url, payload, {}, "Ollama") as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                obj = json.loads(line)
                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    yield piece
    return respond

def openai_compatible(base_url: str, model: str = "local", api_key: str | None = None):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    def respond(user_text, history):
        payload = {"model": model, "stream": True, "messages": _messages(history, user_text)}
        with _post(url, payload, headers, "the OpenAI-compatible server") as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content", "")
                if piece:
                    yield piece
    return respond
