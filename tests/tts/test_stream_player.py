import sys
import types
import numpy as np
import pytest
from openvox.tts.stream import _StreamPlayer


class _FakeStream:
    def __init__(self, **kw):
        self.writes = []
        self.started = False
        self.aborted = False
        self.closed = False
        self.stopped = False

    def start(self):
        self.started = True

    def write(self, a):
        self.writes.append(np.asarray(a).copy())

    def abort(self):
        self.aborted = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _RaisingStream(_FakeStream):
    def write(self, a):
        raise RuntimeError("device fault")


def _install_fake_sd(monkeypatch, holder, stream_cls=_FakeStream):
    fake = types.ModuleType("sounddevice")

    def OutputStream(**kw):
        s = stream_cls(**kw)
        holder.append(s)
        return s

    fake.OutputStream = OutputStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake)


def test_player_drains_and_closes(monkeypatch):
    holder = []
    _install_fake_sd(monkeypatch, holder)
    p = _StreamPlayer(sample_rate=24000, queue_size=8)
    p.start()
    p.put(np.zeros(4, dtype="float32"), 24000)
    p.put(np.ones(4, dtype="float32"), 24000)
    p.finish()
    p.wait_drain(timeout=5)
    s = holder[0]
    assert len(s.writes) == 2
    assert s.closed is True


def test_player_abort_closes_stream_and_stops_consumer(monkeypatch):
    holder = []
    _install_fake_sd(monkeypatch, holder)
    p = _StreamPlayer(sample_rate=24000, queue_size=8)
    p.start()
    p.put(np.zeros(4, dtype="float32"), 24000)
    p.abort()
    s = holder[0]
    assert s.aborted is True
    assert s.closed is True
    assert p._consumer.is_alive() is False


def test_player_write_failure_surfaces_from_wait_drain(monkeypatch):
    holder = []
    _install_fake_sd(monkeypatch, holder, stream_cls=_RaisingStream)
    p = _StreamPlayer(sample_rate=24000, queue_size=8)
    p.start()
    p.put(np.zeros(4, dtype="float32"), 24000)
    p.put(np.ones(4, dtype="float32"), 24000)
    p.finish()
    with pytest.raises(RuntimeError, match="device fault"):
        p.wait_drain(timeout=5)
    assert p._consumer.is_alive() is False
