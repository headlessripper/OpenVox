import queue
import threading

class SpeechHandle:
    """Controls a streaming synthesis+playback session. stop() is the barge-in
    primitive; safe to call from another thread."""

    def __init__(self, player) -> None:
        self._player = player
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread = None
        self._error = None

    def start(self, backend, voice_id: str, speed: float, segments: list[str]) -> None:
        self._player.start()
        self._thread = threading.Thread(
            target=self._produce, args=(backend, voice_id, speed, segments),
            daemon=True)
        self._thread.start()

    def _produce(self, backend, voice_id, speed, segments) -> None:
        try:
            for chunk in backend.stream_segments(segments, voice_id, speed):
                if self._stop.is_set():
                    return
                self._player.put(chunk.audio, chunk.sample_rate)
                if self._stop.is_set():
                    return
            if not self._stop.is_set():
                self._player.finish()
        except Exception as exc:  # surfaced from wait()
            self._error = exc
        finally:
            self._done.set()

    def stop(self) -> None:
        self._stop.set()
        self._player.abort()
        self._done.set()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
        if not self._stop.is_set():
            self._player.wait_drain(timeout)
        if self._error is not None:
            raise self._error

    @property
    def done(self) -> bool:
        return self._done.is_set()


class _StreamPlayer:
    """Real-time player: a consumer thread drains a queue to a sounddevice
    OutputStream. put() applies backpressure; abort() cuts audio immediately."""

    def __init__(self, sample_rate: int, queue_size: int) -> None:
        self._sr = sample_rate
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._abort = threading.Event()
        self._finished = threading.Event()
        self._stream = None
        self._consumer = None

    def start(self) -> None:
        import sounddevice as sd
        self._stream = sd.OutputStream(samplerate=self._sr, channels=1, dtype="float32")
        self._stream.start()
        self._consumer = threading.Thread(target=self._consume, daemon=True)
        self._consumer.start()

    def _consume(self) -> None:
        import numpy as np
        while not self._abort.is_set():
            try:
                audio = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._finished.is_set():
                    break
                continue
            if self._abort.is_set():
                break
            try:
                self._stream.write(np.ascontiguousarray(audio, dtype="float32"))
            except Exception:
                break

    def put(self, audio, sample_rate) -> None:
        # backpressure, but stay responsive to abort so a blocked producer frees
        while not self._abort.is_set():
            try:
                self._queue.put(audio, timeout=0.1)
                return
            except queue.Full:
                continue

    def finish(self) -> None:
        self._finished.set()

    def abort(self) -> None:
        self._abort.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception:
                pass
        if self._consumer is not None:
            self._consumer.join(timeout=1.0)
        self.close()

    def wait_drain(self, timeout: float | None = None) -> None:
        if self._consumer is not None:
            self._consumer.join(timeout)
        self.close()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
