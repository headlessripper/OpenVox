import subprocess
import sys
import types

import numpy as np

from openvox.enroll.backend import EnrollBackend
from openvox.enroll.chatterbox_backend import ChatterboxEnrollBackend


def test_abc_cannot_instantiate():
    import pytest
    with pytest.raises(TypeError):
        EnrollBackend()


def test_backend_module_is_torch_free_at_import():
    # importing the ABC + concrete backend module must not require torch
    code = (
        "import sys\n"
        "for m in ('torch', 'chatterbox'):\n"
        "    sys.modules[m] = None\n"
        "import openvox.enroll.backend\n"
        "import openvox.enroll.chatterbox_backend\n"
        "from openvox.enroll.chatterbox_backend import ChatterboxEnrollBackend\n"
        "ChatterboxEnrollBackend(device='cpu')\n"   # construct without loading
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


class _FakeWav:
    def __init__(self, arr):
        self._a = arr

    def squeeze(self, d):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _FakeModel:
    def __init__(self):
        self.sr = 24000
        self.conds = None
        self.recorded_kwargs = {}

    def generate(self, text, **kwargs):
        self.recorded_kwargs = kwargs
        return _FakeWav(np.zeros((1, 4), dtype=np.float32))


def test_generate_forwards_baked_in_exaggeration():
    backend = ChatterboxEnrollBackend(device="cpu")
    fake_model = _FakeModel()
    backend._model = fake_model

    conditionals = types.SimpleNamespace(
        t3=types.SimpleNamespace(
            emotion_adv=np.array([[[0.7]]], dtype=np.float32)
        )
    )

    wav, sr = backend.generate(conditionals, "hi", 0)

    import pytest
    assert fake_model.recorded_kwargs.get("exaggeration") == pytest.approx(0.7, abs=1e-6)
    assert sr == 24000
    assert wav.shape == (1, 4)


def test_cuda_available_false_when_device_cpu():
    # An explicit device="cpu" must report cuda_available False regardless of hardware,
    # so the engine takes the Stage-A-only path.
    from openvox.enroll.chatterbox_backend import ChatterboxEnrollBackend
    assert ChatterboxEnrollBackend(device="cpu").cuda_available is False
