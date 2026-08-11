import subprocess
import sys

from openvox.enroll.backend import EnrollBackend


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
