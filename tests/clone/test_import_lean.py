import subprocess
import sys

def test_import_openvox_clone_without_heavy_deps():
    # Blocking torch/onnxruntime/kokoro_onnx makes `import <that>` raise ImportError.
    # importing openvox.clone (and constructing VoiceCloneEngine) must still succeed —
    # the heavy deps load only when clone() actually runs.
    code = (
        "import sys\n"
        "for m in ('torch', 'onnxruntime', 'kokoro_onnx'):\n"
        "    sys.modules[m] = None\n"
        "import openvox.clone\n"
        "from openvox.clone import VoiceCloneEngine\n"
        "eng = VoiceCloneEngine(device='cpu')\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
