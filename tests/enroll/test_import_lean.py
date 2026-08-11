import subprocess
import sys

def test_import_openvox_enroll_without_heavy_deps():
    code = (
        "import sys\n"
        "for m in ('torch', 'chatterbox', 'librosa', 'soundfile', 'resemble_enhance'):\n"
        "    sys.modules[m] = None\n"
        "import openvox\n"
        "import openvox.enroll\n"
        "from openvox.enroll import VoiceEnrollEngine, VoiceProfile\n"
        "VoiceEnrollEngine(device='cpu')\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
