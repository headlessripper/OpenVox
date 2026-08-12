import subprocess
import sys


def test_import_openvox_agent_torch_free():
    code = (
        "import sys\n"
        "for m in ('torch','chatterbox'):\n"
        "    sys.modules[m]=None\n"
        "import openvox.agent\n"
        "from openvox.agent import VoiceAgent, listen_once, speak_stream, Turn\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
