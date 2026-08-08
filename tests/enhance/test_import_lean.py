import subprocess
import sys

def test_import_openvox_enhance_without_heavy_deps():
    code = (
        "import sys\n"
        "for m in ('torch', 'resemble_enhance'):\n"
        "    sys.modules[m] = None\n"
        "import openvox.enhance\n"
        "from openvox.enhance import EnhanceEngine\n"
        "eng = EnhanceEngine(device='cpu')\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
