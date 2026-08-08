from openvox.tts.kokoro_backend import _choose_provider

def test_cuda_selected_when_available():
    got = _choose_provider("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert got == "CUDAExecutionProvider"

def test_cuda_requested_but_unavailable_falls_to_cpu():
    # The real-world case: CPU-only onnxruntime advertises Azure + CPU only.
    got = _choose_provider("cuda", ["AzureExecutionProvider", "CPUExecutionProvider"])
    assert got == "CPUExecutionProvider"

def test_cpu_device_always_cpu():
    got = _choose_provider("cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert got == "CPUExecutionProvider"
