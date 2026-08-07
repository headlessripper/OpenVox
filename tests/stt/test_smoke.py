def test_package_imports():
    import openvox
    assert openvox.__version__ == "0.1.0"
    from openvox.stt import STTEngine
    assert STTEngine is not None
