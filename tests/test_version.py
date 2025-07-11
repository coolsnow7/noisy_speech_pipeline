import noisy_speech_pipeline as nsp


def test_version():
    """Project version is pinned in one place."""
    assert nsp.__version__ == "0.0.1"
