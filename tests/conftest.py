import pytest, fitz, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
LEGACY = REPO / "example_pdfs" / "Marshall1929_AnnMagNatHist.pdf"


def pytest_configure(config):
    config.addinivalue_line("markers", "ollama: needs a running Ollama server")


@pytest.fixture(scope="session")
def legacy_pdf():
    assert LEGACY.exists(), f"missing {LEGACY}"
    return LEGACY


@pytest.fixture(scope="session")
def legacy_doc(legacy_pdf):
    return fitz.open(legacy_pdf)
