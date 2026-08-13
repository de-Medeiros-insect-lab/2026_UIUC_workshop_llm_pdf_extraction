import pathlib
import pymupdf as fitz
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
LEGACY = REPO / "example_pdfs" / "Marshall1929_AnnMagNatHist.pdf"
MODERN = REPO / "example_pdfs" / "deMedeiros2013Zootaxa.pdf"


def pytest_configure(config):
    config.addinivalue_line("markers", "ollama: needs a running Ollama server")


@pytest.fixture(scope="session")
def legacy_pdf():
    assert LEGACY.exists(), f"missing {LEGACY}"
    return LEGACY


@pytest.fixture(scope="session")
def legacy_doc(legacy_pdf):
    return fitz.open(legacy_pdf)


@pytest.fixture(scope="session")
def modern_pdf():
    assert MODERN.exists(), f"missing {MODERN}"
    return MODERN
