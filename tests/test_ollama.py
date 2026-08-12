import pytest
from workshop_lib import CHAT_MODEL, OCR_MODEL, server_ready, ocr_page, open_pdf


def test_model_names():
    assert CHAT_MODEL == "qwen3.5:9b"
    assert OCR_MODEL == "deepseek-ocr"


@pytest.mark.ollama
def test_server_ready():
    assert server_ready(timeout=5) is True


@pytest.mark.ollama
def test_ocr_recovers_what_the_text_layer_destroyed(legacy_pdf):
    """The point of the whole OCR section: the image beats the text layer."""
    doc = open_pdf(legacy_pdf)
    out = ocr_page(doc, 5)
    assert "Curculionid" in out       # correct spelling recovered
    assert "Cureulionidse" not in out  # the corrupt form is gone
    assert len(out) > 1500             # a full page, not a summary
