import base64
import pytest
from workshop_lib import open_pdf, get_page_text, render_page, DEFAULT_DPI


def test_default_dpi_is_100():
    assert DEFAULT_DPI == 100


def test_page_numbers_are_one_based(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    # PDF page 2 is printed page 264, the first content page
    assert "264" in get_page_text(doc, 2)[:40]


def test_page_five_is_the_corrupt_description(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    text = get_page_text(doc, 5)
    assert "Cureulionidse" in text      # the corrupt running header
    assert "Curculionidae" not in text  # the correct spelling is absent


def test_render_page_returns_base64_png(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    b64 = render_page(doc, 5, dpi=DEFAULT_DPI)
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_dpi_changes_size(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    assert len(render_page(doc, 5, dpi=150)) > len(render_page(doc, 5, dpi=72))


def test_out_of_range_page_raises(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    with pytest.raises(ValueError):
        get_page_text(doc, 99)
    with pytest.raises(ValueError):
        get_page_text(doc, 0)
