from workshop_lib import looks_corrupt, corruption_report, open_pdf, get_page_text

CLEAN = (
    "Legs approximately equal in length; femora slender, sub-linear and not "
    "toothed, the hind pair not exceeding the apex of the elytra; tibiae "
    "compressed, curved at the base, sub-carinate dorsally."
)
DIRTY = (
    "new South American Cureulionidse. 267 Legs approximately equal in length; "
    "femora slender, suh-linear and not toothed; Elylra oblong; Fi.q. 20; 2~5"
)


def test_clean_text_passes():
    assert looks_corrupt(CLEAN) is False


def test_dirty_text_is_flagged():
    assert looks_corrupt(DIRTY) is True


def test_report_names_the_suspects():
    suspects = " ".join(corruption_report(DIRTY)["suspect_words"])
    assert "2~5" in suspects
    assert "Fi.q." in suspects or "Fi.q" in suspects


def test_report_counts_words():
    r = corruption_report(CLEAN)
    assert r["n_words"] > 20
    assert r["ratio"] == 0.0


def test_empty_text_is_corrupt():
    # a scan with no text layer at all must route to OCR
    assert looks_corrupt("") is True
    assert looks_corrupt("   \n  ") is True


def test_threshold_is_adjustable():
    # one soft-pattern bad token in a long clean passage passes at a loose
    # threshold. Uses "Fi.q." rather than a tilde/ampersand token: those are
    # high-confidence markers (see test_gate_matches_known_pages_across_both_pdfs)
    # and are corrupt regardless of threshold by design, so they can't be
    # used to demonstrate the threshold parameter.
    text = CLEAN + " Fi.q."
    assert looks_corrupt(text, threshold=0.5) is False
    assert looks_corrupt(text, threshold=0.0) is True


def test_real_corrupt_page_is_flagged(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    assert looks_corrupt(get_page_text(doc, 5)) is True


def test_gate_matches_known_pages_across_both_pdfs(legacy_pdf, modern_pdf):
    """Regression test over the full corpus, not just legacy page 5.

    Checking a single page is exactly how a later change to the patterns
    could silently start missing other corrupt pages, or start flagging
    clean ones, without any test noticing -- which is what happened the
    first time this gate was built: it passed on page 5 while quietly
    calling three other genuinely corrupt pages of the same document
    clean, and flagging the (clean, born-digital) publisher cover page
    corrupt just because it contains URLs. This walks every page of both
    example PDFs against the known-correct answer.
    """
    legacy = open_pdf(legacy_pdf)
    assert legacy.page_count == 8, "expected page count changed; update this test"
    expected_legacy = {
        1: False,  # born-digital publisher cover page: clean despite the URLs/DOI
        2: True,
        3: True,
        4: True,
        5: True,
        6: True,
        7: True,  # "Cureulioni&e." -- the same mangled family name as page 5
        8: True,
    }
    for page, expected in expected_legacy.items():
        text = get_page_text(legacy, page)
        actual = looks_corrupt(text)
        assert actual is expected, (
            f"legacy page {page}: expected looks_corrupt={expected}, got "
            f"{actual}; report={corruption_report(text)}"
        )

    modern = open_pdf(modern_pdf)
    assert modern.page_count == 7, "expected page count changed; update this test"
    for page in range(1, modern.page_count + 1):
        text = get_page_text(modern, page)
        actual = looks_corrupt(text)
        assert actual is False, (
            f"modern page {page}: expected looks_corrupt=False (clean, "
            f"born-digital PDF), got True; report={corruption_report(text)}"
        )
