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
    # one bad token in a long clean passage passes at a loose threshold
    text = CLEAN + " 2~5"
    assert looks_corrupt(text, threshold=0.5) is False
    assert looks_corrupt(text, threshold=0.0) is True


def test_real_corrupt_page_is_flagged(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    assert looks_corrupt(get_page_text(doc, 5)) is True
