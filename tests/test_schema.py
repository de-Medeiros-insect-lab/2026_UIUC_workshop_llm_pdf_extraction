from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from workshop_lib import Trait, Species, Extraction, to_dataframe, extract


def test_trait_requires_the_source_sentence():
    with pytest.raises(ValidationError):
        Trait(anatomical_part="elytra", trait="length")


def test_valid_trait():
    t = Trait(anatomical_part="elytra", trait="length", value="2.1",
              units="mm", source_text="Elytra 2.1 mm long")
    assert t.units == "mm"


def test_absurd_measurement_is_rejected():
    """Syntactically valid, semantically impossible -- the real failure mode."""
    with pytest.raises(ValidationError):
        Trait(anatomical_part="body", trait="length", value="5000",
              units="mm", source_text="body 5000 mm long")


def test_non_numeric_value_allowed_when_unitless():
    t = Trait(anatomical_part="pronotum", trait="colour", value="dull black",
              units=None, source_text="Colour dull black")
    assert t.value == "dull black"


def test_species_name_must_look_binomial():
    with pytest.raises(ValidationError):
        Species(name="Huarucus", traits=[])
    ok = Species(name="Huarucus cacti", traits=[])
    assert ok.name.startswith("Huarucus")


def test_to_dataframe_flattens_one_row_per_trait():
    ex = Extraction(species=[
        Species(name="Huarucus cacti", traits=[
            Trait(anatomical_part="elytra", trait="length", value="2.1",
                  units="mm", source_text="a"),
            Trait(anatomical_part="rostrum", trait="shape", value="curved",
                  units=None, source_text="b"),
        ]),
    ])
    df = to_dataframe(ex)
    assert len(df) == 2
    assert list(df.columns[:2]) == ["species", "anatomical_part"]
    assert set(df["species"]) == {"Huarucus cacti"}


def test_to_dataframe_empty_extraction():
    ex = Extraction(species=[])
    df = to_dataframe(ex)
    assert len(df) == 0
    assert list(df.columns) == ["species", "anatomical_part", "trait", "value",
                                 "units", "source_text"]


def _fake_reply(json_content: str):
    """Helper to build a fake chat reply with JSON content."""
    return SimpleNamespace(
        message=SimpleNamespace(content=json_content, tool_calls=None)
    )


def test_extract_passes_think_false():
    """Extraction is one-shot; think=False disables chain-of-thought."""
    seen = {}

    def fake(**kw):
        seen.update(kw)
        # Return valid Extraction JSON
        data = Extraction(species=[]).model_dump_json()
        return _fake_reply(data)

    extract("some text", chat=fake)
    assert seen["think"] is False


def test_extract_passes_format_schema():
    """Schema-constrained decoding enforces structured output."""
    seen = {}

    def fake(**kw):
        seen.update(kw)
        data = Extraction(species=[]).model_dump_json()
        return _fake_reply(data)

    extract("some text", chat=fake)
    assert "format" in seen
    assert seen["format"] == Extraction.model_json_schema()


def test_extract_returns_validated_extraction():
    """extract() parses and validates the chat model's JSON response."""
    def fake(**kw):
        # Return valid JSON with one species and one trait
        data = Extraction(species=[
            Species(name="Huarucus cacti", traits=[
                Trait(anatomical_part="elytra", trait="length", value="2.5",
                      units="mm", source_text="Elytra 2.5 mm long"),
            ]),
        ]).model_dump_json()
        return _fake_reply(data)

    result = extract("some text", chat=fake)
    assert isinstance(result, Extraction)
    assert len(result.species) == 1
    assert result.species[0].name == "Huarucus cacti"
    assert len(result.species[0].traits) == 1
    assert result.species[0].traits[0].anatomical_part == "elytra"


def test_extract_uses_injected_chat():
    """Injected chat callable is actually used; real ollama never touched."""
    chat_was_called = []

    def fake(**kw):
        chat_was_called.append(True)
        data = Extraction(species=[]).model_dump_json()
        return _fake_reply(data)

    extract("some text", chat=fake)
    assert len(chat_was_called) == 1
