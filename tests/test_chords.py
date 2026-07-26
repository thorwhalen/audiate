"""Tests for chord-chart ingest (detection + delegation to accompy).

Routing is exercised with accompy mocked, so these run with or without accompy
installed. A final integration test really arranges a chart, gated on accompy
being importable.
"""

import pretty_midi
import pytest

from audiate import ingest
from audiate.ingest import (
    _looks_like_chord_input,
    _normalize_chord_source,
    detect_format,
    to_pretty_midi,
)


# --- detection (pure, no accompy needed) ---


@pytest.mark.parametrize(
    "s",
    [
        "| Dm7 | G7 | Cmaj7 | A7 |",
        "| C | Am | F | G |",
        "| C / / / | Am / / / |",  # slash beat markers (finding #7)
        "irealb://1r34LbKcu7...",
        "irealbook://x",
        "[C]twinkle [G]twinkle [C]little star",
        "Cmaj7 | Fmaj7 | Bb7 | Ebmaj7",
    ],
)
def test_detects_chord_charts(s):
    assert _looks_like_chord_input(s) is True
    assert detect_format(s) == "chords"


@pytest.mark.parametrize(
    "s",
    [
        "song.mid",
        "score.musicxml",
        "tune.abc",
        "C G Am F",  # bare tokens -> ambiguous with tinyNotation, left alone
        "X:1\nT:Cooley's\nK:Edor\n|:D2|EBBA B2 EB|",  # ABC string (headers + note runs)
        # ABC with chord-shaped cells but a header -> the _ABC_HEADER_RE guard is
        # the ONLY thing rejecting it (finding #13, mutation-guard).
        "X:1\nK:C\n| C | Am | F | G |",
        "[see figure 1] not a chord chart at all",  # loose ChordPro gate (finding #8)
        "Just some prose | with a pipe in it that is not music",
        "",
    ],
)
def test_rejects_non_chord_input(s):
    assert _looks_like_chord_input(s) is False


# --- normalization for accompy (pure; guards findings #1 + #7) ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        # ChordPro -> bar chart of chord symbols (lyrics dropped) -- finding #1
        ("[C]Twinkle twinkle little [G]star", "| C | G |"),
        # slash beat markers stripped, measures preserved -- finding #7
        ("| C / / / | Am / / / |", "| C | Am |"),
        # multi-chord bars preserved
        ("| Dm7 G7 | Cmaj7 |", "| Dm7 G7 | Cmaj7 |"),
        # iReal URL passes through untouched (accompy parses it natively)
        ("irealb://xyz", "irealb://xyz"),
    ],
)
def test_normalize_chord_source(raw, expected):
    assert _normalize_chord_source(raw) == expected


def test_existing_file_is_not_chords(tmp_path):
    f = tmp_path / "weird|name"
    f.write_text("x")
    assert _looks_like_chord_input(str(f)) is False


# --- routing (accompy mocked) ---


def test_routes_chord_input_to_accompy(monkeypatch):
    captured = {}
    marker = pretty_midi.PrettyMIDI()

    def fake_accompy_to_pm(source, **opts):
        captured["source"] = source
        captured["opts"] = opts
        return marker

    monkeypatch.setattr(ingest, "_accompy_available", lambda: True)
    monkeypatch.setattr(ingest, "_accompy_to_pm", fake_accompy_to_pm)

    result = to_pretty_midi("| Dm7 | G7 |", style="bossa", tempo=140)
    assert result is marker
    assert captured["source"] == "| Dm7 | G7 |"
    assert captured["opts"] == {"style": "bossa", "tempo": 140}


def test_chord_input_without_accompy_raises(monkeypatch):
    monkeypatch.setattr(ingest, "_accompy_available", lambda: False)
    with pytest.raises(ImportError) as e:
        to_pretty_midi("| Dm7 | G7 |")
    assert "audiate[chords]" in str(e.value)


def test_non_chord_string_not_routed_to_accompy(monkeypatch):
    # a bare-token string must NOT hit accompy; it goes to the music21 path
    called = {"n": 0}
    monkeypatch.setattr(ingest, "_accompy_available", lambda: True)
    monkeypatch.setattr(
        ingest, "_accompy_to_pm", lambda *a, **k: called.__setitem__("n", 1)
    )
    monkeypatch.setattr(
        ingest, "_music21_parse_to_pm", lambda src: pretty_midi.PrettyMIDI()
    )
    to_pretty_midi("C G Am F")
    assert called["n"] == 0


# --- real integration (only when accompy is installed) ---


def test_arrange_real_chart():
    pytest.importorskip("accompy")
    pm = to_pretty_midi("| Dm7 | G7 | Cmaj7 | A7 |", tempo=120)
    notes = [n for inst in pm.instruments for n in inst.notes]
    assert notes, "accompy arrangement produced no notes"
