"""Symbolic-score ingest: normalize many formats to a ``pretty_midi.PrettyMIDI``.

MIDI is the universal pivot -- every supported symbolic format is read and
converted to a ``PrettyMIDI`` object, which the synthesis engines consume. This
gives N-ingesters x M-engines coverage from a small amount of code.

- MIDI (``.mid`` / ``.midi`` / ``MThd`` bytes) is read directly by ``pretty_midi``.
- MusicXML / ABC / Humdrum-``kern`` / MEI and other notation formats go through
  ``music21`` (the ``audiate[symbolic]`` extra), which is the breadth champion
  for symbolic parsing.
- **Chord charts** (an iReal ``irealb://`` URL, a bar-line chart ``| C | Am |``,
  or ChordPro ``[C]lyrics``) are delegated to ``accompy`` (the ``audiate[chords]``
  extra), which arranges them into a backing-track MIDI; only fires when
  ``accompy`` is importable and the input is *unambiguously* a chord chart.
- A ``pretty_midi.PrettyMIDI`` or a ``music21`` stream may also be passed directly.
"""

from __future__ import annotations

import importlib.util
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

_MIDI_EXTS = {".mid", ".midi"}
#: iReal Pro chord-chart URL schemes (unambiguous chord input).
_IREAL_PREFIXES = ("irealb://", "irealbook://")
#: A single chord symbol, e.g. C, F#m7, Bbmaj7, Dm7b5, G7/B, N.C.
_CHORD_RE = re.compile(
    r"^(?:N\.?C\.?|[A-G][#b]?"
    r"(?:m|min|maj|dim|aug|sus|add|M|Δ|ø|°)?\d{0,2}"
    r"(?:[#b]\d{1,2})*(?:sus\d)?(?:\([^)]*\))?(?:/[A-G][#b]?)?)$",
    re.IGNORECASE,
)
#: ABC/notation info-field lines (e.g. ``X:1``, ``K:C``) -- marks an ABC string.
_ABC_HEADER_RE = re.compile(r"(?mi)^[XKMLTQVWZ]:")
#: Beat / bar-repeat glyphs that appear in chord charts but are not chords.
_BEAT_GLYPHS = frozenset({"/", ".", "%", "x", "X", "-"})
_EXT_FORMAT = {
    ".mid": "midi",
    ".midi": "midi",
    ".xml": "musicxml",
    ".musicxml": "musicxml",
    ".mxl": "musicxml",
    ".abc": "abc",
    ".krn": "kern",
    ".kern": "kern",
    ".mei": "mei",
    ".rntxt": "romantext",
}


def _is_prettymidi(x: Any) -> bool:
    return (
        type(x).__module__.split(".")[0] == "pretty_midi"
        and type(x).__name__ == "PrettyMIDI"
    )


def _is_music21(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "music21"


def _accompy_available() -> bool:
    """True if ``accompy`` (chord-chart arranger) is importable."""
    return importlib.util.find_spec("accompy") is not None


def _looks_like_chord_input(source: Any) -> bool:
    """True only for *unambiguous* chord-chart input (routes to accompy).

    Fires on an iReal ``irealb://`` URL, ChordPro bracket chords (``[C]``), or a
    bar-line chart whose cells are chord-shaped (``| Dm7 | G7 |``). Deliberately
    does NOT fire on a real file path / known notation extension, an ABC string
    (which also uses ``|`` but carries ``X:``/``K:`` headers and note-run cells),
    nor bare space-separated tokens (``"C G Am F"``, ambiguous with tinyNotation).
    """
    if not isinstance(source, (str, os.PathLike)):
        return False
    s = str(source).strip()
    if not s:
        return False
    if s.startswith(_IREAL_PREFIXES):
        return True
    ext = os.path.splitext(s)[1].lower()
    if ext in _EXT_FORMAT or os.path.exists(s):
        return False
    # ChordPro: bracketed tokens that are mostly chord-shaped (not just any [X]).
    brackets = re.findall(r"\[([^\]]*)\]", s)
    if brackets:
        chordish = sum(bool(_CHORD_RE.match(b)) for b in brackets)
        if chordish and chordish / len(brackets) >= 0.6:
            return True
    # Bar-line chart: cells (minus beat/repeat glyphs) mostly chord-shaped.
    if "|" in s and not _ABC_HEADER_RE.search(s):
        cells = _chord_cells(s)
        if cells and sum(bool(_CHORD_RE.match(c)) for c in cells) / len(cells) >= 0.6:
            return True
    return False


def _chord_cells(s: str) -> list:
    """Chart tokens with bar lines / whitespace / beat glyphs removed."""
    return [c for c in re.split(r"[|\s]+", s) if c and c not in _BEAT_GLYPHS]


def _chordpro_symbols(s: str) -> list:
    """Chord symbols from a ChordPro string's ``[..]`` brackets (chord-shaped only)."""
    return [b for b in re.findall(r"\[([^\]]*)\]", s) if _CHORD_RE.match(b)]


def _normalize_chord_source(s: str) -> str:
    """Rewrite chord input into a bar-line chart accompy parses cleanly.

    ``accompy.generate_accompaniment`` has no ChordPro parser and treats ``/``
    beat glyphs as chord tokens, so this: passes iReal URLs through untouched,
    turns ChordPro (``[C]la [G]la``) into ``| C | G |``, and strips beat glyphs
    from bar-line charts (``| C / / / | Am / / / |`` -> ``| C | Am |``) while
    preserving multi-chord bars (``| Dm7 G7 | C |`` stays intact).
    """
    s = s.strip()
    if s.startswith(_IREAL_PREFIXES):
        return s
    syms = _chordpro_symbols(s)
    if syms:
        return "| " + " | ".join(syms) + " |"
    if "|" in s:
        bars = [
            " ".join(t for t in cell.split() if t not in _BEAT_GLYPHS)
            for cell in s.split("|")
        ]
        bars = [b for b in bars if b]
        if bars:
            return "| " + " | ".join(bars) + " |"
    return s


def detect_format(source: Any) -> str:
    """Best-effort detection of a score source's symbolic format.

    Returns one of ``'prettymidi'``, ``'music21'``, ``'midi'``, ``'musicxml'``,
    ``'abc'``, ``'kern'``, ``'mei'``, ``'romantext'``, ``'chords'``, or
    ``'unknown'``. Used to route ingest and to tag the resulting
    :class:`~audiate.base.AudioData`; not authoritative.
    """
    if _is_prettymidi(source):
        return "prettymidi"
    if _is_music21(source):
        return "music21"
    if isinstance(source, (bytes, bytearray)):
        return "midi" if bytes(source[:4]) == b"MThd" else "unknown"
    if isinstance(source, (str, Path)):
        ext = os.path.splitext(str(source))[1].lower()
        if ext in _EXT_FORMAT:
            return _EXT_FORMAT[ext]
        if _looks_like_chord_input(source):
            return "chords"
        return "unknown"
    return "unknown"


def to_pretty_midi(source: Any, **ingest_opts):
    """Normalize any supported score source to a ``pretty_midi.PrettyMIDI``.

    Args:
        source: A ``pretty_midi.PrettyMIDI`` (passthrough), a ``music21`` stream,
            raw MIDI ``bytes``, a filesystem path to a ``.mid``/MusicXML/ABC/
            ``kern``/MEI file (anything ``music21`` can parse), or a **chord
            chart** (iReal ``irealb://`` URL / ``| C | Am |`` / ChordPro) when
            ``accompy`` is installed.
        **ingest_opts: Passed to the ``accompy`` arranger for chord input
            (e.g. ``style``, ``tempo``, ``repeats``, ``backend``); ignored for
            other source kinds.

    Returns:
        A ``pretty_midi.PrettyMIDI`` object.

    Raises:
        TypeError: If ``source`` is not a supported score form.
        ImportError: If a non-MIDI format is given but ``music21`` is not installed.
    """
    import pretty_midi

    if _is_prettymidi(source):
        return source
    if _is_music21(source):
        return _music21_to_pm(source)
    if isinstance(source, (bytes, bytearray)):
        b = bytes(source)
        if b[:4] == b"MThd":
            return pretty_midi.PrettyMIDI(BytesIO(b))
        return _music21_parse_to_pm(b)
    if isinstance(source, (str, Path)):
        p = str(source)
        # Chord chart -> accompy arranger (before the notation fallback, which
        # would mis-parse a chord chart as tinyNotation and fail).
        if _looks_like_chord_input(p):
            if not _accompy_available():
                raise ImportError(
                    f"Input looks like a chord chart ({p[:40]!r}...), which needs "
                    "accompy: pip install 'audiate[chords]'."
                )
            return _accompy_to_pm(p, **ingest_opts)
        ext = os.path.splitext(p)[1].lower()
        if ext in _MIDI_EXTS:
            return pretty_midi.PrettyMIDI(p)
        # Everything else -> music21 (it sniffs/handles many formats by extension).
        return _music21_parse_to_pm(p)
    raise TypeError(
        f"Unsupported score source type {type(source).__name__!r}. Pass a "
        "pretty_midi.PrettyMIDI, a music21 stream, MIDI bytes, a file path, or "
        "a chord chart (needs accompy)."
    )


def _accompy_to_pm(
    source: Any,
    *,
    style: str = "swing",
    tempo: int = 120,
    repeats: int = 1,
    backend: str = "builtin",
    **_ignored,
):
    """Arrange a chord chart into a backing-track ``PrettyMIDI`` via ``accompy``.

    Uses ``accompy.generate_accompaniment(..., output_format="midi")`` (a full
    drums/bass/piano arrangement) and reads the resulting ``.mid`` into a
    ``PrettyMIDI``. ``backend="builtin"`` keeps it dependency-light (no external
    MMA CLI). audiate synthesizes the MIDI itself, so accompy's own synth setup
    check is silenced.
    """
    os.environ.setdefault("ACCOMPY_SKIP_SETUP_CHECK", "1")
    import accompy
    import pretty_midi

    chart = _normalize_chord_source(str(source))
    path = accompy.generate_accompaniment(
        chart,
        style=style,
        tempo=tempo,
        repeats=repeats,
        output_format="midi",
        backend=backend,
    )
    try:
        return pretty_midi.PrettyMIDI(str(path))
    finally:
        _safe_remove(str(path))


def _import_converter():
    try:
        from music21 import converter
    except ImportError as e:  # pragma: no cover - exercised only w/o music21
        raise ImportError(
            "Reading non-MIDI symbolic formats (MusicXML/ABC/kern/MEI) needs "
            "music21 -- install the extra: pip install 'audiate[symbolic]'."
        ) from e
    return converter


def _music21_parse_to_pm(source: Any):
    converter = _import_converter()
    if isinstance(source, (bytes, bytearray)):
        fd, tmp = tempfile.mkstemp()
        os.close(fd)
        with open(tmp, "wb") as f:
            f.write(bytes(source))
        try:
            score = converter.parse(tmp)
        finally:
            _safe_remove(tmp)
    else:
        score = converter.parse(source)
    return _music21_to_pm(score)


def _music21_to_pm(stream):
    import pretty_midi

    fd, tmp = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    try:
        stream.write("midi", fp=tmp)
        return pretty_midi.PrettyMIDI(tmp)
    finally:
        _safe_remove(tmp)


def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:  # pragma: no cover
        pass
