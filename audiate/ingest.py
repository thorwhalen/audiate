"""Symbolic-score ingest: normalize many formats to a ``pretty_midi.PrettyMIDI``.

MIDI is the universal pivot -- every supported symbolic format is read and
converted to a ``PrettyMIDI`` object, which the synthesis engines consume. This
gives N-ingesters x M-engines coverage from a small amount of code.

- MIDI (``.mid`` / ``.midi`` / ``MThd`` bytes) is read directly by ``pretty_midi``.
- MusicXML / ABC / Humdrum-``kern`` / MEI and other notation formats go through
  ``music21`` (the ``audiate[symbolic]`` extra), which is the breadth champion
  for symbolic parsing.
- A ``pretty_midi.PrettyMIDI`` or a ``music21`` stream may also be passed directly.
"""

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

_MIDI_EXTS = {".mid", ".midi"}
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


def detect_format(source: Any) -> str:
    """Best-effort detection of a score source's symbolic format.

    Returns one of ``'prettymidi'``, ``'music21'``, ``'midi'``, ``'musicxml'``,
    ``'abc'``, ``'kern'``, ``'mei'``, ``'romantext'``, or ``'unknown'``. Used to
    route ingest and to tag the resulting :class:`~audiate.base.AudioData`; not
    authoritative.
    """
    if _is_prettymidi(source):
        return "prettymidi"
    if _is_music21(source):
        return "music21"
    if isinstance(source, (bytes, bytearray)):
        return "midi" if bytes(source[:4]) == b"MThd" else "unknown"
    if isinstance(source, (str, Path)):
        ext = os.path.splitext(str(source))[1].lower()
        return _EXT_FORMAT.get(ext, "unknown")
    return "unknown"


def to_pretty_midi(source: Any):
    """Normalize any supported score source to a ``pretty_midi.PrettyMIDI``.

    Args:
        source: A ``pretty_midi.PrettyMIDI`` (passthrough), a ``music21`` stream,
            raw MIDI ``bytes``, or a filesystem path to a ``.mid``/MusicXML/ABC/
            ``kern``/MEI file (anything ``music21`` can parse).

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
        ext = os.path.splitext(p)[1].lower()
        if ext in _MIDI_EXTS:
            return pretty_midi.PrettyMIDI(p)
        # Everything else -> music21 (it sniffs/handles many formats by extension).
        return _music21_parse_to_pm(p)
    raise TypeError(
        f"Unsupported score source type {type(source).__name__!r}. Pass a "
        "pretty_midi.PrettyMIDI, a music21 stream, MIDI bytes, or a file path."
    )


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
