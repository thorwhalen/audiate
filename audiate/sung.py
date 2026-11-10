"""The sung line of a score: its vocal notes, timed in seconds, with lyric syllables.

A singing backend needs a different view of a score than a synthesizer: one
monophonic line of notes, each with its onset and duration in seconds, its MIDI
pitch, the syllable sung on it, and where the words start and end.
:func:`sung_line` reads that from a MIDI file or from anything ``music21`` reads
(MusicXML first of all)::

    >>> import audiate                                          # doctest: +SKIP
    >>> line = audiate.sung_line("song.musicxml")               # doctest: +SKIP
    >>> [(n.start, n.dur, n.pitch, n.syllable, n.word) for n in line][:2]  # doctest: +SKIP
    [(0.0, 0.5, 64, 'can', 'candle'), (0.5, 0.5, 67, 'dle', 'candle')]

Each :class:`SungNote` carries the whole ``word`` it belongs to and
``starts_word`` (False when it continues the word of the note before: a later
syllable, or a melisma, which has an empty ``syllable``). A note with no word
(``word == ''``) is a sung note the score gives no lyric for.

The times are on the same clock as :func:`audiate.render`: MIDI ticks and
music21 offsets go through the same ``pretty_midi`` tempo map the render uses,
so the sung line lines up with the rendered audio (for music21 scores, the
tempo marks of every part count, not only those of the sung part).

**Which line.** In a MIDI file, a line is one track and channel; drums (channel
10) are skipped. With several candidates, the one whose track carries the
lyrics is taken; otherwise pass ``track=`` (index in the file) and, for a file
that mixes channels in one track, ``channel=`` (1 to 16). In a music21 score, a
line is one part (``part=``: an index, or a part name or id), the one with
lyrics by default, and one voice of it (``voice=``), the one with lyrics by
default. Errors name the candidates.

A bar the MusicXML gives to another voice is silent for the chosen one.
Repeats are played as the render plays them (music21's ``expandRepeats``), and
on the n-th pass through a bar, verse n is sung when the bar has one.

**Polyphony is refused.** Two notes starting together (a chord, or a MIDI
dyad, within ``chord_window_s``) raise unless ``chords="top"`` or ``"bottom"``
picks one; a MIDI note overlapping the next by more than ``max_overlap_s``
raises (less is legato, and the note is cut at the next onset). Grace notes and
zero-length notes are skipped, a lyric on one going to the next note; tied
notes become one.

**Lyrics.** MusicXML lyrics are read from the chosen verse (``verse=``, the
lowest by default), syllables joined into words by ``<syllabic>`` (a
``middle`` or ``end`` joins only a word a ``begin`` opened); an elision (two
syllables on one note) keeps both, separated by a space. MIDI lyrics come from
``lyrics`` meta events only, on the line's track, else on the one track that
has lyrics and no notes (a conductor or words track). Each goes to its note by
time, never by counting, so a missing syllable cannot shift the ones after it:
the note starting within a 64th note of it, else the note sounding then, or
the next one when the lyric is early for it (see ``_midi_lyrics_on_notes``).
In a file that uses hyphens, hyphens alone join syllables (``can-`` ``dle``, or
``can`` ``-dle``) and spaces only end words; in a hyphen-free file that marks
word ends with spaces (``Hel`` ``lo `` ``world``), a syllable not preceded by a
space joins the one before. Karaoke files (``.kar``, or a ``@KMIDI KARAOKE FILE`` header) without
``lyrics`` events are read from their ``text`` events the same way, with the
``/`` (new line) and ``\\`` (new paragraph) marks starting a word and ``@``
header lines skipped; ``text`` events of any other file are never sung.

**Melismas.** A note without a syllable continues the word before it, unless a
rest came between; a word left open (``can-``, a ``begin`` syllable) or a
syllable with a MusicXML ``<extend/>`` line holds across rests too.

:func:`attach_lyrics` puts a lyric line written by hand (``"Light a can-dle"``)
on notes instead.

The field names and order of :class:`SungNote` (``start, dur, pitch, word,
starts_word``, then ``syllable``) match the note type of singing back ends
such as ``voxy.score.Note``, so ``Note(n.start, n.dur, n.pitch, n.word,
n.starts_word)`` converts one.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from audiate.ingest import (
    MIDI_EXTENSIONS,
    _is_music21,
    _is_prettymidi,
    detect_format,
    music21_to_pretty_midi,
    parse_music21,
    to_pretty_midi,
)

Chords = Literal["error", "top", "bottom"]

#: MIDI channel 10 (index 9): percussion, never a sung line.
DRUM_CHANNEL = 9
#: A MIDI note may overlap the next by this much (legato) before the line counts
#: as polyphonic.
DFLT_MAX_OVERLAP_S = 0.1
#: MIDI notes starting this close together (seconds) are one chord.
DFLT_CHORD_WINDOW_S = 0.03
#: A gap this long between notes is a rest: a note after it without a syllable
#: of its own starts no word instead of holding the previous one (melisma).
DFLT_WORD_BREAK_REST_S = 0.2
#: A MIDI lyric this many beats before a note's onset belongs to that note.
LYRIC_LEAD_BEATS = 1 / 16
#: A MIDI lyric inside a sounding note may be early for the next note by up to
#: this many beats.
LYRIC_EARLY_WINDOW_BEATS = 1 / 2
#: Lyric tokens that hold the previous syllable one more note (a melisma).
HOLD_TOKENS = frozenset({"-", "_", "~"})
#: A hyphen-free file marks word ends with spaces when at least this share of
#: the gaps between its lyric texts has a space (or a karaoke mark) on either
#: side; below it, a stray space is ignored and every text is a word.
SPACED_MIN_SHARE = 0.2
#: Karaoke line marks at the start of a text event: new line, new paragraph.
KARAOKE_MARKS = "/\\"
#: The ``text`` event that marks a Soft Karaoke file.
KARAOKE_HEADER = "@KMIDI"


@dataclass(frozen=True)
class SungNote:
    """One sung note, in absolute seconds.

    Args:
        start: Onset, in seconds from the start of the score.
        dur: Duration in seconds.
        pitch: MIDI note number (60 is middle C).
        word: The whole word sung on it ('' for a note without lyrics).
        starts_word: True if the word starts on this note; False if the note
            continues the previous note's word (a later syllable, or a melisma).
        syllable: The syllable sung on this note ('' for a melisma note, which
            holds the syllable before, and for a note without lyrics).
    """

    start: float
    dur: float
    pitch: int
    word: str = ""
    starts_word: bool = True
    syllable: str = ""

    @property
    def end(self) -> float:
        """Where the note stops (seconds)."""
        return self.start + self.dur


# --------------------------------------------------------------------------- facade


def sung_line(
    source: Any,
    *,
    part: int | str | None = None,
    voice: int | str | None = None,
    verse: int | None = None,
    track: int | None = None,
    channel: int | None = None,
    chords: Chords = "error",
    with_lyrics: bool = True,
    max_overlap_s: float = DFLT_MAX_OVERLAP_S,
    chord_window_s: float = DFLT_CHORD_WINDOW_S,
    word_break_rest_s: float = DFLT_WORD_BREAK_REST_S,
) -> list[SungNote]:
    """The sung line of a score: its notes in seconds, with their syllables and words.

    Args:
        source: A MIDI file (``.mid``, ``.midi``, ``.kar``) or its bytes; a
            MusicXML file (or anything ``music21`` parses: ABC, kern, MEI, ...)
            or its bytes; or a ``music21`` stream.
        part: The music21 part to read: an index, or a part name or id. Default:
            the only part, else the only part with lyrics.
        voice: The voice of that part (its MusicXML voice number). Default: the
            only voice, else the only one with lyrics, else the first.
        verse: The lyric verse (MusicXML lyric number) sung throughout. Default:
            verse n on the n-th pass through a repeat, else the lowest.
        track: The MIDI track (index in the file). Default: see the module doc.
        channel: The MIDI channel, 1 to 16, for a track that mixes channels.
        chords: Two notes at once: ``"error"`` (default), or sing the
            ``"top"`` or ``"bottom"`` one.
        with_lyrics: False reads the notes only (every ``word`` empty).
        max_overlap_s: MIDI legato tolerance: a note overlapping the next by
            more than this is polyphony.
        chord_window_s: MIDI notes starting this close together are one chord
            (a played, slightly spread chord).
        word_break_rest_s: A gap this long is a rest, which ends a melisma.

    Returns:
        The notes, in time order.

    Raises:
        ValueError: Several candidate lines, polyphony, a lyric on no note or
            two on one note, an unknown part, voice or verse; the message names
            the choices.
        NotImplementedError: A karaoke file whose text marks no word boundary.
    """
    if _is_prettymidi(source):
        raise TypeError(
            "A pretty_midi.PrettyMIDI has merged the file's tracks and lost which "
            "lyric goes with which line: pass the MIDI file's path or bytes"
        )
    if _is_midi(source):
        _refuse_options("a MIDI file", part=part, voice=voice, verse=verse)
        return sung_line_from_midi(
            source,
            track=track,
            channel=channel,
            chords=chords,
            with_lyrics=with_lyrics,
            max_overlap_s=max_overlap_s,
            chord_window_s=chord_window_s,
            word_break_rest_s=word_break_rest_s,
        )
    _refuse_options("a music21 score", track=track, channel=channel)
    return sung_line_from_music21(
        source,
        part=part,
        voice=voice,
        verse=verse,
        chords=chords,
        with_lyrics=with_lyrics,
        word_break_rest_s=word_break_rest_s,
    )


def _is_midi(source: Any) -> bool:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source[:4]) == b"MThd"
    if isinstance(source, (str, os.PathLike)) and not _is_music21(source):
        return Path(source).suffix.lower() in MIDI_EXTENSIONS
    return False


def _refuse_options(kind: str, **options) -> None:
    given = [name for name, value in options.items() if value is not None]
    if given:
        verb = "does" if len(given) == 1 else "do"
        raise ValueError(f"{' and '.join(given)} {verb} not apply to {kind}")


def _check_chords(chords: str) -> None:
    if chords not in ("error", "top", "bottom"):
        raise ValueError(f"chords must be 'error', 'top' or 'bottom', got {chords!r}")


# --------------------------------------------------------------------------- MIDI


def sung_line_from_midi(
    source: str | os.PathLike | bytes,
    *,
    track: int | None = None,
    channel: int | None = None,
    chords: Chords = "error",
    with_lyrics: bool = True,
    max_overlap_s: float = DFLT_MAX_OVERLAP_S,
    chord_window_s: float = DFLT_CHORD_WINDOW_S,
    word_break_rest_s: float = DFLT_WORD_BREAK_REST_S,
) -> list[SungNote]:
    """The sung line of a MIDI file (path or bytes); see :func:`sung_line`."""
    import mido

    _check_chords(chords)
    if channel is not None and not 1 <= channel <= 16:
        raise ValueError(f"channel is 1 to 16, got {channel}")
    data, name = _midi_data(source)
    mid = mido.MidiFile(file=BytesIO(data))
    lines, lyrics, texts = _scan_midi(mid)
    sung_texts = lyrics
    karaoke = False
    if not _has_words(lyrics) and _is_karaoke(name, texts):
        karaoke = True
        sung_texts = {
            t: [(tick, text) for tick, text in events if not text.startswith("@")]
            for t, events in texts.items()
        }
    (t, ch), raw = _pick_midi_line(
        lines, sung_texts, track=track, channel=channel, name=name
    )
    pm = to_pretty_midi(data)  # the clock audiate.render plays the file on

    def seconds(tick: int) -> float:
        return float(pm.tick_to_time(int(tick)))

    where = f"track {t} channel {ch + 1} of {name}"
    spans = _monophonic(
        raw,
        seconds,
        chords=chords,
        chord_window_s=chord_window_s,
        max_overlap_s=max_overlap_s,
        where=where,
    )
    notes = [
        SungNote(seconds(b), seconds(e) - seconds(b), pitch) for b, e, pitch in spans
    ]
    if not with_lyrics:
        return notes
    note_tracks = {key[0] for key in lines}
    events = _midi_lyric_events(sung_texts, t, note_tracks=note_tracks, name=name)
    events = sorted(events, key=lambda e: e[0])  # stable: same-tick order kept
    syllables = _midi_syllables([text for _, text in events], karaoke=karaoke)
    on_note = _midi_lyrics_on_notes(
        [(tick, s) for (tick, _), s in zip(events, syllables) if s is not None],
        [(b, e) for b, e, _ in spans],
        lead=round(mid.ticks_per_beat * LYRIC_LEAD_BEATS),
        window=round(mid.ticks_per_beat * LYRIC_EARLY_WINDOW_BEATS),
        seconds=seconds,
    )
    items = [
        (note, *on_note.get(i, (None, False, False))) for i, note in enumerate(notes)
    ]
    return _join_words(items, rest_s=word_break_rest_s)


def _midi_data(source) -> tuple[bytes, str]:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), "the MIDI bytes"
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"No score at {path}")
    return path.read_bytes(), str(path)


def _scan_midi(mid):
    """``(lines, lyrics, texts)`` of a ``mido.MidiFile``.

    ``lines`` maps ``(track, channel)`` to ``[(begin_tick, end_tick, pitch)]``;
    ``lyrics`` and ``texts`` map a track to its ``[(tick, text)]`` ``lyrics`` and
    ``text`` meta events, space-only ones included (they mark word ends). A note
    struck again before its note-off ends there.
    """
    lines: dict = {}
    lyrics: dict = {}
    texts: dict = {}
    for t, trk in enumerate(mid.tracks):
        tick, active = 0, {}
        for msg in trk:
            tick += msg.time
            if msg.type in ("lyrics", "text"):
                if msg.text:
                    target = lyrics if msg.type == "lyrics" else texts
                    target.setdefault(t, []).append((tick, msg.text))
                continue
            if msg.type not in ("note_on", "note_off"):
                continue
            key = (msg.channel, msg.note)
            begin = active.pop(key, None)
            if begin is not None and tick > begin:
                lines.setdefault((t, msg.channel), []).append((begin, tick, msg.note))
            if msg.type == "note_on" and msg.velocity > 0:
                active[key] = tick
    return lines, lyrics, texts


def _has_words(texts_by_track: dict, track: Optional[int] = None) -> bool:
    """True if the (one ``track`` of the) map has a text that is not only spaces."""
    tracks = texts_by_track if track is None else {track: texts_by_track.get(track)}
    return any(text.strip() for events in tracks.values() for _, text in events or ())


def _is_karaoke(name: str, texts: dict) -> bool:
    return name.lower().endswith(".kar") or any(
        text.startswith(KARAOKE_HEADER)
        for events in texts.values()
        for _, text in events
    )


def _pick_midi_line(lines, sung_texts, *, track, channel, name):
    """The one ``((track, channel), notes)`` that is the sung line, or an error
    naming the choices."""
    wanted = {
        key: notes
        for key, notes in lines.items()
        if (track is None or key[0] == track)
        and (key[1] == channel - 1 if channel is not None else key[1] != DRUM_CHANNEL)
    }
    if not wanted:
        found = ", ".join(f"track={t} channel={c + 1}" for t, c in sorted(lines))
        raise ValueError(
            f"No notes in {name} for track={track}, channel={channel} "
            f"(lines with notes: {found or 'none'})"
        )
    if len(wanted) == 1:
        return next(iter(wanted.items()))
    worded = [key for key in wanted if _has_words(sung_texts, key[0])]
    if len(worded) == 1:
        return worded[0], wanted[worded[0]]
    choices = ", ".join(
        f"track={t} channel={c + 1} ({len(n)} notes"
        f"{', lyrics on its track' if _has_words(sung_texts, t) else ''})"
        for (t, c), n in sorted(wanted.items())
    )
    raise ValueError(
        f"{name} has several lines; pick one (track=, channel=): {choices}"
    )


def _monophonic(raw, seconds, *, chords, chord_window_s, max_overlap_s, where):
    """``[(begin, end, pitch)]`` of a one-voice line.

    Notes starting within ``chord_window_s`` of each other are a chord, resolved
    by ``chords``; a doubled note (same pitch, same onset) is one note, as long
    as its longest copy. Legato overlaps are cut at the next onset; longer ones
    are polyphony.
    """
    groups: list[list] = []
    for begin, end, pitch in sorted(raw):
        if groups and seconds(begin) - seconds(groups[-1][0][0]) <= chord_window_s:
            groups[-1].append((begin, end, pitch))
        else:
            groups.append([(begin, end, pitch)])
    spans = []
    for group in groups:
        by_pitch: dict = {}
        for begin, end, pitch in group:
            first = by_pitch.get(pitch)
            by_pitch[pitch] = (
                (begin, end) if first is None else (first[0], max(first[1], end))
            )
        if len(by_pitch) > 1 and chords == "error":
            raise ValueError(
                f"{where} is polyphonic: {len(by_pitch)} notes start at "
                f"{seconds(group[0][0]):.2f} s. Pick a one-voice line, or pass "
                "chords='top' or 'bottom'"
            )
        pitch = min(by_pitch) if chords == "bottom" else max(by_pitch)
        spans.append([*by_pitch[pitch], pitch])
    for here, following in zip(spans, spans[1:]):
        if here[1] > following[0]:
            overlap = seconds(here[1]) - seconds(following[0])
            if overlap > max_overlap_s:
                raise ValueError(
                    f"{where} is polyphonic at {seconds(following[0]):.2f} s (a note "
                    f"still sounds {overlap:.2f} s into the next): pick a one-voice "
                    "line (track=, channel=) or raise max_overlap_s"
                )
            here[1] = following[0]
    return [tuple(s) for s in spans]


def _midi_lyric_events(
    sung_texts, track: int, *, note_tracks, name
) -> list[tuple[int, str]]:
    """The lyric events of the line's ``track``, else of the one track that has
    lyrics and no notes (a conductor or words track). Lyrics on a track with notes
    of its own belong to that track's line, never to another one."""
    if _has_words(sung_texts, track):
        return sung_texts[track]
    others = sorted(
        t for t in sung_texts if t not in note_tracks and _has_words(sung_texts, t)
    )
    if len(others) > 1:
        raise ValueError(
            f"{name}: the line (track {track}) has no lyrics of its own, and tracks "
            f"{others} without notes have some: which are sung is not known, pass "
            "with_lyrics=False"
        )
    return sung_texts[others[0]] if others else []


def _midi_syllables(
    texts: Sequence[str], *, karaoke: bool = False
) -> list[Optional[tuple[str, bool, bool]]]:
    """``(syllable, joins_the_previous_word, leaves_the_word_open)`` per MIDI lyric
    text, or None for a text that is no syllable (spaces only, a hold mark, a bare
    karaoke line mark).

    In a file that uses hyphens, hyphens alone join syllables (``can-`` ``dle``,
    ``can`` ``-dle``) and spaces only end words. In a hyphen-free file whose texts
    mark word ends with spaces, a syllable with no space before it (and none
    after the one before) joins the previous one. Karaoke text always follows
    the space convention, with ``/`` and ``\\`` starting a word.

    >>> _midi_syllables(["can-", "dle", "light"])
    [('can', False, True), ('dle', True, False), ('light', False, False)]
    >>> _midi_syllables(["Hel", "lo ", "world"])
    [('Hel', False, False), ('lo', True, False), ('world', False, False)]
    >>> [s[0:2] for s in _midi_syllables(["Hap-", "py ", "birth-", "day ", "to"])]
    [('Hap', False), ('py', True), ('birth', False), ('day', True), ('to', False)]
    >>> [s[0:2] for s in _midi_syllables(["\\\\Twin", "kle ", "/Lit", "tle"], karaoke=True)]
    [('Twin', False), ('kle', True), ('Lit', False), ('tle', True)]
    """

    def core_of(text):
        return text.strip().lstrip(KARAOKE_MARKS).strip()

    def starts_apart(text):
        return text[:1].isspace() or text.lstrip()[:1] in tuple(KARAOKE_MARKS)

    gaps = len(texts) - 1
    boundaries = sum(
        a[-1:].isspace() or not a.strip() or starts_apart(b)
        for a, b in zip(texts, texts[1:])
    )
    hyphenated = any(
        core_of(t) not in HOLD_TOKENS
        and (core_of(t).startswith("-") or core_of(t).endswith("-"))
        for t in texts
    )
    if karaoke:
        if gaps > 0 and not boundaries:
            raise NotImplementedError(
                "This karaoke file's text events mark no word boundary (no spaces, "
                "no '/' or '\\' marks), so its syllables cannot be joined into "
                "words. Pass with_lyrics=False and attach the words with "
                "attach_lyrics()."
            )
        spaced = True
    else:
        spaced = not hyphenated and gaps > 0 and boundaries / gaps >= SPACED_MIN_SHARE
    out: list = []
    word_open = False  # the previous syllable asked for the next one (hyphen)
    word_ended = True  # a space or a mark came since the previous syllable
    for text in texts:
        stripped = text.lstrip()
        mark = stripped[:1] != "" and stripped[:1] in KARAOKE_MARKS
        body = stripped.lstrip(KARAOKE_MARKS)
        lead_space = text[:1].isspace() or body[:1].isspace() or mark
        core = body.strip()
        if not core or core in HOLD_TOKENS:
            out.append(None)
            if mark or not core or text[-1:].isspace():
                word_ended = True
            continue
        syllable = core.strip("-").strip()
        joins = (
            core.startswith("-")
            or word_open
            or (spaced and not lead_space and not word_ended)
        )
        word_open = core.endswith("-")
        out.append((syllable, joins, word_open) if syllable else None)
        word_ended = text[-1:].isspace()
    return out


def _midi_lyrics_on_notes(events, spans, *, lead, window, seconds) -> dict:
    """``{note index: (syllable, joins, open)}``: each lyric on its note.

    A lyric belongs to the note starting at its tick or up to ``lead`` ticks
    later, or starting up to ``lead`` ticks before it. Otherwise it falls inside
    a sounding note and may be late for that note or early (up to ``window``
    ticks) for the next one: the next one when the sounding note has its own
    lyric, the sounding one when the next has its own, else the one it is
    clearly nearer to; too close to call is an error. A lyric on no note, or two
    on one, is an error rather than a word sung on the wrong note.
    """
    found: dict = {}

    def place(i, tick, syllable):
        if i in found:
            raise ValueError(
                f"Two lyrics ({found[i][0]!r}, {syllable[0]!r}) on the note at "
                f"{seconds(spans[i][0]):.2f} s: fix the file, or pass with_lyrics=False"
            )
        found[i] = syllable

    pending = []
    for tick, syllable in events:
        i = next(
            (k for k, (b, _) in enumerate(spans) if tick <= b <= tick + lead), None
        )
        if i is None:
            i = next(
                (
                    k
                    for k, (b, e) in enumerate(spans)
                    if b <= tick <= b + lead and tick < e
                ),
                None,
            )
        if i is None:
            pending.append((tick, syllable))
        else:
            place(i, tick, syllable)
    for tick, syllable in pending:
        sounding = next((k for k, (b, e) in enumerate(spans) if b <= tick < e), None)
        upcoming = next(
            (k for k, (b, _) in enumerate(spans) if tick < b <= tick + window), None
        )
        if sounding is None and upcoming is None:
            raise ValueError(
                f"The lyric {syllable[0]!r} at {seconds(tick):.2f} s falls on no note "
                "of the line: fix the file, or pass with_lyrics=False"
            )
        if upcoming is None or sounding is None:
            i = sounding if upcoming is None else upcoming
        elif (sounding in found) != (upcoming in found):
            i = upcoming if sounding in found else sounding
        else:
            late, early = tick - spans[sounding][0], spans[upcoming][0] - tick
            if late >= 2 * early:
                i = upcoming
            elif early >= 2 * late:
                i = sounding
            else:
                raise ValueError(
                    f"The lyric {syllable[0]!r} at {seconds(tick):.2f} s could be late "
                    f"for the note at {seconds(spans[sounding][0]):.2f} s or early for "
                    f"the one at {seconds(spans[upcoming][0]):.2f} s: fix the file, or "
                    "pass with_lyrics=False"
                )
        place(i, tick, syllable)
    return found


# --------------------------------------------------------------------------- music21


def sung_line_from_music21(
    source: Any,
    *,
    part: int | str | None = None,
    voice: int | str | None = None,
    verse: int | None = None,
    chords: Chords = "error",
    with_lyrics: bool = True,
    word_break_rest_s: float = DFLT_WORD_BREAK_REST_S,
) -> list[SungNote]:
    """The sung line of a MusicXML file (or anything ``music21`` parses, or a
    ``music21`` stream); see :func:`sung_line`."""
    from music21 import stream as m21stream

    _check_chords(chords)
    if (
        isinstance(source, (str, os.PathLike))
        and detect_format(source) not in ("unknown", "chords")
        and not Path(source).exists()
    ):
        raise FileNotFoundError(f"No score at {source}")
    score = parse_music21(source)
    if not _is_music21(source):
        _annotate_from_musicxml(score, _musicxml_root(source))
    if score[m21stream.Measure]:  # as music21's MIDI writer does: repeats played
        score = score.expandRepeats()
    parts = list(getattr(score, "parts", None) or []) or [score]
    chosen = _pick_part(parts, part)
    elements = _voice_elements(chosen.stripTies(), voice)
    verse_at = _verse_chooser(elements, verse) if with_lyrics else None
    pm = music21_to_pretty_midi(score)

    def seconds(offset: float) -> float:
        return float(pm.tick_to_time(int(round(offset * pm.resolution))))

    items: list = []
    open_word = False
    carried = None  # the lyric of a skipped (grace) note, sung on the next note
    previous_end = None
    for offset, element, bar in elements:
        if element is None or element.isRest:
            items.append(None)
            continue
        lyric = _verse_lyric(element, verse_at(bar)) if verse_at else None
        length = float(element.duration.quarterLength)
        pitches = [p.midi for p in getattr(element, "pitches", ())]
        if element.duration.isGrace or length <= 0 or not pitches:
            carried = _merge_lyrics(carried, lyric)
            continue
        lyric, carried = _merge_lyrics(carried, lyric), None
        if len(pitches) > 1 and chords == "error":
            raise ValueError(
                f"A chord in the sung line (measure {element.measureNumber}, "
                f"{seconds(offset):.2f} s): pass chords='top' or 'bottom', or "
                "voice= for a one-voice line"
            )
        pitch = min(pitches) if chords == "bottom" else max(pitches)
        if previous_end is not None and offset < previous_end - 1e-9:
            raise ValueError(
                f"Overlapping notes in the sung line at {seconds(offset):.2f} s "
                f"(measure {element.measureNumber}): pass voice= for one voice"
            )
        previous_end = offset + length
        start, end = seconds(offset), seconds(offset + length)
        note = SungNote(start, end - start, int(pitch))
        if lyric is None:
            items.append((note, None, False, False))
            continue
        text, first, last, extend = lyric
        joins = open_word and first in ("middle", "end")
        open_word = last in ("begin", "middle")
        items.append((note, text, joins, open_word or extend))
    return _join_words(items, rest_s=word_break_rest_s)


def _merge_lyrics(earlier, later):
    """One lyric from a skipped note's lyric and the next note's (either None)."""
    if earlier is None or later is None:
        return earlier or later
    glue = "" if earlier[2] in ("begin", "middle") else " "
    return (earlier[0] + glue + later[0], earlier[1], later[2], later[3])


def _musicxml_root(source):
    """The root element of a MusicXML document (path, bytes, text or ``.mxl``), or None."""
    import xml.etree.ElementTree as ET
    import zipfile

    data = None
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, (str, os.PathLike)):
        path = Path(source)
        try:
            if path.suffix.lower() in (".musicxml", ".xml", ".mxl") and path.exists():
                data = path.read_bytes()
        except OSError:  # notation text too long to be a path
            data = None
        if data is None and isinstance(source, str) and "<score-partwise" in source:
            data = source.encode()
    if not data:
        return None
    try:
        if data[:2] == b"PK":
            with zipfile.ZipFile(BytesIO(data)) as z:
                names = [
                    n
                    for n in z.namelist()
                    if n.lower().endswith((".xml", ".musicxml"))
                    and not n.startswith("META-INF")
                ]
                data = z.read(names[0]) if names else b""
        return ET.fromstring(data)
    except (ET.ParseError, zipfile.BadZipFile, KeyError):
        return None


def _annotate_from_musicxml(score, root) -> None:
    """Mark what music21 does not keep from the MusicXML: the voice number of each
    bar (``editorial['voices']`` on the measure) and the lyric numbers with an
    ``<extend/>`` line on each note (``editorial['extend']``)."""
    from music21 import stream as m21stream

    if root is None or root.tag != "score-partwise":
        return
    xml_parts = root.findall("part")
    parts = list(getattr(score, "parts", []) or [])
    if len(xml_parts) != len(parts):  # music21 keeps the parts in document order
        return
    for p, xml_part in zip(parts, xml_parts):
        measures = list(p.getElementsByClass(m21stream.Measure))
        if len(xml_part.findall("measure")) != len(measures):
            continue
        for m, xm in zip(measures, xml_part.findall("measure")):
            by_voice: dict = {}
            for xn in xm.findall("note"):
                v = (xn.findtext("voice") or "1").strip()
                extend = {
                    int(ly.get("number") or 1)
                    for ly in xn.findall("lyric")
                    if ly.find("extend") is not None
                    and ly.find("extend").get("type") != "stop"
                }
                if xn.find("chord") is not None and by_voice.get(v):
                    by_voice[v][-1] |= extend
                else:
                    by_voice.setdefault(v, []).append(extend)
            m.editorial["voices"] = sorted(by_voice)
            if m.voices:
                groups = {str(v.id): list(v.notesAndRests) for v in m.voices}
            elif len(by_voice) == 1:
                groups = {next(iter(by_voice)): list(m.notesAndRests)}
            else:
                groups = {}
            for vid, elements in groups.items():
                marks = by_voice.get(vid)
                if marks is not None and len(marks) == len(elements):
                    for element, extend in zip(elements, marks):
                        if extend:
                            element.editorial["extend"] = extend


def _part_label(i: int, p) -> str:
    return f"{i} {getattr(p, 'partName', None) or p.id!s}"


def _has_lyrics(stream) -> bool:
    return any(
        (ly.text or "").strip() for n in stream.recurse().notes for ly in n.lyrics
    )


def _pick_part(parts, part):
    """The part to sing: by index, name or id, else the only one (with lyrics)."""
    labels = ", ".join(
        _part_label(i, p) + (" (lyrics)" if _has_lyrics(p) else "")
        for i, p in enumerate(parts)
    )
    if part is None:
        if len(parts) == 1:
            return parts[0]
        worded = [p for p in parts if _has_lyrics(p)]
        if len(worded) == 1:
            return worded[0]
        raise ValueError(
            f"The score has {len(parts)} parts; pick one with part=: {labels}"
        )
    if isinstance(part, int):
        if not -len(parts) <= part < len(parts):
            raise ValueError(f"No part {part}; the parts are: {labels}")
        return parts[part]
    matches = [
        p
        for p in parts
        if str(part).casefold()
        in {str(getattr(p, "partName", None) or "").casefold(), str(p.id).casefold()}
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{len(matches)} parts are named {part!r}; the parts are: {labels}"
        )
    return matches[0]


def _voice_elements(line, voice) -> list[tuple[float, Any, tuple]]:
    """``[(offset_in_quarters, note_or_rest_or_None, bar)]`` of one voice of a part,
    in time order; ``bar`` is ``(pass, bar_index)`` (the pass counts the times a
    repeated bar was played before), None an element a silent bar of the voice.

    A bar music21 holds without voices belongs to the voice the MusicXML gives
    it; a bar whose voice is not known (a music21 stream with several voices in
    other bars) is an error rather than another voice's notes sung.
    """
    from music21 import stream as m21stream

    measures = list(line.getElementsByClass(m21stream.Measure))
    if not measures:
        return sorted(
            ((float(el.offset), el, (0, 0)) for el in line.flatten().notesAndRests),
            key=lambda x: x[0],
        )
    ids: list[str] = []
    for m in measures:
        for v in m.voices:
            if str(v.id) not in ids:
                ids.append(str(v.id))
    chosen = _pick_voice(measures, ids, voice) if ids else None
    out = []
    seen: dict = {}
    for index, m in enumerate(measures):
        key = m.number  # expandRepeats renumbers a repeated bar 1 as 1a
        bar = (seen.get(key, 0), index)
        seen[key] = bar[0] + 1
        base = float(m.offset)
        if m.voices:
            for v in m.voices:
                if str(v.id) == chosen:
                    out.extend(
                        (base + float(v.offset) + float(el.offset), el, bar)
                        for el in v.notesAndRests
                    )
            out.extend(  # notes held by the bar itself, outside its voices
                (base + float(el.offset), el, bar)
                for el in m.getElementsByClass(("Note", "Chord", "Rest"))
            )
            continue
        bar_voices = m.editorial.get("voices")
        if chosen is not None and bar_voices is None:
            raise ValueError(
                f"Bar {m.number} has one voice and others have several; which voice "
                "it is is not known for a music21 stream: pass the MusicXML file"
            )
        if chosen is None or chosen in (bar_voices or []):
            out.extend((base + float(el.offset), el, bar) for el in m.notesAndRests)
        else:
            out.append((base, None, bar))  # the voice is silent in this bar
    return sorted(out, key=lambda x: x[0])


def _pick_voice(measures, ids, voice) -> str:
    def voice_has_lyrics(vid):
        return any(
            _has_lyrics(v) for m in measures for v in m.voices if str(v.id) == vid
        )

    if voice is not None:
        if str(voice) not in ids:
            raise ValueError(f"No voice {voice!r} in this part; its voices are {ids}")
        return str(voice)
    worded = [vid for vid in ids if voice_has_lyrics(vid)]
    if len(worded) > 1:
        raise ValueError(
            f"Voices {worded} of this part all have lyrics: pick one with voice="
        )
    if worded:
        return worded[0]
    return sorted(
        ids, key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else 0, v)
    )[0]


def _verse_chooser(elements, verse) -> Optional[Callable]:
    """``bar -> verse number``: ``verse`` throughout if given; else verse n on the
    n-th pass through a bar when that bar has a verse n, else the lowest verse."""

    def numbers_of(el):
        return {
            int(ly.number)
            for ly in getattr(el, "lyrics", ())
            if (ly.text or "").strip() or ly.isComposite
        }

    numbers = sorted(set().union(*(numbers_of(el) for _, el, _ in elements if el)))
    if verse is not None:
        if verse not in numbers:
            raise ValueError(
                f"No verse {verse} in the sung line; its verses are {numbers}"
            )
        return lambda bar: verse
    if not numbers:
        return None
    in_bar: dict = {}
    for _, el, bar in elements:
        if el is not None:
            in_bar.setdefault(bar, set()).update(numbers_of(el))

    def verse_at(bar):
        wanted = numbers[min(bar[0], len(numbers) - 1)]
        return wanted if wanted in in_bar.get(bar, ()) else numbers[0]

    return verse_at


def _verse_lyric(element, verse):
    """``(text, first_syllabic, last_syllabic, extend)`` of the element's lyric in
    ``verse``, or None; two lyrics of one verse on one note (a chord) is an error."""
    lyrics = [ly for ly in element.lyrics if int(ly.number) == verse]
    if len(lyrics) > 1:
        raise ValueError(
            f"{len(lyrics)} lyrics of verse {verse} on one chord (measure "
            f"{element.measureNumber}): keep one, or pick a voice with voice="
        )
    if not lyrics:
        return None
    lyric = lyrics[0]
    parts = list(lyric.components) if lyric.isComposite else [lyric]
    texts = [(p.text or "").strip() for p in parts]
    if not any(texts):
        return None
    extend = verse in (element.editorial.get("extend") or ())
    return (
        " ".join(t for t in texts if t),
        parts[0].syllabic or "single",
        parts[-1].syllabic or "single",
        extend,
    )


# --------------------------------------------------------------------------- words


def _join_words(items: Iterable, *, rest_s: float) -> list[SungNote]:
    """Notes with whole words, from ``(note, syllable, joins, holds)`` items
    (``syllable`` None for a note without one) and None items (rests).

    A syllable that ``joins`` continues the current word, even across a rest (the
    score said so). A note without a syllable continues the current word as a
    melisma, unless a rest (a None item, or a gap of ``rest_s``) came between it
    and the word, and the word's last syllable does not hold across rests
    (``holds``: a word left open by a hyphen, or a MusicXML ``<extend/>`` line).
    """
    notes: list[SungNote] = []
    word: list[int] = []
    parts: list[str] = []
    can_hold, holds_over_rest, previous = False, False, None
    for item in items:
        if item is None:
            can_hold = can_hold and holds_over_rest
            continue
        note, syllable, joins, holds = item
        if previous is not None and note.start - previous.end >= rest_s - 1e-9:
            can_hold = can_hold and holds_over_rest
        if syllable is not None:
            starts = not (joins and word)
            if starts:
                word, parts = [], []
            parts.append(syllable)
            notes.append(replace(note, syllable=syllable, starts_word=starts))
            word.append(len(notes) - 1)
            can_hold, holds_over_rest = True, holds
        elif word and can_hold:
            notes.append(replace(note, syllable="", starts_word=False))
            word.append(len(notes) - 1)
        else:
            notes.append(replace(note, word="", syllable="", starts_word=True))
            word, parts = [], []
            can_hold = holds_over_rest = False
        if word:
            text = "".join(parts)
            for i in word:
                notes[i] = replace(notes[i], word=text)
        previous = note
    return notes


def lyric_tokens(lyrics: str) -> list[tuple[str, str, bool]]:
    """``(syllable, word, starts_word)`` per sung note, from a lyric line.

    Tokens are whitespace-separated, one per note: ``can-dle`` spreads one word
    over two notes, a bare ``-`` (or ``_``, ``~``) holds the previous word one
    more note (a melisma, empty syllable).

    >>> lyric_tokens("Light can-dle -")
    [('Light', 'Light', True), ('can', 'candle', True), ('dle', 'candle', False), ('', 'candle', False)]
    """
    tokens: list[tuple[str, str, bool]] = []
    for token in lyrics.split():
        if token in HOLD_TOKENS:
            if not tokens:
                raise ValueError(f"A hold token ({token!r}) can't come first")
            tokens.append(("", tokens[-1][1], False))
            continue
        syllables = [s for s in token.split("-") if s]
        word = "".join(syllables)
        tokens.extend((s, word, i == 0) for i, s in enumerate(syllables))
    return tokens


def attach_lyrics(notes: Sequence[SungNote], lyrics: str) -> list[SungNote]:
    """``notes`` with the words of a hand-written lyric line, one token per note
    (see :func:`lyric_tokens`).

    >>> line = [SungNote(0.0, 0.5, 60), SungNote(0.5, 0.5, 62)]
    >>> [(n.syllable, n.word, n.starts_word) for n in attach_lyrics(line, "hel-lo")]
    [('hel', 'hello', True), ('lo', 'hello', False)]
    """
    tokens = lyric_tokens(lyrics)
    if len(tokens) != len(notes):
        raise ValueError(
            f"{len(tokens)} lyric syllables for {len(notes)} notes: write 'can-dle' "
            "for one word over two notes, '-' to hold a word"
        )
    return [
        replace(n, syllable=s, word=w, starts_word=starts)
        for n, (s, w, starts) in zip(notes, tokens)
    ]
