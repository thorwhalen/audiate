"""Score operations on the MIDI pivot: crop a score to a time range, export it.

Every operation takes any score source :func:`audiate.to_pretty_midi` accepts
and works on the ``pretty_midi.PrettyMIDI`` pivot, so cropping a MusicXML file
and cropping a MIDI file are the same call.

- :func:`crop` keeps the music between two times (in seconds of the score's own
  timeline): notes that straddle a cut are clipped rather than dropped, the
  controller state in force at the start (volume, pan, sustain, pitch bend) is
  carried to time zero, and the tempo map, time signatures and key signatures
  are shifted. A cut inside a bar opens with a short bar so later bar lines
  stay where they were (when the cut is on a beat unit). With
  ``snap="downbeat"`` the cut points move outward to bar lines, which is what
  you want before turning the result into notation.
- :func:`export` writes a score to ``.mid``, or through the MuseScore CLI to
  MusicXML (``.musicxml`` / ``.mxl``), PDF or audio; MusicXML falls back to
  ``music21`` when MuseScore is not installed.

``pretty_midi.PrettyMIDI.adjust_times`` is not used for cropping: it drops the
notes that straddle the cut and collapses the tempo map.
"""

from __future__ import annotations

import bisect
import copy
import os
import subprocess
from pathlib import Path
from typing import Literal, Optional

from audiate.ingest import to_pretty_midi

Snap = Optional[Literal["downbeat", "beat"]]

#: Extensions :func:`export` hands to the MuseScore CLI.
_MUSESCORE_EXPORTS = {".musicxml", ".mxl", ".xml", ".pdf", ".wav", ".mp3", ".flac", ".png", ".mscz"}


def _snap_times(pm, start: float, end: float, snap: Snap) -> tuple:
    """Move ``start`` down and ``end`` up to the nearest grid line."""
    if snap is None:
        return start, end
    if snap not in ("downbeat", "beat"):
        raise ValueError(f"snap must be None, 'downbeat' or 'beat', got {snap!r}")
    grid = pm.get_downbeats() if snap == "downbeat" else pm.get_beats()
    grid = sorted(float(t) for t in grid)
    if not grid:
        return start, end
    i = bisect.bisect_right(grid, start + 1e-6) - 1
    new_start = grid[i] if i >= 0 else start
    j = bisect.bisect_left(grid, end - 1e-6)
    new_end = grid[j] if j < len(grid) else end
    return new_start, new_end


def _shifted_tempo_map(pm, new, start: float, end: float) -> None:
    """Give ``new`` the tempo map of ``pm`` on ``[start, end)``, shifted to zero.

    pretty_midi keeps tempo as ``_tick_scales`` (tick, seconds-per-tick) and
    has no public setter, so this writes the private attribute and rebuilds
    the tick table the way ``PrettyMIDI.__init__`` does.
    """
    t0 = pm.time_to_tick(start)
    t1 = pm.time_to_tick(end)
    scales = [(tick, scale) for tick, scale in pm._tick_scales if tick < t1]
    active = scales[0][1]
    shifted = []
    for tick, scale in scales:
        if tick <= t0:
            active = scale
        else:
            shifted.append((tick - t0, scale))
    new._tick_scales = [(0, active)] + shifted
    new._update_tick_to_time(max(t1 - t0, 1))


#: Seconds below which a clipped note or a bar remainder counts as zero.
_EPS = 1e-6


def _cropped_time_signatures(pm, start: float, end: float) -> list:
    """Time signatures for the window, with a partial first bar when cut mid-bar.

    Moving the signature in force to time 0 would start a full bar at the cut.
    When the cut falls inside a bar, on a whole number of the signature's
    beat units before the next bar line, the window instead opens with a
    short bar (``k/denominator``) and the original signature resumes at that
    bar line, so every later bar line lands where it was. A cut that is not
    on a beat unit keeps the plain shifted signatures (bar lines then start
    at the cut); ``snap="downbeat"`` avoids the question.
    """
    import pretty_midi

    def make(e, t):
        return pretty_midi.TimeSignature(e.numerator, e.denominator, t)

    shifted = _shift_meta(pm.time_signature_changes, start, end, make=make)
    if not shifted or shifted[0].time > _EPS:
        return shifted
    downbeats = [float(t) for t in pm.get_downbeats()]
    later = [t for t in downbeats if t > start + _EPS]
    on_bar = any(abs(t - start) <= _EPS for t in downbeats)
    if on_bar or not later or later[0] >= end - _EPS:
        return shifted
    first, next_bar = shifted[0], later[0]
    unit_ticks = pm.resolution * 4 / first.denominator
    units = (pm.time_to_tick(next_bar) - pm.time_to_tick(start)) / unit_ticks
    k = round(units)
    if k < 1 or abs(units - k) > 1e-3:
        return shifted
    resume_at = next_bar - start
    rest = [e for e in shifted[1:]]
    pickup = [pretty_midi.TimeSignature(k, first.denominator, 0.0)]
    if not any(abs(e.time - resume_at) <= _EPS for e in rest):
        pickup.append(make(first, resume_at))
    return sorted(pickup + rest, key=lambda e: e.time)


def _shift_meta(events, start: float, end: float, *, make):
    """Keep the event in force at ``start`` (moved to 0) and those inside."""
    before = [e for e in events if e.time <= start]
    inside = [e for e in events if start < e.time < end]
    kept = ([make(before[-1], 0.0)] if before else []) + [
        make(e, e.time - start) for e in inside
    ]
    return kept


def crop(
    source,
    start: float = 0.0,
    end: Optional[float] = None,
    *,
    snap: Snap = None,
    keep_straddling: bool = True,
):
    """Return a new ``PrettyMIDI`` holding only ``[start, end)`` of a score.

    Args:
        source: Any score :func:`audiate.to_pretty_midi` accepts (a path to a
            MIDI/MusicXML/ABC/... file, MIDI bytes, a PrettyMIDI, a music21
            stream). A PrettyMIDI passed in is not modified.
        start: Start time in seconds on the score's timeline.
        end: End time in seconds (default: the end of the score).
        snap: ``None`` (cut exactly), ``"downbeat"`` (widen to bar lines) or
            ``"beat"`` (widen to beats).
        keep_straddling: Keep notes that cross a cut, clipped to the window
            (default). ``False`` keeps only notes wholly inside it.

    Returns:
        A ``pretty_midi.PrettyMIDI`` whose time zero is ``start``.

    Example::

        first_minute = audiate.crop("theme.mid", 0, 60, snap="downbeat")
        audiate.render(first_minute).write("first_minute.wav")
    """
    import pretty_midi

    pm = to_pretty_midi(source)
    score_end = pm.get_end_time()
    end = score_end if end is None else min(end, score_end)
    if not 0 <= start < end:
        raise ValueError(f"need 0 <= start < end, got start={start}, end={end}")
    start, end = _snap_times(pm, start, end, snap)

    new = pretty_midi.PrettyMIDI(resolution=pm.resolution)
    _shifted_tempo_map(pm, new, start, end)
    new.time_signature_changes = _cropped_time_signatures(pm, start, end)
    new.key_signature_changes = _shift_meta(
        pm.key_signature_changes,
        start,
        end,
        make=lambda e, t: pretty_midi.KeySignature(e.key_number, t),
    )
    new.lyrics = [
        pretty_midi.Lyric(x.text, x.time - start)
        for x in pm.lyrics
        if start <= x.time < end
    ]

    for inst in pm.instruments:
        out = pretty_midi.Instrument(inst.program, is_drum=inst.is_drum, name=inst.name)
        for n in inst.notes:
            inside = start <= n.start and n.end <= end
            # Clipped length must be positive (beyond float noise), or the
            # note would vanish on write() and the counts would disagree.
            overlaps = min(n.end, end) - max(n.start, start) > _EPS
            if overlaps and (inside or keep_straddling):
                out.notes.append(
                    pretty_midi.Note(
                        n.velocity,
                        n.pitch,
                        max(n.start, start) - start,
                        min(n.end, end) - start,
                    )
                )
        by_number = {}
        for c in inst.control_changes:
            by_number.setdefault(c.number, []).append(c)
        for number, events in by_number.items():
            out.control_changes += _shift_meta(
                events,
                start,
                end,
                make=lambda e, t: pretty_midi.ControlChange(e.number, e.value, t),
            )
        out.pitch_bends = _shift_meta(
            inst.pitch_bends,
            start,
            end,
            make=lambda e, t: pretty_midi.PitchBend(e.pitch, t),
        )
        if out.notes:
            new.instruments.append(out)
    return new


def export(source, path, *, musescore_bin: str = None, timeout: float = 600):
    """Write a score to ``path``; the extension picks the format.

    ``.mid``/``.midi`` are written by pretty_midi. MusicXML (``.musicxml``,
    ``.mxl``, ``.xml``), ``.pdf``, ``.png``, ``.mscz`` and audio go through the
    MuseScore CLI, which also quantizes the MIDI into readable notation;
    without MuseScore, MusicXML is written by ``music21`` instead.

    Args:
        source: Any score :func:`audiate.to_pretty_midi` accepts.
        path: Output file path.
        musescore_bin: The MuseScore CLI binary (default: found the same way
            the ``musescore`` engine finds it).
        timeout: Seconds to allow the MuseScore CLI.

    Returns:
        The output path as a :class:`pathlib.Path`.
    """
    import tempfile

    from audiate.synth import find_musescore

    path = Path(path)
    ext = path.suffix.lower()
    pm = to_pretty_midi(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    if ext in (".mid", ".midi"):
        pm.write(str(path))
        return path
    if ext not in _MUSESCORE_EXPORTS:
        raise ValueError(f"Don't know how to export to {ext!r}")
    exe = musescore_bin or find_musescore()
    fd, midi_path = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    try:
        pm.write(midi_path)
        if path.exists():  # never mistake an earlier run's file for this one's
            path.unlink()
        if exe:
            proc = subprocess.run(
                [exe, "-o", str(path), midi_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0 or not path.exists():
                raise RuntimeError(
                    f"MuseScore CLI failed to export {path.name} "
                    f"(exit {proc.returncode}): {proc.stderr.strip()[:400]}"
                )
            return path
        if ext in (".musicxml", ".xml", ".mxl"):
            try:
                import music21
            except ImportError as exc:
                raise RuntimeError(
                    "MusicXML export needs the MuseScore CLI or music21 "
                    "(pip install 'audiate[symbolic]')."
                ) from exc
            music21.converter.parse(midi_path).write(
                "mxl" if ext == ".mxl" else "musicxml", fp=str(path)
            )
            return path
        raise RuntimeError(f"Exporting {ext!r} needs the MuseScore CLI ($AUDIATE_MUSESCORE).")
    finally:
        if os.path.exists(midi_path):
            os.remove(midi_path)
