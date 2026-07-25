"""Built-in synthesis engines: ``sine``, ``fluidsynth``, ``musescore``.

Importing this module registers the built-in engines into
:data:`audiate.registry.engines`. Each engine takes a ``pretty_midi.PrettyMIDI``
and returns ``(waveform, sample_rate)``.

Tiers (progressive disclosure):

- ``sine`` -- pure NumPy additive-sine synth. Zero system dependencies; always
  available. Deterministic and cheap -- ideal for tests, CI, and quick previews.
- ``fluidsynth`` -- SoundFont synthesis via ``pretty_midi.fluidsynth`` (needs the
  FluidSynth library + ``pyFluidSynth`` + a ``.sf2``/``.sf3`` SoundFont). The
  quality default once a SoundFont is available.
- ``musescore`` -- shells out to the MuseScore CLI for notation-aware audio
  (dynamics/articulations/repeats). Highest quality; needs the MuseScore binary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional

from audiate.registry import register_engine

# ---------------------------------------------------------------------------
# sine -- pure NumPy, no system deps
# ---------------------------------------------------------------------------

#: Attack/release ramp length (seconds) used by the sine engine to avoid clicks.
_ENV_RAMP_S = 0.005


def _envelope(n: int, sample_rate: int):
    import numpy as np

    env = np.ones(n, dtype="float32")
    ramp = min(int(_ENV_RAMP_S * sample_rate), n // 2)
    if ramp > 0:
        env[:ramp] = np.linspace(0.0, 1.0, ramp, dtype="float32")
        env[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype="float32")
    return env


@register_engine("sine")
def render_sine(pm, *, sample_rate: int = 44100, **opts):
    """Additive-sine synth in pure NumPy (no system dependencies).

    Sums one enveloped sine per (non-drum) note at its equal-tempered frequency.
    Returns ``(float32 mono waveform, sample_rate)``.
    """
    import numpy as np

    end = pm.get_end_time() if pm.instruments else 0.0
    n = int(np.ceil(end * sample_rate)) + 1
    out = np.zeros(n, dtype="float32")
    if n <= 1:
        return out, sample_rate

    for inst in pm.instruments:
        if getattr(inst, "is_drum", False):
            continue
        for note in inst.notes:
            freq = 440.0 * 2.0 ** ((note.pitch - 69) / 12.0)
            s = int(note.start * sample_rate)
            e = int(note.end * sample_rate)
            if e <= s:
                continue
            e = min(e, n)
            seg_n = e - s
            local_t = np.arange(seg_n, dtype="float32") / sample_rate
            amp = (note.velocity / 127.0) * 0.2
            wave = (
                amp
                * _envelope(seg_n, sample_rate)
                * np.sin(2.0 * np.pi * freq * local_t)
            )
            out[s:e] += wave.astype("float32")

    peak = float(np.max(np.abs(out))) if n else 0.0
    if peak > 1.0:
        out = (out / peak).astype("float32")
    return out, sample_rate


# ---------------------------------------------------------------------------
# fluidsynth -- SoundFont synthesis
# ---------------------------------------------------------------------------

#: Candidate SoundFont locations probed by :func:`default_soundfont`.
_SOUNDFONT_CANDIDATES = (
    os.path.expanduser("~/.fluidsynth/default_sound_font.sf2"),
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/soundfonts/default.sf2",
    "/Applications/MuseScore 4.app/Contents/Resources/soundfonts/MS Basic.sf3",
    "/Applications/MuseScore 4.app/Contents/Resources/sound/MS Basic.sf3",
    "/opt/homebrew/share/soundfonts/default.sf2",
)


def default_soundfont() -> Optional[str]:
    """Locate a usable SoundFont, or ``None`` if none is found.

    Honors ``$AUDIATE_SOUNDFONT`` first, then a list of common OS locations
    (including MuseScore's bundled SoundFont).
    """
    env = os.environ.get("AUDIATE_SOUNDFONT")
    if env and os.path.exists(env):
        return env
    for path in _SOUNDFONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


@register_engine("fluidsynth")
def render_fluidsynth(pm, *, sample_rate: int = 44100, soundfont: str = None, **opts):
    """SoundFont synthesis via ``pretty_midi.fluidsynth``.

    Needs the FluidSynth library + ``pyFluidSynth`` and a ``.sf2``/``.sf3``
    SoundFont (``soundfont=`` or :func:`default_soundfont`). Returns
    ``(float32 mono waveform, sample_rate)``.
    """
    import numpy as np

    sf2 = soundfont or default_soundfont()
    if sf2 is None:
        raise RuntimeError(
            "No SoundFont available for the 'fluidsynth' engine. Pass "
            "soundfont=<path to .sf2/.sf3>, set $AUDIATE_SOUNDFONT, or use "
            "engine='sine'."
        )
    audio = pm.fluidsynth(fs=sample_rate, sf2_path=sf2)
    return np.asarray(audio, dtype="float32"), sample_rate


# ---------------------------------------------------------------------------
# musescore -- notation-aware audio via the MuseScore CLI (subprocess)
# ---------------------------------------------------------------------------

#: Candidate MuseScore CLI binary locations probed by :func:`find_musescore`.
_MUSESCORE_CANDIDATES = (
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
)


def find_musescore() -> Optional[str]:
    """Locate the MuseScore CLI binary, or ``None`` if not found.

    Honors ``$AUDIATE_MUSESCORE`` first, then ``PATH`` (``mscore`` /
    ``musescore`` / ``MuseScore4``), then common app-bundle locations.
    """
    env = os.environ.get("AUDIATE_MUSESCORE")
    if env and os.path.exists(env):
        return env
    for name in ("mscore", "musescore", "MuseScore4", "MuseScore"):
        found = shutil.which(name)
        if found:
            return found
    for path in _MUSESCORE_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


@register_engine("musescore")
def render_musescore(
    pm, *, sample_rate: int = 44100, musescore_bin: str = None, **opts
):
    """Render notation-aware audio by shelling out to the MuseScore CLI.

    Writes the MIDI to a temp file, runs ``mscore -o out.wav in.mid``, and reads
    the result back. Returns ``(waveform, actual_sample_rate)`` -- MuseScore uses
    its own rate (typically 44100), which is reported back rather than the
    requested ``sample_rate``.
    """
    import soundfile as sf

    exe = musescore_bin or find_musescore()
    if not exe:
        raise RuntimeError(
            "MuseScore CLI not found for the 'musescore' engine. Set "
            "$AUDIATE_MUSESCORE to the binary, or use engine='fluidsynth'/'sine'."
        )
    fd_in, midi_path = tempfile.mkstemp(suffix=".mid")
    os.close(fd_in)
    fd_out, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd_out)
    try:
        pm.write(midi_path)
        proc = subprocess.run(
            [exe, "-o", wav_path, midi_path],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not os.path.getsize(wav_path):
            raise RuntimeError(
                f"MuseScore CLI failed (exit {proc.returncode}). "
                f"stderr: {proc.stderr.strip()[:400]}"
            )
        array, sr = sf.read(wav_path)
        import numpy as np

        return np.asarray(array, dtype="float32"), int(sr)
    finally:
        for p in (midi_path, wav_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:  # pragma: no cover
                pass
