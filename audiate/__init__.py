"""audiate -- render a symbolic music score to audio.

Turn a score in any of several standard formats (MIDI, MusicXML, ABC,
Humdrum-``kern``, MEI, ...) into an audio waveform. The universal pivot is
**MIDI**: every input is normalized to a ``pretty_midi.PrettyMIDI`` (ingest),
then synthesized by a pluggable **engine** (``sine`` / ``fluidsynth`` /
``musescore``).

Simple things simple::

    import audiate

    audio = audiate.render("song.mid")                 # -> AudioData
    audio = audiate.render("lead_sheet.musicxml")      # auto-detects format
    audio.write("out.wav")

Complex things possible::

    # pick an engine + SoundFont explicitly
    audio = audiate.render("tune.abc", engine="fluidsynth", soundfont="MyPiano.sf2")

    # register your own engine
    from audiate import register_engine

    @register_engine("my_synth")
    def my_synth(pm, *, sample_rate=44100, **opts):
        ...
        return waveform, sample_rate

This is the "score2audio" stage of a larger song->score->audio->AI-enhanced
pipeline; the AI-enhancement stage lives in ``arioso``, and chord-chart
arrangement lives in ``accompy``.
"""

from audiate.base import AudioData
from audiate.ingest import detect_format, to_pretty_midi
from audiate.registry import engines, register_engine

# Importing the synth module registers the built-in engines into ``engines``.
from audiate import synth as _synth  # noqa: F401

__all__ = [
    "render",
    "list_engines",
    "to_pretty_midi",
    "detect_format",
    "AudioData",
    "engines",
    "register_engine",
]

#: Default render sample rate (Hz).
DFLT_SAMPLE_RATE = 44100


def render(
    source,
    *,
    engine: str = "auto",
    sample_rate: int = DFLT_SAMPLE_RATE,
    soundfont: str = None,
    **opts,
) -> AudioData:
    """Render a symbolic score to audio.

    Args:
        source: The score -- a ``pretty_midi.PrettyMIDI``, a ``music21`` stream,
            MIDI ``bytes``, or a file path (``.mid``/MusicXML/ABC/``kern``/MEI).
        engine: Synthesis engine name, or ``'auto'`` (default) which picks
            ``'fluidsynth'`` when a SoundFont is available, else ``'sine'``.
        sample_rate: Requested output sample rate in Hz (engines that control it
            honor it; an external tool may report its own rate back).
        soundfont: Path to a ``.sf2``/``.sf3`` SoundFont (for ``fluidsynth``).
        **opts: Extra engine-specific options.

    Returns:
        An :class:`~audiate.base.AudioData`.
    """
    pm = to_pretty_midi(source)
    name = _resolve_engine(engine, soundfont=soundfont)
    array, sr = engines[name](pm, sample_rate=sample_rate, soundfont=soundfont, **opts)
    return AudioData(
        array=array,
        sample_rate=sr,
        source_format=detect_format(source),
        engine=name,
    )


def list_engines() -> list:
    """Return the names of the registered synthesis engines (sorted)."""
    return sorted(engines)


def _resolve_engine(engine: str, *, soundfont: str = None) -> str:
    if engine != "auto":
        if engine not in engines:
            raise ValueError(f"Unknown engine {engine!r}. Available: {list_engines()}")
        return engine
    from audiate.synth import default_soundfont

    if soundfont or default_soundfont():
        return "fluidsynth"
    return "sine"
