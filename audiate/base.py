"""Core container for audiate: :class:`AudioData`, a rendered audio waveform.

A render produces an :class:`AudioData` -- a NumPy waveform plus its sample rate
and a little provenance (what symbolic format it came from, which engine made
it). It knows how to serialize itself to WAV bytes or a file.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any


@dataclass
class AudioData:
    """A rendered audio waveform.

    Attributes:
        array: float32 NumPy array -- shape ``(n_samples,)`` (mono) or
            ``(n_samples, channels)``.
        sample_rate: Samples per second (Hz).
        source_format: The symbolic format rendered from (e.g. ``'midi'``,
            ``'musicxml'``), best-effort; ``''`` if unknown.
        engine: The synthesis engine used (e.g. ``'sine'``, ``'fluidsynth'``).
    """

    array: Any
    sample_rate: int
    source_format: str = ""
    engine: str = ""

    @property
    def duration_seconds(self) -> float:
        """Length of the audio in seconds."""
        n = self.array.shape[0]
        return n / self.sample_rate if self.sample_rate else 0.0

    def to_wav_bytes(self) -> bytes:
        """Return the waveform encoded as WAV file bytes (needs ``soundfile``)."""
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, self.array, self.sample_rate, format="WAV")
        return buf.getvalue()

    def write(self, path: str) -> str:
        """Write the waveform to ``path`` (format inferred from extension).

        Returns the path written to.
        """
        import soundfile as sf

        sf.write(path, self.array, self.sample_rate)
        return path
