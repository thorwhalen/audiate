"""Tests for the synthesis engines.

The ``sine`` engine is pure NumPy, so it is verified end-to-end (including a
real spectral check that a rendered note has energy at the right frequency).
"""

import numpy as np
import pretty_midi
import pytest

from audiate.registry import engines
from audiate.synth import default_soundfont, render_sine


def _one_note_pm(pitch=69, start=0.0, end=1.0, velocity=100):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(
        pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
    )
    pm.instruments.append(inst)
    return pm


def test_sine_produces_nonsilent_audio():
    pm = _one_note_pm(end=1.0)
    array, sr = render_sine(pm, sample_rate=8000)
    assert sr == 8000
    assert array.dtype == np.float32
    assert array.shape[0] >= 8000  # ~1 second
    assert float(np.max(np.abs(array))) > 0.0  # not silent


def test_sine_frequency_is_correct():
    # A4 (MIDI 69) should put its spectral peak at ~440 Hz.
    pm = _one_note_pm(pitch=69, end=1.0)
    array, sr = render_sine(pm, sample_rate=8000)
    spectrum = np.abs(np.fft.rfft(array))
    freqs = np.fft.rfftfreq(len(array), 1.0 / sr)
    peak_freq = freqs[int(np.argmax(spectrum))]
    assert abs(peak_freq - 440.0) < 10.0


def test_sine_empty_score_is_silent():
    array, sr = render_sine(pretty_midi.PrettyMIDI(), sample_rate=8000)
    assert sr == 8000
    assert float(np.max(np.abs(array))) == 0.0


def test_sine_skips_drums():
    pm = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True)
    drum.notes.append(pretty_midi.Note(velocity=100, pitch=38, start=0.0, end=1.0))
    pm.instruments.append(drum)
    array, _ = render_sine(pm, sample_rate=8000)
    assert float(np.max(np.abs(array))) == 0.0  # drum note skipped -> silence


def test_builtin_engines_registered():
    for name in ("sine", "fluidsynth", "musescore"):
        assert name in engines
        assert callable(engines[name])


@pytest.mark.skipif(default_soundfont() is None, reason="no SoundFont installed")
def test_fluidsynth_renders_when_soundfont_available():
    from audiate.synth import render_fluidsynth

    pm = _one_note_pm(end=0.5)
    array, sr = render_fluidsynth(pm, sample_rate=22050)
    assert sr == 22050
    assert array.shape[0] > 0
    assert float(np.max(np.abs(array))) > 0.0
