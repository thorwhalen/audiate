"""End-to-end tests for the ``render()`` facade."""

import numpy as np
import pretty_midi
import pytest

import audiate


def _one_note_pm(pitch=60, end=0.5):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=0.0, end=end))
    pm.instruments.append(inst)
    return pm


def test_render_prettymidi_sine():
    audio = audiate.render(_one_note_pm(), engine="sine", sample_rate=16000)
    assert isinstance(audio, audiate.AudioData)
    assert audio.sample_rate == 16000
    assert audio.engine == "sine"
    assert audio.source_format == "prettymidi"
    assert audio.array.shape[0] > 0
    assert float(np.max(np.abs(audio.array))) > 0.0
    assert audio.duration_seconds > 0.0


def test_render_music21_stream_sine():
    from music21 import note, stream

    s = stream.Stream()
    s.append(note.Note("C4", quarterLength=1))
    audio = audiate.render(s, engine="sine", sample_rate=16000)
    assert audio.array.shape[0] > 0
    assert audio.source_format == "music21"


def test_render_auto_engine_selects_sine_without_soundfont(monkeypatch):
    # Force "no soundfont" so auto resolves to the always-available sine engine.
    monkeypatch.setattr("audiate.synth.default_soundfont", lambda: None)
    audio = audiate.render(_one_note_pm(), sample_rate=16000)
    assert audio.engine == "sine"


def test_list_engines():
    names = audiate.list_engines()
    assert "sine" in names
    assert "fluidsynth" in names
    assert "musescore" in names


def test_unknown_engine_raises():
    with pytest.raises(ValueError, match="Unknown engine"):
        audiate.render(_one_note_pm(), engine="does_not_exist")


def test_audiodata_wav_roundtrip(tmp_path):
    audio = audiate.render(_one_note_pm(), engine="sine", sample_rate=16000)
    wav_bytes = audio.to_wav_bytes()
    assert wav_bytes[:4] == b"RIFF"
    out = audio.write(str(tmp_path / "out.wav"))
    import os

    assert os.path.exists(out) and os.path.getsize(out) > 0
