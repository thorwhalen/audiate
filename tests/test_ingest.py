"""Tests for symbolic-score ingest (format detection + normalization to MIDI)."""

import pretty_midi

from audiate.ingest import detect_format, to_pretty_midi


def test_passthrough_prettymidi():
    pm = pretty_midi.PrettyMIDI()
    assert to_pretty_midi(pm) is pm


def test_detect_format_by_extension():
    assert detect_format("x.mid") == "midi"
    assert detect_format("x.midi") == "midi"
    assert detect_format("x.musicxml") == "musicxml"
    assert detect_format("x.mxl") == "musicxml"
    assert detect_format("x.abc") == "abc"
    assert detect_format("x.krn") == "kern"
    assert detect_format("song.unknownext") == "unknown"


def test_detect_format_midi_bytes():
    assert detect_format(b"MThd\x00\x00\x00\x06") == "midi"
    assert detect_format(b"not a midi") == "unknown"


def test_detect_format_objects():
    assert detect_format(pretty_midi.PrettyMIDI()) == "prettymidi"


def test_ingest_music21_stream():
    from music21 import note, stream

    s = stream.Stream()
    s.append(note.Note("C4", quarterLength=1))
    s.append(note.Note("E4", quarterLength=1))
    pm = to_pretty_midi(s)
    all_notes = [n for inst in pm.instruments for n in inst.notes]
    assert len(all_notes) == 2
    assert {n.pitch for n in all_notes} == {60, 64}  # C4, E4


def test_ingest_musicxml_file(tmp_path):
    from music21 import note, stream

    s = stream.Stream()
    s.append(note.Note("G4", quarterLength=2))
    xml_path = str(tmp_path / "tune.musicxml")
    s.write("musicxml", fp=xml_path)

    pm = to_pretty_midi(xml_path)
    all_notes = [n for inst in pm.instruments for n in inst.notes]
    assert len(all_notes) == 1
    assert all_notes[0].pitch == 67  # G4


def test_ingest_midi_bytes_roundtrip(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    pm.instruments.append(inst)
    midi_path = str(tmp_path / "x.mid")
    pm.write(midi_path)
    with open(midi_path, "rb") as f:
        data = f.read()
    got = to_pretty_midi(data)
    assert len([n for i in got.instruments for n in i.notes]) == 1
