"""Tests for audiate.ops: crop (time window on the MIDI pivot) and export."""

import pretty_midi
import pytest

import audiate


def _score(tempo_change_at=None):
    """Two bars of 4/4 quarter notes at 120 bpm (0.5 s each), one long note."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))
    piano = pretty_midi.Instrument(0, name="piano")
    for k in range(8):
        piano.notes.append(pretty_midi.Note(80, 60 + k, k * 0.5, k * 0.5 + 0.4))
    piano.notes.append(pretty_midi.Note(70, 48, 0.0, 4.0))  # straddles any cut
    piano.control_changes.append(pretty_midi.ControlChange(7, 90, 0.0))
    piano.control_changes.append(pretty_midi.ControlChange(7, 50, 1.2))
    pm.instruments.append(piano)
    return pm


def test_crop_shifts_and_clips():
    out = audiate.crop(_score(), 1.0, 3.0)
    starts = sorted(round(n.start, 3) for n in out.instruments[0].notes)
    assert starts == [0.0, 0.0, 0.5, 1.0, 1.5]  # long note clipped + 4 quarters
    long_note = [n for n in out.instruments[0].notes if n.pitch == 48][0]
    assert long_note.end == pytest.approx(2.0)
    assert out.get_end_time() == pytest.approx(2.0)


def test_crop_can_drop_straddling_notes():
    out = audiate.crop(_score(), 1.0, 3.0, keep_straddling=False)
    assert all(n.pitch != 48 for n in out.instruments[0].notes)


def test_crop_carries_controller_state_to_zero():
    out = audiate.crop(_score(), 1.0, 3.0)
    ccs = [(c.number, c.value, round(c.time, 3)) for c in out.instruments[0].control_changes]
    assert ccs == [(7, 90, 0.0), (7, 50, 0.2)]


def test_crop_does_not_mutate_input():
    pm = _score()
    audiate.crop(pm, 1.0, 3.0)
    assert len(pm.instruments[0].notes) == 9


def test_crop_snap_to_downbeat_widens_to_bars():
    out = audiate.crop(_score(), 0.7, 1.3, snap="downbeat")  # bars at 0, 2, 4 s
    assert out.get_end_time() == pytest.approx(2.0)
    assert len(out.time_signature_changes) == 1


def test_crop_keeps_tempo_after_start():
    pm = _score()
    pm._tick_scales.append((pm.time_to_tick(2.0), 60.0 / (60 * pm.resolution)))
    pm._update_tick_to_time(pm.time_to_tick(10.0))
    out = audiate.crop(pm, 1.0, 3.0)
    times, tempi = out.get_tempo_changes()
    assert list(tempi.round()) == [120, 60]
    assert times[1] == pytest.approx(1.0)


def test_crop_rejects_empty_window():
    with pytest.raises(ValueError):
        audiate.crop(_score(), 3.0, 3.0)


def test_crop_roundtrips_through_a_midi_file(tmp_path):
    path = audiate.export(audiate.crop(_score(), 1.0, 3.0), tmp_path / "c.mid")
    back = pretty_midi.PrettyMIDI(str(path))
    assert len(back.instruments[0].notes) == 5


def test_export_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        audiate.export(_score(), tmp_path / "x.foo")
