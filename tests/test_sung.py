"""Tests for the sung line of a score (``audiate.sung_line``) on tiny MIDI and MusicXML files."""

import mido
import pretty_midi
import pytest

import audiate
from audiate import SungNote, attach_lyrics, sung_line
from audiate.ingest import detect_format

TPB = 480  # ticks per beat


# --------------------------------------------------------------------------- MIDI fixtures


def note(begin, end, pitch, channel=0):
    """Note-on/off events at absolute ticks."""
    return [
        (begin, mido.Message("note_on", note=pitch, velocity=80, channel=channel)),
        (end, mido.Message("note_off", note=pitch, velocity=0, channel=channel)),
    ]


def lyric(tick, text):
    return [(tick, mido.MetaMessage("lyrics", text=text))]


def text(tick, s):
    return [(tick, mido.MetaMessage("text", text=s))]


def tempo(tick, bpm):
    return [(tick, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))]


def melody(pitches, *, beats=1, start=0, channel=0, gap=0):
    """Back-to-back notes, ``beats`` long, ``gap`` beats apart."""
    events, t = [], start
    for p in pitches:
        events += note(t, t + int(beats * TPB), p, channel)
        t += int((beats + gap) * TPB)
    return events


def write_midi(path, *tracks, type=1):
    """A MIDI file from tracks of ``(absolute_tick, message)`` events."""
    mid = mido.MidiFile(type=type, ticks_per_beat=TPB)
    for events in tracks:
        trk = mido.MidiTrack()
        last = 0
        for tick, msg in sorted(events, key=lambda e: (e[0], e[1].type == "note_on")):
            trk.append(msg.copy(time=tick - last))
            last = tick
        mid.tracks.append(trk)
    mid.save(str(path))
    return path


def rows(rows_, ndigits=6):
    """Rows of numbers rounded (exact comparison of nested tuples, any pytest)."""
    return [
        tuple(round(float(x), ndigits) if isinstance(x, float) else x for x in row)
        for row in rows_
    ]


def words(line):
    return [(n.word, n.starts_word) for n in line]


def syllables(line):
    return [n.syllable for n in line]


# --------------------------------------------------------------------------- MIDI: timing and lines


def test_midi_times_follow_the_tempo_map_like_the_render(tmp_path):
    path = write_midi(
        tmp_path / "t.mid",
        tempo(0, 120) + tempo(2 * TPB, 60),
        melody([60, 62, 64, 65]) + lyric(0, "a") + lyric(TPB, "b"),
    )
    line = sung_line(path)
    assert rows([(n.start, n.dur) for n in line]) == rows(
        [(0.0, 0.5), (0.5, 0.5), (1.0, 1.0), (2.0, 1.0)]
    )
    rendered = sorted(
        (n.start, n.end)
        for i in audiate.to_pretty_midi(str(path)).instruments
        for n in i.notes
    )
    assert rows([(n.start, n.end) for n in line]) == rows(rendered)
    assert [n.pitch for n in line] == [60, 62, 64, 65]


def test_midi_bytes_and_kar_extension(tmp_path):
    path = write_midi(
        tmp_path / "t.mid", melody([60, 62]) + lyric(0, "do") + lyric(TPB, "re")
    )
    assert words(sung_line(path.read_bytes())) == [("do", True), ("re", True)]
    assert detect_format("song.kar") == "midi"


def test_separate_tracks_are_never_merged_into_one_line(tmp_path):
    path = write_midi(
        tmp_path / "t.mid",
        tempo(0, 120),
        melody([72, 74]),
        melody([48], beats=2, channel=1),
    )
    with pytest.raises(ValueError, match=r"track=1 channel=1.*track=2 channel=2"):
        sung_line(path)
    assert [n.pitch for n in sung_line(path, track=1)] == [72, 74]
    assert [n.pitch for n in sung_line(path, track=2)] == [48]


def test_the_line_with_lyrics_on_its_track_is_chosen_and_drums_skipped(tmp_path):
    path = write_midi(
        tmp_path / "t.mid",
        tempo(0, 120),
        melody([48, 43], channel=1),
        melody([72, 74]) + lyric(0, "la") + lyric(TPB, "lo"),
        melody([36, 38, 36, 38], beats=0.5, channel=9),
    )
    line = sung_line(path)
    assert [n.pitch for n in line] == [72, 74]
    assert words(line) == [("la", True), ("lo", True)]


def test_a_type0_file_mixing_channels_needs_a_channel(tmp_path):
    path = write_midi(
        tmp_path / "t.mid",
        melody([72, 74]) + melody([48, 50], channel=2),
        type=0,
    )
    with pytest.raises(ValueError, match="several lines"):
        sung_line(path)
    assert [n.pitch for n in sung_line(path, channel=3)] == [48, 50]
    with pytest.raises(ValueError, match="1 to 16"):
        sung_line(path, channel=0)


def test_polyphony_is_refused_unless_a_chord_rule_is_given(tmp_path):
    chord = write_midi(
        tmp_path / "c.mid", melody([60]) + melody([64]) + melody([67], start=TPB)
    )
    with pytest.raises(ValueError, match="polyphonic: 2 notes start"):
        sung_line(chord)
    assert [n.pitch for n in sung_line(chord, chords="top")] == [64, 67]
    assert [n.pitch for n in sung_line(chord, chords="bottom")] == [60, 67]
    overlap = write_midi(
        tmp_path / "o.mid", note(0, 2 * TPB, 60) + note(TPB, 3 * TPB, 64)
    )
    with pytest.raises(ValueError, match="polyphonic at 0.50 s"):
        sung_line(overlap)


def test_a_slight_overlap_is_legato_cut_at_the_next_onset(tmp_path):
    path = write_midi(
        tmp_path / "t.mid", note(0, TPB + 40, 60) + note(TPB, 2 * TPB, 62)
    )
    first, second = sung_line(path)
    assert first.end == pytest.approx(second.start) and second.start == pytest.approx(
        0.5
    )


# --------------------------------------------------------------------------- MIDI: lyrics


def test_hyphenated_syllables_join_into_words(tmp_path):
    texts = ["Light", "a", "can-", "dle", "in", "the", "win", "-dow"]
    events = melody([60, 62, 64, 65, 67, 69, 71, 72])
    for i, t in enumerate(texts):
        events += lyric(i * TPB, t)
    line = sung_line(write_midi(tmp_path / "t.mid", events))
    assert words(line) == [
        ("Light", True),
        ("a", True),
        ("candle", True),
        ("candle", False),
        ("in", True),
        ("the", True),
        ("window", True),
        ("window", False),
    ]
    assert syllables(line) == ["Light", "a", "can", "dle", "in", "the", "win", "dow"]


def test_space_marked_word_ends_join_the_syllables_between(tmp_path):
    texts = ["Hel", "lo ", "big ", "wor", "ld "]
    events = melody([60, 62, 64, 65, 67])
    for i, t in enumerate(texts):
        events += lyric(i * TPB, t)
    line = sung_line(write_midi(tmp_path / "t.mid", events))
    assert words(line) == [
        ("Hello", True),
        ("Hello", False),
        ("big", True),
        ("world", True),
        ("world", False),
    ]


def test_one_stray_space_does_not_turn_hyphen_lyrics_into_one_word(tmp_path):
    texts = ["Light", "a ", "can-", "dle", "in", "the", "win-", "dow"]
    events = melody(range(60, 68))
    for i, t in enumerate(texts):
        events += lyric(i * TPB, t)
    line = sung_line(write_midi(tmp_path / "t.mid", events))
    assert [n.word for n in line] == [
        "Light",
        "a",
        "candle",
        "candle",
        "in",
        "the",
        "window",
        "window",
    ]


def test_a_missing_lyric_does_not_shift_the_later_ones(tmp_path):
    # Note 2 has no lyric: it is a melisma of 'two', and 'three' stays on note 3.
    events = (
        melody([60, 62, 64, 65])
        + lyric(0, "one")
        + lyric(TPB, "two")
        + lyric(3 * TPB, "three")
    )
    line = sung_line(write_midi(tmp_path / "t.mid", events))
    assert words(line) == [
        ("one", True),
        ("two", True),
        ("two", False),
        ("three", True),
    ]
    assert syllables(line) == ["one", "two", "", "three"]


def test_early_and_late_lyrics_go_to_their_own_note(tmp_path):
    # A legato line: 'b' comes 10 ticks before its note, while 'a' still sounds;
    # 'c' comes 100 ticks after its note's onset.
    events = (
        melody([60, 62, 64])
        + lyric(0, "a")
        + lyric(TPB - 10, "b")
        + lyric(2 * TPB + 100, "c")
    )
    line = sung_line(write_midi(tmp_path / "t.mid", events))
    assert syllables(line) == ["a", "b", "c"]


def test_a_lyric_on_no_note_or_two_on_one_note_is_an_error(tmp_path):
    stray = (
        melody([60, 62], gap=2)
        + lyric(0, "a")
        + lyric(2 * TPB, "lost")
        + lyric(3 * TPB, "b")
    )
    with pytest.raises(ValueError, match="'lost' at 1.00 s falls on no note"):
        sung_line(write_midi(tmp_path / "s.mid", stray))
    twice = melody([60, 62]) + lyric(0, "a") + lyric(100, "b")
    with pytest.raises(ValueError, match=r"Two lyrics \('a', 'b'\)"):
        sung_line(write_midi(tmp_path / "d.mid", twice))


def test_lyrics_on_the_conductor_track_and_on_several_tracks(tmp_path):
    path = write_midi(
        tmp_path / "t.mid",
        tempo(0, 120) + lyric(0, "do") + lyric(TPB, "re"),
        melody([60, 62]),
    )
    assert words(sung_line(path)) == [("do", True), ("re", True)]
    ambiguous = write_midi(
        tmp_path / "a.mid",
        lyric(0, "do"),
        melody([60, 62]),
        lyric(0, "la"),
    )
    with pytest.raises(ValueError, match=r"tracks \[0, 2\] without notes"):
        sung_line(ambiguous)


def test_text_events_of_a_plain_midi_file_are_never_sung(tmp_path):
    path = write_midi(
        tmp_path / "t.mid", melody([60, 62]) + text(0, "Verse") + text(TPB, "Chorus")
    )
    assert words(sung_line(path)) == [("", True), ("", True)]


def test_melismas_hold_the_word_until_a_rest(tmp_path):
    # 'Ah' over three notes, a rest, then a note with no lyric of its own.
    events = melody([60, 62, 64]) + note(4 * TPB, 5 * TPB, 65) + lyric(0, "Ah")
    line = sung_line(write_midi(tmp_path / "t.mid", events))
    assert words(line) == [("Ah", True), ("Ah", False), ("Ah", False), ("", True)]


def test_a_hyphen_joins_a_word_across_a_rest(tmp_path):
    events = melody([60, 62], gap=1) + lyric(0, "A-") + lyric(2 * TPB, "men")
    line = sung_line(write_midi(tmp_path / "t.mid", events))
    assert words(line) == [("Amen", True), ("Amen", False)]


def test_without_lyrics_only_the_notes_are_read(tmp_path):
    events = melody([60, 62]) + lyric(0, "a") + lyric(100, "b")  # two on one note
    line = sung_line(write_midi(tmp_path / "t.mid", events), with_lyrics=False)
    assert [(n.pitch, n.word) for n in line] == [(60, ""), (62, "")]


# --------------------------------------------------------------------------- karaoke


def karaoke_tracks(texts):
    words_track = (
        text(0, "@KMIDI KARAOKE FILE") + text(0, "@LENGL") + text(0, "@TTwinkle")
    )
    for i, t in enumerate(texts):
        words_track += text(i * TPB, t)
    return words_track


def test_karaoke_text_events_are_sung_with_their_marks(tmp_path):
    texts = ["\\Twin", "kle ", "twin", "kle ", "/lit", "tle ", "star"]
    path = write_midi(
        tmp_path / "song.kar",
        tempo(0, 120),
        karaoke_tracks(texts),
        melody(range(60, 67)),
        melody([48], beats=7, channel=1),
    )
    line = sung_line(path, track=2)
    assert [n.word for n in line] == [
        "Twinkle",
        "Twinkle",
        "twinkle",
        "twinkle",
        "little",
        "little",
        "star",
    ]
    assert [n.starts_word for n in line] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert syllables(line) == ["Twin", "kle", "twin", "kle", "lit", "tle", "star"]


def test_a_karaoke_header_marks_a_mid_file_too(tmp_path):
    path = write_midi(
        tmp_path / "song.mid",
        karaoke_tracks(["Hel", "lo ", "/you"]),
        melody([60, 62, 64]),
    )
    assert [n.word for n in sung_line(path)] == ["Hello", "Hello", "you"]


def test_karaoke_text_without_word_boundaries_is_not_guessed(tmp_path):
    path = write_midi(
        tmp_path / "song.kar",
        karaoke_tracks(["Twin", "kle", "twin", "kle"]),
        melody(range(60, 64)),
    )
    with pytest.raises(NotImplementedError, match="no word boundary"):
        sung_line(path)
    assert len(sung_line(path, with_lyrics=False)) == 4


def test_lyrics_events_win_over_karaoke_text(tmp_path):
    path = write_midi(
        tmp_path / "song.kar",
        karaoke_tracks(["\\wrong", " words"]),
        melody([60, 62]) + lyric(0, "right") + lyric(TPB, "ones"),
    )
    assert [n.word for n in sung_line(path)] == ["right", "ones"]


# --------------------------------------------------------------------------- MusicXML fixtures


def xml_note(
    step,
    octave=4,
    dur=1,
    *,
    lyric=None,
    syllabic="single",
    verse=1,
    voice=None,
    chord=False,
    grace=False,
    tie=None,
    rest=False,
    lyrics=None,
    extend=False,
):
    """One ``<note>``; ``lyrics`` is a list of ``(verse, text, syllabic)``."""
    parts = []
    if grace:
        parts.append("<grace/>")
    if chord:
        parts.append("<chord/>")
    parts.append(
        "<rest/>"
        if rest
        else f"<pitch><step>{step}</step><octave>{octave}</octave></pitch>"
    )
    if not grace:
        parts.append(f"<duration>{dur}</duration>")
    if tie:
        parts.append(f'<tie type="{tie}"/>')
    if voice is not None:
        parts.append(f"<voice>{voice}</voice>")
    if tie:
        parts.append(f'<notations><tied type="{tie}"/></notations>')
    all_lyrics = list(lyrics or [])
    if lyric is not None:
        all_lyrics.append((verse, lyric, syllabic))
    for number, txt, syl in all_lyrics:
        parts.append(
            f'<lyric number="{number}"><syllabic>{syl}</syllabic><text>{txt}</text>'
            + ("<extend/>" if extend else "")
            + "</lyric>"
        )
    return "<note>" + "".join(parts) + "</note>"


def xml_score(*parts, tempo_bpm=None):
    """A score-partwise document from ``(name, [measure_body, ...])`` parts.

    The tempo mark, if any, goes in the first part only.
    """
    header = "".join(
        f'<score-part id="P{i + 1}"><part-name>{name}</part-name></score-part>'
        for i, (name, _) in enumerate(parts)
    )
    body = []
    for i, (_, measures) in enumerate(parts):
        ms = []
        for k, m in enumerate(measures):
            attrs = ""
            if k == 0:
                attrs = (
                    "<attributes><divisions>1</divisions>"
                    "<time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
                )
                if i == 0 and tempo_bpm:
                    attrs += f'<direction><sound tempo="{tempo_bpm}"/></direction>'
            ms.append(f'<measure number="{k + 1}">{attrs}{m}</measure>')
        body.append(f'<part id="P{i + 1}">{"".join(ms)}</part>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?><score-partwise version="3.1">'
        f"<part-list>{header}</part-list>{''.join(body)}</score-partwise>"
    )


def save(tmp_path, xml, name="s.musicxml"):
    path = tmp_path / name
    path.write_text(xml)
    return path


# --------------------------------------------------------------------------- MusicXML


def test_musicxml_syllables_ties_rests_and_melismas(tmp_path):
    m1 = (
        xml_note("E", lyric="can", syllabic="begin")
        + xml_note("G", lyric="dle", syllabic="end")
        + xml_note("A", tie="start", lyric="light")
        + xml_note("A", tie="stop")
    )
    m2 = (
        xml_note("B", lyric="oh")
        + xml_note("C", 5)
        + xml_note("", rest=True)
        + xml_note("D", 5)
    )
    line = sung_line(save(tmp_path, xml_score(("Voice", [m1, m2]), tempo_bpm=120)))
    assert rows([(n.start, n.dur, n.pitch) for n in line]) == rows(
        [
            (0.0, 0.5, 64),
            (0.5, 0.5, 67),
            (1.0, 1.0, 69),
            (2.0, 0.5, 71),
            (2.5, 0.5, 72),
            (3.5, 0.5, 74),
        ]
    )
    assert words(line) == [
        ("candle", True),
        ("candle", False),
        ("light", True),
        ("oh", True),
        ("oh", False),
        ("", True),
    ]
    assert syllables(line) == ["can", "dle", "light", "oh", "", ""]


def test_the_tempo_of_another_part_times_the_sung_part(tmp_path):
    piano = [xml_note("C", 3, 4), xml_note("C", 3, 4)]
    voice = [
        xml_note("E", dur=2, lyric="hel", syllabic="begin")
        + xml_note("G", dur=2, lyric="lo", syllabic="end"),
        xml_note("A", dur=4, lyric="you"),
    ]
    path = save(tmp_path, xml_score(("Piano", piano), ("Voice", voice), tempo_bpm=60))
    line = sung_line(path)  # the only part with lyrics
    assert [n.start for n in line] == pytest.approx([0.0, 2.0, 4.0])
    rendered = sorted(
        n.start
        for i in audiate.to_pretty_midi(str(path)).instruments
        for n in i.notes
        if n.pitch >= 60
    )
    assert [n.start for n in line] == pytest.approx(rendered)


def test_one_voice_is_read_and_keeps_the_tempo(tmp_path):
    m = (
        xml_note("E", dur=2, voice=1, lyric="sing")
        + xml_note("G", dur=2, voice=1, lyric="it")
        + "<backup><duration>4</duration></backup>"
        + xml_note("C", dur=1, voice=2)
        + xml_note("D", dur=1, voice=2)
        + xml_note("E", dur=2, voice=2)
    )
    path = save(tmp_path, xml_score(("Voice", [m]), tempo_bpm=60))
    line = sung_line(path)
    assert rows([(n.start, n.pitch, n.word) for n in line]) == rows(
        [(0.0, 64, "sing"), (2.0, 67, "it")]
    )
    other = sung_line(path, voice=2)
    assert rows([(n.start, n.pitch) for n in other]) == rows(
        [(0.0, 60), (1.0, 62), (2.0, 64)]
    )
    with pytest.raises(ValueError, match="No voice 3 "):
        sung_line(path, voice=3)


def test_grace_notes_are_skipped_and_chords_refused_or_resolved(tmp_path):
    m = (
        xml_note("D", grace=True)
        + xml_note("E", lyric="a")
        + xml_note("G", lyric="b")
        + xml_note("C", 5, chord=True)
        + xml_note("A", dur=2, lyric="c")
    )
    path = save(tmp_path, xml_score(("Voice", [m]), tempo_bpm=120))
    with pytest.raises(ValueError, match="A chord in the sung line"):
        sung_line(path)
    assert [n.pitch for n in sung_line(path, chords="top")] == [64, 72, 69]
    assert [n.pitch for n in sung_line(path, chords="bottom")] == [64, 67, 69]


def test_the_chosen_verse_is_sung(tmp_path):
    m = (
        xml_note("E", lyrics=[(1, "one", "single"), (2, "two", "single")])
        + xml_note("G", lyrics=[(2, "more", "single")])  # verse 1 holds 'one' here
        + xml_note("A", dur=2, lyrics=[(1, "fish", "single"), (2, "fish", "single")])
    )
    path = save(tmp_path, xml_score(("Voice", [m])))
    assert words(sung_line(path)) == [("one", True), ("one", False), ("fish", True)]
    assert words(sung_line(path, verse=2)) == [
        ("two", True),
        ("more", True),
        ("fish", True),
    ]
    with pytest.raises(ValueError, match=r"No verse 3.*\[1, 2\]"):
        sung_line(path, verse=3)


def test_the_part_is_chosen_by_lyrics_name_or_index(tmp_path):
    soprano = [xml_note("E", 5, 4, lyric="ah")]
    alto = [xml_note("C", 5, 4, lyric="oh")]
    path = save(tmp_path, xml_score(("Soprano", soprano), ("Alto", alto)))
    with pytest.raises(
        ValueError, match=r"2 parts.*0 Soprano \(lyrics\), 1 Alto \(lyrics\)"
    ):
        sung_line(path)
    assert [n.word for n in sung_line(path, part="alto")] == ["oh"]
    assert [n.word for n in sung_line(path, part=0)] == ["ah"]
    with pytest.raises(ValueError, match="No part 5"):
        sung_line(path, part=5)


def test_an_elision_keeps_both_syllables(tmp_path):
    elided = (
        "<note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration>"
        '<lyric number="1"><syllabic>single</syllabic><text>the</text><elision/>'
        "<syllabic>single</syllabic><text>a</text></lyric></note>"
    )
    line = sung_line(save(tmp_path, xml_score(("Voice", [elided]))))
    assert [(n.syllable, n.word) for n in line] == [("the a", "the a")]


def test_midi_and_musicxml_of_one_melody_give_the_same_line(tmp_path):
    m = (
        xml_note("E", lyric="can", syllabic="begin")
        + xml_note("G", lyric="dle", syllabic="end")
        + xml_note("A", lyric="light")
        + xml_note("B")
    )
    xml_line = sung_line(save(tmp_path, xml_score(("Voice", [m]), tempo_bpm=90)))
    events = (
        tempo(0, 90)
        + melody([64, 67, 69, 71])
        + lyric(0, "can-")
        + lyric(TPB, "dle")
        + lyric(2 * TPB, "light")
    )
    midi_line = sung_line(write_midi(tmp_path / "t.mid", events))
    assert rows([(n.start, n.dur) for n in xml_line]) == rows(
        [(n.start, n.dur) for n in midi_line]
    )
    assert [(n.pitch, n.word, n.starts_word, n.syllable) for n in xml_line] == [
        (n.pitch, n.word, n.starts_word, n.syllable) for n in midi_line
    ]


def test_a_music21_stream_is_read_directly():
    from music21 import note as m21note, stream

    s = stream.Stream()
    for pitch, word in (("C4", "do"), ("D4", "re")):
        n = m21note.Note(pitch, quarterLength=1)
        n.lyric = word
        s.append(n)
    assert words(sung_line(s)) == [("do", True), ("re", True)]


# --------------------------------------------------------------------------- facade


def test_wrong_options_and_sources_say_so(tmp_path):
    path = write_midi(tmp_path / "t.mid", melody([60]))
    with pytest.raises(ValueError, match="part do"):
        sung_line(path, part=0)
    with pytest.raises(ValueError, match="track do"):
        sung_line(save(tmp_path, xml_score(("Voice", [xml_note("C", dur=4)]))), track=1)
    with pytest.raises(TypeError, match="PrettyMIDI"):
        sung_line(pretty_midi.PrettyMIDI(str(path)))
    with pytest.raises(FileNotFoundError):
        sung_line(tmp_path / "missing.mid")
    with pytest.raises(FileNotFoundError):
        sung_line(tmp_path / "missing.musicxml")
    with pytest.raises(ValueError, match="chords must be"):
        sung_line(path, chords="middle")


def test_attach_lyrics_spreads_and_holds_words():
    notes = [SungNote(float(i), 1.0, 60 + i) for i in range(4)]
    line = attach_lyrics(notes, "Light can-dle -")
    assert [(n.syllable, n.word, n.starts_word) for n in line] == [
        ("Light", "Light", True),
        ("can", "candle", True),
        ("dle", "candle", False),
        ("", "candle", False),
    ]
    with pytest.raises(ValueError, match="3 lyric syllables for 4 notes"):
        attach_lyrics(notes, "one two three")


def test_sung_notes_convert_field_for_field():
    # A singing back end's note type with the same leading fields takes one directly.
    n = SungNote(0.5, 0.25, 64, "candle", False, "dle")
    assert (n.start, n.dur, n.pitch, n.word, n.starts_word) == (
        0.5,
        0.25,
        64,
        "candle",
        False,
    )
    assert n.end == pytest.approx(0.75)


# --------------------------------------------------------------------------- review of PR 3


def lyric_line(texts, pitches=None, **kw):
    """A melody with one lyric text per note."""
    events = melody(pitches or range(60, 60 + len(texts)), **kw)
    for i, t in enumerate(texts):
        events += lyric(i * TPB, t)
    return events


def test_hyphens_decide_joins_once_a_file_uses_them(tmp_path):
    texts = ["Hap-", "py", "birth-", "day ", "to", "you ", "dear", "friend "]
    line = sung_line(write_midi(tmp_path / "t.mid", lyric_line(texts)))
    assert [n.word for n in line] == [
        "Happy",
        "Happy",
        "birthday",
        "birthday",
        "to",
        "you",
        "dear",
        "friend",
    ]


def test_repeats_are_sung_as_rendered_with_verse_n_on_pass_n(tmp_path):
    m1 = (
        "".join(
            xml_note(step, lyrics=[(1, w1, "single"), (2, w2, "single")])
            for step, w1, w2 in (("C", "one", "three"), ("D", "two", "four"))
        )
        + xml_note("E", dur=2)
        + '<barline location="right"><repeat direction="backward"/></barline>'
    )
    m2 = xml_note("G", dur=4, lyric="end")
    path = save(tmp_path, xml_score(("Voice", [m1, m2]), tempo_bpm=60))
    line = sung_line(path)
    rendered = sorted(
        n.start for i in audiate.to_pretty_midi(str(path)).instruments for n in i.notes
    )
    assert [n.start for n in line] == pytest.approx(rendered)
    assert [n.start for n in line] == pytest.approx([0, 1, 2, 4, 5, 6, 8])
    assert [n.word for n in line] == [
        "one",
        "two",
        "two",
        "three",
        "four",
        "four",
        "end",
    ]
    assert [n.word for n in sung_line(path, verse=1)][3:5] == ["one", "two"]


def test_a_lyric_on_a_grace_note_is_sung_on_the_next_note(tmp_path):
    m = (
        xml_note("C", lyric="the")
        + xml_note("D", grace=True, lyric="can", syllabic="begin")
        + xml_note("E", lyric="dle", syllabic="end")
        + xml_note("F", dur=2, lyric="x")
    )
    line = sung_line(save(tmp_path, xml_score(("Voice", [m]))))
    assert [(n.syllable, n.word) for n in line] == [
        ("the", "the"),
        ("candle", "candle"),
        ("x", "x"),
    ]


def test_an_end_syllable_joins_only_a_word_a_begin_opened(tmp_path):
    m = (
        xml_note("C", lyric="the")
        + xml_note("E", lyric="dle", syllabic="end")
        + xml_note("F", dur=2, lyric="x")
    )
    line = sung_line(save(tmp_path, xml_score(("Voice", [m]))))
    assert words(line) == [("the", True), ("dle", True), ("x", True)]


def test_an_early_lyric_during_a_melisma_goes_to_the_next_note(tmp_path):
    events = melody([60, 62, 64]) + lyric(0, "ah") + lyric(2 * TPB - 60, "men")
    line = sung_line(write_midi(tmp_path / "t.mid", events))
    assert words(line) == [("ah", True), ("ah", False), ("men", True)]
    # 'y' is as late for the note at 200 as it is early for the one at 400.
    unsure = lyric(0, "x") + note(0, 200, 60) + note(200, 400, 62) + note(400, 600, 64)
    with pytest.raises(ValueError, match="could be late"):
        sung_line(write_midi(tmp_path / "h.mid", unsure + lyric(300, "y")))


def test_a_bar_given_to_another_voice_is_silent(tmp_path):
    m1 = (
        "".join(xml_note("E", voice=1, lyric=w) for w in "abcd")
        + "<backup><duration>4</duration></backup>"
        + xml_note("C", dur=4, voice=2)
    )
    m2 = xml_note("C", dur=4, voice=2)
    path = save(tmp_path, xml_score(("Voice", [m1, m2])))
    assert [(n.start, n.pitch) for n in sung_line(path)] == [
        (0.0, 64),
        (0.5, 64),
        (1.0, 64),
        (1.5, 64),
    ]
    assert [n.pitch for n in sung_line(path, voice=2)] == [60, 60]
    from music21 import converter

    with pytest.raises(ValueError, match="Bar 2 has one voice"):
        sung_line(converter.parse(str(path)))


def test_an_explicit_track_never_borrows_another_lines_lyrics(tmp_path):
    vocal = melody([60, 62]) + lyric(0, "la") + lyric(TPB, "li")
    path = write_midi(tmp_path / "t.mid", vocal, melody([48, 50], channel=1))
    assert [n.word for n in sung_line(path, track=1)] == ["", ""]
    assert [n.word for n in sung_line(path, track=0)] == ["la", "li"]


def test_same_tick_karaoke_texts_keep_their_file_order(tmp_path):
    words_track = text(0, "@KMIDI KARAOKE FILE") + text(0, "\\") + text(0, "Twin")
    words_track += text(TPB, "kle ") + text(2 * TPB, "/Lit") + text(3 * TPB, "tle")
    path = write_midi(tmp_path / "s.kar", words_track, melody([60, 62, 64, 65]))
    assert [n.word for n in sung_line(path)] == [
        "Twinkle",
        "Twinkle",
        "Little",
        "Little",
    ]


def test_space_only_lyrics_mark_word_ends(tmp_path):
    events = melody([60, 62, 64]) + lyric(0, "Hel") + lyric(TPB, "lo")
    events += lyric(2 * TPB - 1, " ") + lyric(2 * TPB, "world")
    assert [n.word for n in sung_line(write_midi(tmp_path / "t.mid", events))] == [
        "Hello",
        "Hello",
        "world",
    ]


def test_a_slightly_spread_chord_is_one_chord(tmp_path):
    events = note(0, TPB, 64) + note(5, TPB, 60) + note(TPB, 2 * TPB, 62)
    path = write_midi(tmp_path / "t.mid", events)
    assert [n.pitch for n in sung_line(path, chords="top")] == [64, 62]
    assert [n.pitch for n in sung_line(path, chords="bottom")] == [60, 62]
    with pytest.raises(ValueError, match="polyphonic: 2 notes start"):
        sung_line(path)
    with pytest.raises(ValueError, match="polyphonic at"):
        sung_line(path, chords="top", chord_window_s=0)


def test_a_restruck_note_within_the_chord_window_is_one_note(tmp_path):
    events = [
        (0, mido.Message("note_on", note=60, velocity=80)),
        (5, mido.Message("note_on", note=60, velocity=80)),
        (2 * TPB, mido.Message("note_off", note=60, velocity=0)),
    ]
    line = sung_line(write_midi(tmp_path / "t.mid", events))
    assert rows([(n.start, n.dur, n.pitch) for n in line]) == rows([(0.0, 1.0, 60)])


def test_bar_level_notes_of_a_bar_with_voices_are_sung():
    from music21 import note as m21note, stream

    m = stream.Measure(number=1)
    v = stream.Voice(id="1")
    n = m21note.Note("E4", quarterLength=2)
    n.lyric = "hi"
    v.insert(0, n)
    m.insert(0, v)
    loose = m21note.Note("G4", quarterLength=2)
    loose.lyric = "there"
    m.insert(2, loose)
    p = stream.Part([m])
    line = sung_line(stream.Score([p]))
    assert [(n.pitch, n.word) for n in line] == [(64, "hi"), (67, "there")]


def test_an_elision_links_its_two_words(tmp_path):
    elided = (
        "<note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration>"
        '<lyric number="1"><syllabic>end</syllabic><text>dle</text><elision/>'
        "<syllabic>begin</syllabic><text>a</text></lyric></note>"
    )
    m = xml_note("C", lyric="can", syllabic="begin") + elided
    m += xml_note("G", dur=2, lyric="men", syllabic="end")
    line = sung_line(save(tmp_path, xml_score(("Voice", [m]))))
    assert [(n.syllable, n.starts_word) for n in line] == [
        ("can", True),
        ("dle a", False),
        ("men", False),
    ]
    assert {n.word for n in line} == {"candle amen"}


def test_the_lines_own_lyrics_come_before_a_words_track(tmp_path):
    path = write_midi(
        tmp_path / "t.mid",
        lyric(0, "wrong") + lyric(TPB, "words"),
        melody([60, 62]) + lyric(0, "right") + lyric(TPB, "ones"),
    )
    assert [n.word for n in sung_line(path)] == ["right", "ones"]


def test_two_lyrics_of_one_verse_on_a_chord_are_refused(tmp_path):
    m = xml_note("C", lyric="a") + xml_note("E", chord=True, lyric="b")
    m += xml_note("D", dur=3, lyric="c")
    with pytest.raises(ValueError, match="2 lyrics of verse 1 on one chord"):
        sung_line(save(tmp_path, xml_score(("Voice", [m]))), chords="top")


def test_an_extend_line_holds_the_word_across_a_rest(tmp_path):
    def score(extend):
        m = (
            xml_note("C", lyric="Ah", extend=extend)
            + xml_note("D")
            + xml_note("", rest=True)
            + xml_note("E")
        )
        return save(tmp_path, xml_score(("Voice", [m])), f"e{extend}.musicxml")

    assert [n.word for n in sung_line(score(True))] == ["Ah", "Ah", "Ah"]
    assert [n.word for n in sung_line(score(False))] == ["Ah", "Ah", ""]


def test_midi_timing_comes_from_audiates_own_ingest(tmp_path, monkeypatch):
    import audiate.sung as sung

    calls = []
    real = sung.to_pretty_midi
    monkeypatch.setattr(sung, "to_pretty_midi", lambda s: calls.append(s) or real(s))
    path = write_midi(tmp_path / "t.mid", melody([60]))
    sung_line(path)
    assert len(calls) == 1


def test_option_errors_are_grammatical(tmp_path):
    path = write_midi(tmp_path / "t.mid", melody([60]))
    with pytest.raises(ValueError, match="^part does not apply"):
        sung_line(path, part=0)
    with pytest.raises(ValueError, match="^part and verse do not apply"):
        sung_line(path, part=0, verse=1)
