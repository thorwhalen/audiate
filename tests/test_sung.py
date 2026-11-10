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
    with pytest.raises(
        ValueError, match=r"spread over several tracks \(track 0: 1, track 2: 1\)"
    ):
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
    staff=None,
    cue=False,
):
    """One ``<note>``; ``lyrics`` is a list of ``(verse, text, syllabic)`` or
    ``(verse, text, syllabic, extend)``: text or syllabic None leaves it out, an
    extend True or a type (``"stop"``) adds the ``<extend/>``."""
    parts = []
    if grace:
        parts.append("<grace/>")
    if cue:
        parts.append("<cue/>")
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
    if staff is not None:
        parts.append(f"<staff>{staff}</staff>")
    if tie:
        parts.append(f'<notations><tied type="{tie}"/></notations>')
    all_lyrics = [tuple(x) + (None,) * (4 - len(x)) for x in lyrics or []]
    if lyric is not None:
        all_lyrics.append((verse, lyric, syllabic, extend))
    for number, txt, syl, ext in all_lyrics:
        body = f"<syllabic>{syl}</syllabic>" if syl else ""
        body += f"<text>{txt}</text>" if txt is not None else ""
        if ext:
            body += "<extend/>" if ext is True else f'<extend type="{ext}"/>'
        parts.append(f'<lyric number="{number}">{body}</lyric>')
    return "<note>" + "".join(parts) + "</note>"


def xml_score(*parts, tempo_bpm=None):
    """A score-partwise document from ``(name, [measure_body, ...])`` parts, or
    ``(name, measures, staves)`` for a part on several staves.

    The tempo mark, if any, goes in the first part only.
    """
    header = "".join(
        f'<score-part id="P{i + 1}"><part-name>{p[0]}</part-name></score-part>'
        for i, p in enumerate(parts)
    )
    body = []
    for i, (_, measures, *staves) in enumerate(parts):
        ms = []
        for k, m in enumerate(measures):
            attrs = ""
            if k == 0:
                attrs = (
                    "<attributes><divisions>1</divisions>"
                    "<time><beats>4</beats><beat-type>4</beat-type></time>"
                    + (f"<staves>{staves[0]}</staves>" if staves else "")
                    + "</attributes>"
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


def test_the_track_holding_most_lyrics_is_the_lyric_track(tmp_path):
    path = write_midi(
        tmp_path / "t.mid",
        lyric(0, "Title"),
        melody([60, 62, 64])
        + lyric(0, "right")
        + lyric(TPB, "ones")
        + lyric(2 * TPB, "here"),
    )
    assert [n.word for n in sung_line(path)] == ["right", "ones", "here"]
    even = write_midi(
        tmp_path / "e.mid",
        lyric(0, "wrong") + lyric(TPB, "words"),
        melody([60, 62]) + lyric(0, "right") + lyric(TPB, "ones"),
    )
    with pytest.raises(ValueError, match="spread over several tracks"):
        sung_line(even)


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


def test_option_errors_are_grammatical(tmp_path):
    path = write_midi(tmp_path / "t.mid", melody([60]))
    with pytest.raises(ValueError, match="^part does not apply"):
        sung_line(path, part=0)
    with pytest.raises(ValueError, match="^part and verse do not apply"):
        sung_line(path, part=0, verse=1)


# --------------------------------------------------------------------------- review of PR 3, round 2

FORWARD = '<barline location="left"><repeat direction="forward"/></barline>'
BACKWARD = '<barline location="right"><repeat direction="backward"/></barline>'


def two_verses(a, b, sa="single", sb="single", ext=None):
    return [(1, a, sa, ext), (2, b, sb, ext)]


def grand_staff_bar():
    return (
        xml_note("C", 3, 4, voice=1, staff=1)
        + "<backup><duration>4</duration></backup>"
        + xml_note("C", 2, 4, voice=5, staff=2)
    )


def test_a_text_less_extend_stop_is_not_a_second_lyric(tmp_path):
    m = (
        xml_note("C", lyrics=two_verses("Glo", "Hail", "begin", "single", True))
        + xml_note("D", lyrics=[(1, None, None, "stop"), (2, None, None, "stop")])
        + xml_note("E", lyrics=two_verses("ri", "thee", "middle"))
        + xml_note("F", lyrics=two_verses("a", "now", "end"))
    )
    path = save(tmp_path, xml_score(("Voice", [m])))
    assert [n.word for n in sung_line(path)] == ["Gloria"] * 4
    assert [n.word for n in sung_line(path, verse=2)] == ["Hail", "Hail", "thee", "now"]


def test_two_lyrics_numbered_alike_on_one_note_sing_the_first_with_a_warning(tmp_path):
    m = xml_note("C", 4, 4, lyrics=[(1, "Verse", "single"), (1, "Chorus", "single")])
    with pytest.warns(UserWarning, match="2 lyrics numbered 1 on one note"):
        line = sung_line(save(tmp_path, xml_score(("Voice", [m]))))
    assert [n.word for n in line] == ["Verse"]


def test_a_grand_staff_part_beside_the_voice_keeps_voices_and_extends(tmp_path):
    held = (
        xml_note("C", voice=1, lyric="Ah", extend=True)
        + xml_note("D", voice=1)
        + xml_note("", rest=True, voice=1)
        + xml_note("E", voice=1)
    )
    divisi = (
        xml_note("F", dur=2, voice=1, lyric="la")
        + xml_note("G", dur=2, voice=1, lyric="lo")
        + "<backup><duration>4</duration></backup>"
        + xml_note("C", dur=4, voice=2)
    )
    piano = [grand_staff_bar(), grand_staff_bar()]
    path = save(tmp_path, xml_score(("Voice", [held, divisi]), ("Piano", piano, 2)))
    line = sung_line(path, part="Voice")
    assert [n.word for n in line] == ["Ah", "Ah", "Ah", "la", "lo"]
    assert [n.pitch for n in line] == [60, 62, 64, 65, 67]


def test_a_volta_with_only_verse_2_sings_it(tmp_path):
    end1 = '<barline location="left"><ending number="1" type="start"/></barline>'
    end1 += ""
    stop1 = '<barline location="right"><ending number="1" type="stop"/><repeat direction="backward"/></barline>'
    start2 = '<barline location="left"><ending number="2" type="start"/></barline>'
    stop2 = '<barline location="right"><ending number="2" type="stop"/></barline>'
    m1 = FORWARD + xml_note("C", dur=2, lyrics=two_verses("Praise", "Sing"))
    m1 += xml_note("D", dur=2, lyrics=two_verses("him", "it"))
    m2 = end1 + xml_note("E", dur=4, lyrics=[(1, "now", "single")]) + stop1
    m3 = start2 + xml_note("F", dur=4, lyrics=[(2, "loud", "single")]) + stop2
    m4 = xml_note("G", dur=4, lyrics=two_verses("Amen", "Amen"))
    path = save(tmp_path, xml_score(("Voice", [m1, m2, m3, m4]), tempo_bpm=60))
    assert [n.word for n in sung_line(path)] == [
        "Praise",
        "him",
        "now",
        "Sing",
        "it",
        "loud",
        "Amen",
    ]


def test_da_capo_sings_the_second_verse(tmp_path):
    fine = '<direction><direction-type><words>Fine</words></direction-type><sound fine="yes"/></direction>'
    dc = '<direction><direction-type><words>D.C. al Fine</words></direction-type><sound dacapo="yes"/></direction>'
    m1 = xml_note("C", dur=4, lyrics=two_verses("a", "c"))
    m2 = xml_note("D", dur=4, lyrics=two_verses("b", "d")) + fine
    m3 = xml_note("E", dur=4, lyric="mid") + dc
    path = save(tmp_path, xml_score(("Voice", [m1, m2, m3]), tempo_bpm=60))
    line = sung_line(path)
    assert [n.word for n in line] == ["a", "b", "mid", "c", "d"]
    rendered = sorted(
        n.start for i in audiate.to_pretty_midi(str(path)).instruments for n in i.notes
    )
    assert [n.start for n in line] == pytest.approx(rendered)


def test_musicxml_and_mxl_bytes_are_read(tmp_path):
    import zipfile

    m = xml_note("C", lyric="can", syllabic="begin") + xml_note(
        "D", dur=3, lyric="dle", syllabic="end"
    )
    xml = xml_score(("Voice", [m]))
    assert [n.word for n in sung_line(xml.encode())] == ["candle", "candle"]
    mxl = tmp_path / "s.mxl"
    with zipfile.ZipFile(mxl, "w") as z:
        z.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container><rootfiles>'
            '<rootfile full-path="score.musicxml"/></rootfiles></container>',
        )
        z.writestr("score.musicxml", xml)
    assert [n.word for n in sung_line(mxl)] == ["candle", "candle"]
    assert [n.word for n in sung_line(mxl.read_bytes())] == ["candle", "candle"]


def test_a_karaoke_file_with_accompaniment_finds_its_melody(tmp_path):
    words = text(0, "@LENGL") + text(0, "@TMy Song") + text(0, "\\Twin")
    words += text(TPB, "kle ") + text(2 * TPB, "twin") + text(3 * TPB, "kle ")
    path = write_midi(
        tmp_path / "real.kar",
        text(0, "@KMIDI KARAOKE FILE"),
        words,
        melody([60, 60, 67, 67]),
        melody([48, 43], beats=2, channel=1),
        melody([36] * 4, beats=0.1, gap=0.9, channel=9),
    )
    assert [n.pitch for n in sung_line(path)] == [60, 60, 67, 67]
    assert [n.word for n in sung_line(path)] == ["Twinkle"] * 2 + ["twinkle"] * 2


def test_a_type0_pop_file_needs_no_channel(tmp_path):
    vocal = melody([64, 67, 69, 67]) + sum(
        (lyric(i * TPB, w) for i, w in enumerate(["la", "li", "lo", "lu"])), []
    )
    bass = melody([40, 43], beats=2, channel=1)
    path = write_midi(tmp_path / "t.mid", tempo(0, 100) + vocal + bass, type=0)
    assert [n.pitch for n in sung_line(path)] == [64, 67, 69, 67]
    tie = write_midi(
        tmp_path / "tie.mid", melody([60, 62]) + melody([48, 50], channel=1), type=0
    )
    with pytest.raises(ValueError, match="several lines"):
        sung_line(tie)


def test_cue_notes_are_not_sung(tmp_path):
    m = (
        xml_note("C", lyric="one")
        + xml_note("G", 5, cue=True)
        + xml_note("G", 5, cue=True)
        + xml_note("D", lyric="two")
    )
    line = sung_line(save(tmp_path, xml_score(("Voice", [m]))))
    assert [(n.pitch, n.word) for n in line] == [(60, "one"), (62, "two")]


def test_a_title_lyric_before_the_music_can_be_skipped(tmp_path):
    events = sum(
        (note(4 * TPB + k * TPB, 5 * TPB + k * TPB, 60 + k) for k in range(2)), []
    )
    events += lyric(0, "Yesterday") + lyric(4 * TPB, "Yes-") + lyric(5 * TPB, "ter")
    path = write_midi(tmp_path / "t.mid", events)
    with pytest.raises(ValueError, match="on_stray_lyric='skip'"):
        sung_line(path)
    assert [n.word for n in sung_line(path, on_stray_lyric="skip")] == ["Yester"] * 2


def test_badly_formed_repeats_say_where(tmp_path):
    m1 = xml_note("C", dur=4, lyric="a")
    m2 = xml_note("D", dur=4) + BACKWARD
    m3 = FORWARD + xml_note("E", dur=4)
    with pytest.raises(
        ValueError, match=r"repeats are badly formed \(repeat marks in bars \[2, 3\]\)"
    ):
        sung_line(save(tmp_path, xml_score(("Voice", [m1, m2, m3]))))


def test_bar_level_notes_are_left_out_for_an_explicit_voice():
    from music21 import note as m21note, stream

    m = stream.Measure(number=1)
    v = stream.Voice(id="1")
    v.insert(0, m21note.Note("E4", quarterLength=2))
    m.insert(0, v)
    m.insert(2, m21note.Note("G4", quarterLength=2))
    score = stream.Score([stream.Part([m])])
    assert [n.pitch for n in sung_line(score, with_lyrics=False)] == [64, 67]
    assert [n.pitch for n in sung_line(score, voice=1, with_lyrics=False)] == [64]


def test_notes_the_xml_and_music21_disagree_on_warn_when_marks_are_lost():
    import xml.etree.ElementTree as ET
    from music21 import note as m21note, stream

    from audiate.sung import _mark_notes

    xm = ET.fromstring(
        "<measure>"
        + xml_note("C", lyric="Ah", extend=True)
        + xml_note("D")
        + "</measure>"
    )
    marks = [{"extend": {1}, "cue": False}, {"extend": set(), "cue": False}]
    with pytest.warns(UserWarning, match="do not agree on its notes"):
        _mark_notes(
            [m21note.Note("C4")], "1", xm, None, marks, stream.Measure(number=3)
        )


# --------------------------------------------------------------------------- review of PR 3, round 3


def harmony(root):
    return f"<harmony><root><root-step>{root}</root-step></root><kind>major</kind></harmony>"


def test_a_dense_accompaniment_meeting_every_lyric_is_not_the_melody(tmp_path):
    words = "Hel- lo dear world how are you now".split()
    lyrics = lyric(0, "My Song Title")
    lyrics += sum((lyric(TPB + i * TPB, w) for i, w in enumerate(words)), [])
    vocal = melody([64, 65, 66] * 2 + [64, 65], start=TPB)
    arpeggio = melody([48, 52, 56, 60] * 4 + [48, 52], beats=0.5, channel=1)
    path = write_midi(tmp_path / "t.mid", vocal + arpeggio + lyrics, type=0)
    line = sung_line(path, on_stray_lyric="skip")
    assert [n.pitch for n in line] == [64, 65, 66, 64, 65, 66, 64, 65]
    assert [n.word for n in line][:3] == ["Hello", "Hello", "dear"]


def test_two_lines_agreeing_alike_with_the_lyrics_are_the_callers_choice(tmp_path):
    events = melody([64, 65, 66, 67]) + melody([48, 50, 52, 53], channel=1)
    events += sum((lyric(i * TPB, w) for i, w in enumerate("a b c d".split())), [])
    with pytest.raises(ValueError, match=r"not clear.*meets 100%.*meets 100%"):
        sung_line(write_midi(tmp_path / "t.mid", events, type=0))


def test_a_credit_on_an_instrument_track_is_not_the_lyrics(tmp_path):
    words = text(0, "@TMy Song")
    words += sum(
        (
            text(i * TPB, w)
            for i, w in enumerate(
                ["\\Twin", "kle ", "twin", "kle ", "/lit", "tle ", "star"]
            )
        ),
        [],
    )
    path = write_midi(
        tmp_path / "s.kar",
        text(0, "@KMIDI KARAOKE FILE"),
        words,
        melody(range(60, 67)),
        melody([40] * 4, beats=2, channel=1) + text(0, "Bass: J. Smith"),
    )
    line = sung_line(path)
    assert [n.pitch for n in line] == list(range(60, 67))
    assert line[0].word == "Twinkle"


def test_one_stray_lyric_on_an_instrument_track_borrows_nothing(tmp_path):
    vocal = melody([60, 62, 64]) + sum(
        (lyric(i * TPB, w) for i, w in enumerate("do re mi".split())), []
    )
    bass = melody([40], beats=3, channel=1) + lyric(0, "oops")
    path = write_midi(tmp_path / "t.mid", vocal, bass)
    assert [n.word for n in sung_line(path)] == ["do", "re", "mi"]
    assert [n.word for n in sung_line(path, track=1)] == [""]


def test_a_non_numeric_lyric_number_keeps_its_extend(tmp_path):
    m = (
        xml_note("C", lyrics=[("part1verse1", "Ah", "single", True)])
        + xml_note("D")
        + xml_note("", rest=True)
        + xml_note("F", lyrics=[("part1verse1", "men", "single")])
    )
    line = sung_line(save(tmp_path, xml_score(("Voice", [m]))))
    assert [n.word for n in line] == ["Ah", "Ah", "men"]


def test_a_lead_sheet_with_chord_symbols_keeps_extends_and_skips_cues(tmp_path):
    m1 = harmony("C") + xml_note("C", lyric="Ah", extend=True) + xml_note("D")
    m1 += harmony("F") + xml_note("", rest=True) + xml_note("F")
    m2 = harmony("G") + xml_note("G", lyric="one") + xml_note("A", cue=True)
    m2 += xml_note("B", lyric="two") + xml_note("C", 5, lyric="three")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the XML and music21 agree on every note
        line = sung_line(save(tmp_path, xml_score(("Lead", [m1, m2]))))
    assert [(n.pitch, n.word) for n in line] == [
        (60, "Ah"),
        (62, "Ah"),
        (65, "Ah"),
        (67, "one"),
        (71, "two"),
        (72, "three"),
    ]


def test_a_verse_holding_its_word_is_not_borrowed_under_a_repeat(tmp_path):
    m1 = FORWARD + xml_note(
        "C", dur=2, lyrics=[(1, "Sing", "single"), (2, "Glo", "begin", True)]
    )
    m1 += xml_note("D", dur=2, lyrics=[(1, "we", "single")])
    m2 = xml_note("E", dur=2, lyric="now") + xml_note("F", dur=2, lyric="and")
    m3 = (
        xml_note("G", dur=4, lyrics=[(1, "ever", "single"), (2, "ri", "end")])
        + BACKWARD
    )
    line = sung_line(save(tmp_path, xml_score(("Voice", [m1, m2, m3]))))
    assert [n.word for n in line][5:] == ["Glori"] * 5
    assert [n.syllable for n in line][5:] == ["Glo", "", "", "", "ri"]


def test_a_bar_lacking_the_pass_verse_borrows_by_pass_not_the_lowest(tmp_path):
    m1 = FORWARD + xml_note(
        "C", dur=4, lyrics=[(1, "a", "single"), (2, "b", "single"), (3, "c", "single")]
    )
    m2 = (
        xml_note("D", dur=4, lyrics=[(1, "one", "single"), (3, "three", "single")])
        + BACKWARD
    )
    line = sung_line(save(tmp_path, xml_score(("Voice", [m1, m2]))))
    assert [n.word for n in line] == ["a", "one", "b", "three"]


def test_verse_takes_a_lyric_lines_name(tmp_path):
    m = xml_note("C", 4, 4, lyrics=[(1, "Verse", "single"), (1, "Chorus", "single")])
    m = m.replace(
        '<lyric number="1"><syllabic>single</syllabic><text>Chorus',
        '<lyric number="1" name="chorus"><syllabic>single</syllabic><text>Chorus',
    )
    path = save(tmp_path, xml_score(("Voice", [m])))
    assert [n.word for n in sung_line(path, verse="chorus")] == ["Chorus"]
    with pytest.raises(ValueError, match=r"named lyric lines \['chorus'\]"):
        sung_line(path, verse="bridge")


def test_a_voice_missing_from_a_closed_score_bar_warns(tmp_path):
    back = "<backup><duration>4</duration></backup>"
    m1 = "".join(
        xml_note(s, 5, voice=1, staff=1, lyric=w)
        for s, w in zip("EFGA", ["God", "is", "love", "so"])
    )
    m1 += back + "".join(xml_note(s, 5, voice=2, staff=1) for s in "CDEF")
    m1 += back + xml_note("G", 3, 4, voice=3, staff=2)
    m2 = (
        xml_note("G", 5, 4, voice=1, staff=1, lyric="true")
        + back
        + xml_note("C", 3, 4, voice=3, staff=2)
    )
    path = save(tmp_path, xml_score(("Choir", [m1, m2], 2)))
    with pytest.warns(UserWarning, match="voice 2 is silent there"):
        alto = sung_line(path, part=0, voice=2, with_lyrics=False)
    assert [n.pitch for n in alto] == [72, 74, 76, 77]


def test_a_grand_staff_piano_before_the_voice(tmp_path):
    held = (
        xml_note("C", voice=1, lyric="Ah", extend=True)
        + xml_note("D", voice=1)
        + xml_note("", rest=True, voice=1)
        + xml_note("E", voice=1)
    )
    path = save(
        tmp_path, xml_score(("Piano", [grand_staff_bar()], 2), ("Voice", [held]))
    )
    assert [n.word for n in sung_line(path, part="Voice")] == ["Ah", "Ah", "Ah"]


# --------------------------------------------------------------------------- review of PR 3, round 4


def sung_after_intro(tmp_path, name, *others, type=1):
    """A karaoke file: a words track, a melody with an instrumental intro, others."""
    sung = [32 * TPB + i * TPB for i in range(24)]
    words = ["\\Hap", "py ", "birth", "day ", "to ", "you"] * 4
    intro = melody([72 + (i % 5) for i in range(64)], beats=0.5)
    tune = sum((note(t, t + TPB, 64 + k % 5) for k, t in enumerate(sung)), [])
    tracks = [
        text(0, "@KMIDI KARAOKE FILE"),
        sum((text(t, w) for t, w in zip(sung, words)), []),
        intro + tune,
        *(f(sung) for f in others),
    ]
    return write_midi(tmp_path / name, *tracks)


def test_a_harmony_or_doubling_meeting_every_lyric_is_the_callers_choice(tmp_path):
    def third_above(sung):
        return sum(
            (note(t, t + TPB, 67 + k % 5, channel=1) for k, t in enumerate(sung)), []
        )

    def octave_above(sung):
        return sum(
            (note(t, t + TPB, 76 + k % 5, channel=1) for k, t in enumerate(sung)), []
        )

    for maker in (third_above, octave_above):
        path = sung_after_intro(tmp_path, f"{maker.__name__}.kar", maker)
        with pytest.raises(
            ValueError,
            match=r"track=2 channel=1 .*meets 100%.*track=3 channel=2 .*meets 100%",
        ):
            sung_line(path)
        assert sung_line(path, track=2)[64].pitch == 64


def test_a_chordal_accompaniment_is_not_a_candidate_line(tmp_path):
    def piano(sung):
        beats = range(56)
        return sum(
            (
                note(i * TPB, (i + 1) * TPB, p, channel=2)
                for i in beats
                for p in (48, 52, 55)
            ),
            [],
        )

    path = sung_after_intro(tmp_path, "p.kar", piano)
    line = sung_line(path)
    assert len(line) == 88 and line[64].word == "Happy" and line[64].pitch == 64


def test_a_stutter_hyphen_does_not_turn_a_space_file_into_a_hyphen_file(tmp_path):
    texts = ["Hap", "py ", "birth", "day ", "I- ", "I ", "love ", "you\r"]
    line = sung_line(write_midi(tmp_path / "t.mid", lyric_line(texts)))
    assert [n.word for n in line] == [
        "Happy",
        "Happy",
        "birthday",
        "birthday",
        "I",
        "I",
        "love",
        "you",
    ]


def test_the_reviewers_karaoke_shape_raises_naming_melody_and_harmony(tmp_path):
    # Words on a note-less text track; the melody track plays a 64-eighth intro
    # then the 24 sung notes; a harmony a third above sings only the sung notes;
    # a chordal piano; drums on channel 10.
    def harmony_line(sung):
        return sum(
            (note(t, t + TPB, 67 + k % 5, channel=1) for k, t in enumerate(sung)), []
        )

    def piano(sung):
        return sum(
            (
                note(i * TPB, (i + 1) * TPB, p, channel=2)
                for i in range(56)
                for p in (48, 52, 55)
            ),
            [],
        )

    def drums(sung):
        return sum((note(i * TPB, i * TPB + 100, 36, channel=9) for i in range(56)), [])

    path = sung_after_intro(tmp_path, "m1.kar", harmony_line, piano, drums)
    with pytest.raises(
        ValueError, match=r"track=2 channel=1 \(88 notes.*track=3 channel=2 \(24 notes"
    ) as e:
        sung_line(path)
    assert "channel=10" not in str(e.value)
    melody_line = sung_line(path, track=2)
    assert [n.pitch for n in melody_line[64:69]] == [64, 65, 66, 67, 68]
    assert {n.word for n in melody_line if n.word} == {"Happy", "birthday", "to", "you"}


@pytest.mark.parametrize("mark", ["\\", "/"])
@pytest.mark.parametrize("kind", ["karaoke", "lyrics"])
def test_a_line_mark_ends_the_word_before_it(tmp_path, mark, kind):
    texts = ["Hap", "py ", "birth", "day ", "to ", "you", mark + "Hap", "py"]
    events = melody(range(60, 68))
    if kind == "karaoke":
        events += sum((text(i * TPB, t) for i, t in enumerate(texts)), [])
        tracks = [text(0, "@KMIDI KARAOKE FILE"), events]
        name = "s.kar"
    else:
        events += sum((lyric(i * TPB, t) for i, t in enumerate(texts)), [])
        tracks = [events]
        name = "s.mid"
    line = sung_line(write_midi(tmp_path / name, *tracks))
    assert [n.word for n in line] == [
        "Happy",
        "Happy",
        "birthday",
        "birthday",
        "to",
        "you",
        "Happy",
        "Happy",
    ]
