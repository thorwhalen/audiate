# audiate

Render a symbolic music **score** to **audio**.

Give `audiate` a score in any of several standard formats — MIDI, MusicXML, ABC,
Humdrum-`kern`, MEI — and get back an audio waveform. One call:

```python
import audiate

audio = audiate.render("song.mid")            # -> AudioData
audio.write("song.wav")
```

The name is [*audiation*](https://en.wikipedia.org/wiki/Audiation) — hearing
music in your mind. `audiate` makes a written score actually audible.

## Install

```bash
pip install audiate                 # core: MIDI in, pure-Python "sine" synth out
pip install "audiate[symbolic]"     # + MusicXML / ABC / kern / MEI (music21)
pip install "audiate[fluidsynth]"   # + SoundFont synthesis (pyFluidSynth)
pip install "audiate[all]"          # everything
```

## Quick start

```python
import audiate

# Start from any standard format — it's auto-detected
audio = audiate.render("lead_sheet.musicxml")
audio = audiate.render("tune.abc")
audio = audiate.render("bach.krn")

# ...or from objects you already have
import pretty_midi
audio = audiate.render(pretty_midi.PrettyMIDI("song.mid"))

from music21 import corpus
audio = audiate.render(corpus.parse("bach/bwv66.6"))

# The result is an AudioData
audio.array            # float32 numpy waveform
audio.sample_rate      # e.g. 44100
audio.duration_seconds
audio.to_wav_bytes()   # WAV bytes
audio.write("out.wav") # write to disk
```

## How it works

The universal pivot is **MIDI**. Every input format is normalized to a
`pretty_midi.PrettyMIDI` (the *ingest* layer), then a pluggable *engine*
synthesizes it to audio. Adding one ingester or one engine multiplies coverage
across all the others.

```
score (MIDI / MusicXML / ABC / kern / MEI / …)
        │  ingest  (pretty_midi + music21)
        ▼
   pretty_midi.PrettyMIDI          ← the MIDI-in-the-middle pivot
        │  engine
        ▼
   AudioData (waveform + sample_rate)
```

### Engines (progressive disclosure)

| Engine | Quality | Needs | Use for |
|---|---|---|---|
| `sine` | basic | nothing (pure NumPy) | tests, CI, quick previews — always works |
| `fluidsynth` | good | FluidSynth + a SoundFont | the default once a SoundFont is present |
| `musescore` | best | the MuseScore CLI | notation-aware audio (dynamics, articulations, repeats) |

`engine="auto"` (the default) uses `fluidsynth` when a SoundFont is available,
otherwise falls back to `sine`.

```python
audio = audiate.render("song.mid", engine="fluidsynth", soundfont="MyPiano.sf2")
audio = audiate.render("score.musicxml", engine="musescore")   # highest fidelity
```

Register your own engine — it's an open registry:

```python
from audiate import register_engine

@register_engine("my_synth")
def my_synth(pm, *, sample_rate=44100, **opts):
    ...
    return waveform, sample_rate   # (np.ndarray, int)
```

## System dependencies

- **FluidSynth** (for the `fluidsynth` engine): `brew install fluid-synth` /
  `apt install fluidsynth`, plus a SoundFont (`.sf2`/`.sf3`). Point `audiate` at
  one with `soundfont=...` or `$AUDIATE_SOUNDFONT`; good free options include
  **MuseScore_General.sf3** (MIT) and **FluidR3_GM** (MIT).
- **MuseScore** (for the `musescore` engine): the desktop app's `mscore` binary.
  Set `$AUDIATE_MUSESCORE` if it isn't on your `PATH`.

## Where it fits

`audiate` is the **score → audio** stage of a larger pipeline:

- **search** a score by metadata → *(a search facade)*
- **`audiate`** — render that score to "MIDI audio"
- **[`arioso`](https://github.com/thorwhalen/arioso)** — turn that into
  AI-enhanced audio (`arioso.enhance`)

Chord-chart *arrangement* (chords → a backing band) is
[`accompy`](https://github.com/thorwhalen/accompy)'s job; `audiate` focuses on
rendering fully-specified symbolic scores.

## License

MIT
