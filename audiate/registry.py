"""Pluggable synthesis-engine registry.

An *engine* turns a ``pretty_midi.PrettyMIDI`` into audio. Engines are stored in
a :class:`EngineRegistry` (a ``MutableMapping`` of ``name -> callable``) so the
set is open for extension: register your own with :func:`register_engine`.

Each engine has the signature::

    engine(pm, *, sample_rate: int, **opts) -> tuple[np.ndarray, int]

returning the waveform and the sample rate it was actually rendered at (which
may differ from the requested rate -- e.g. an external tool with a fixed rate).
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping


class EngineRegistry(MutableMapping):
    """A ``MutableMapping`` of engine name -> engine callable."""

    def __init__(self):
        self._d: dict = {}

    def __getitem__(self, key):
        try:
            return self._d[key]
        except KeyError:
            raise KeyError(
                f"Unknown engine {key!r}. Registered engines: {sorted(self._d)}"
            ) from None

    def __setitem__(self, key, value):
        self._d[key] = value

    def __delitem__(self, key):
        del self._d[key]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)


#: The process-wide registry of synthesis engines.
engines = EngineRegistry()


def register_engine(name: str, func: Callable = None):
    """Register a synthesis engine under ``name`` (usable as a decorator).

    Example::

        @register_engine("my_synth")
        def my_synth(pm, *, sample_rate=44100, **opts):
            ...
            return array, sample_rate
    """
    if func is None:
        return lambda f: register_engine(name, f)
    engines[name] = func
    return func
