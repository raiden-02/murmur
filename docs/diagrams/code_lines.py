"""Line ranges in src/ for the code pictures, found by text the way
tools/docs/resolve_tour.py finds the tour's links, so a picture's
"lines 307–325" labels follow the code when it moves. Each anchor must
appear exactly once in its file."""
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
_cache = {}


def line_of(path, anchor):
    """1-based line where `anchor` starts in src/`path` (a \\n in it spans lines)."""
    if path not in _cache:
        with open(os.path.join(SRC, path), encoding="utf-8") as f:
            _cache[path] = f.read().replace("\r\n", "\n")
    source = _cache[path]
    count = source.count(anchor)
    if count != 1:
        raise ValueError(f"{path}: {anchor!r} found {count} times")
    return source[: source.index(anchor)].count("\n") + 1


def lines(path, first, last=None):
    """'line n' or 'lines a–b', from the line of `first` to the line of `last`."""
    a = line_of(path, first)
    if last is None:
        return f"line {a}"
    b = line_of(path, last)
    return f"lines {a}–{b}"
