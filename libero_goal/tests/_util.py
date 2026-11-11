import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)


def have(*mods):
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in mods)
