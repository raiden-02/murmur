"""Real data for the quantization diagrams (docs/quantization.md), written to
data/quant_*.json:

- quant_importance.json: the measured importance of the input channels of
  one down_proj (from importance.bin, made by `murmur-lab calibrate`);
- quant_keys.json: each layer's largest key magnitude on the reference
  prompt (from the reference dumps).

Run with the reference environment, from the repository root:
  reference/.venv/Scripts/python.exe docs/diagrams/extract_quant_data.py
"""
import json
import os
import struct

import numpy as np
from safetensors.numpy import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODEL = os.path.join(ROOT, "models", "qwen2.5-1.5b-instruct")
DATA = os.path.join(HERE, "data")
IMPORTANCE_LAYER = 1   # the most uneven importance of the model (one channel 10^7 times the median)


def rounded(values, digits=4):
    """Values to `digits` significant digits, to keep the JSON small."""
    return [float(f"{v:.{digits}g}") for v in values]


def importance():
    path = os.path.join(MODEL, "importance.bin")
    with open(path, "rb") as f:
        blob = f.read()
    assert blob[:8] == b"MURIMP01"
    count = struct.unpack_from("<I", blob, 8)[0]
    at = 12
    found = {}
    for _ in range(count):
        n = struct.unpack_from("<I", blob, at)[0]
        name = blob[at + 4:at + 4 + n].decode()
        at += 4 + n
        m = struct.unpack_from("<I", blob, at)[0]
        found[name] = np.frombuffer(blob, dtype=np.float32, count=m, offset=at + 4)
        at += 4 + 4 * m
    values = found[f"layers.{IMPORTANCE_LAYER}.down"]
    return {"layer": IMPORTANCE_LAYER, "matrix": "down", "values": rounded(values)}


def keys():
    dump = load_file(os.path.join(ROOT, "reference", "out", "noodle_vendor.safetensors"))
    maxima = [float(np.abs(dump[f"layers.{layer}.k"]).max()) for layer in range(28)]
    return {"max_abs_key": maxima}


if __name__ == "__main__":
    for name, make in (("quant_importance.json", importance), ("quant_keys.json", keys)):
        with open(os.path.join(DATA, name), "w", encoding="utf-8") as f:
            json.dump(make(), f)
        print("wrote", name)
