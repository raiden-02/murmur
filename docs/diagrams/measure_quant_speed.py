"""Measures every weight format's speed with murmur-lab and writes
data/quant_speed.json for the quantization diagrams and report. Takes about
six minutes; run it on an otherwise idle GPU, from the repository root, after
a Release build:
  python docs/diagrams/measure_quant_speed.py
"""
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LAB = os.path.join(ROOT, "out", "build", "x64-release", "murmur-lab.exe")
FORMATS = ["bf16", "fp8", "int4"]


def lab(*args):
    result = subprocess.run([LAB, *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout


def numbers(line):
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", line.replace(",", ""))]


def decode(fmt):
    """Single-stream decode at 1,024 tokens of context: tokens/s."""
    rows = [numbers(l) for l in lab("decode", "1024", "--weights", fmt).splitlines() if l.strip().startswith("1024")]
    return rows[0][2]


def batch(fmt):
    """Prefill of one 512-token chunk (ms) and tokens/s for 1..64 conversations."""
    out = lab("batch", "--weights", fmt)
    prefill = numbers(next(l for l in out.splitlines() if l.startswith("prefill:")))[1]
    table = {}
    for l in out.splitlines():
        v = numbers(l)
        if len(v) == 6 and l.strip()[0].isdigit():
            table[int(v[0])] = {"ms_per_step": v[1], "tokens_per_second": v[2]}
    weights_gb = float(re.search(r"([\d.]+) GB", out).group(1))
    return weights_gb, prefill, table


def market(fmt):
    """30 NPCs talking for 60 s: total tokens/s and the player's time to first token."""
    out = lab("market", "--seconds", "60", "--weights", fmt)
    speed = next(l for l in out.splitlines() if l.startswith("steps"))
    ttft = re.search(r"p50 ([\d.]+) ms, p99 ([\d.]+) ms", out)
    return {"tokens_per_second": numbers(speed)[2], "ttft_p50_ms": float(ttft.group(1)),
            "ttft_p99_ms": float(ttft.group(2))}


if __name__ == "__main__":
    data = {"formats": FORMATS, "context": 1024, "prefill_tokens": 512}
    for fmt in FORMATS:
        weights_gb, prefill, table = batch(fmt)
        data[fmt] = {"weights_gb": weights_gb, "decode_tokens_per_second": decode(fmt), "prefill_ms": prefill,
                     "batch": table, "market": market(fmt)}
        print(fmt, data[fmt]["decode_tokens_per_second"], prefill, data[fmt]["market"])
    with open(os.path.join(HERE, "data", "quant_speed.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print("wrote data/quant_speed.json")
