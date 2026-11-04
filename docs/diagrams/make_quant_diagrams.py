"""Diagrams for docs/quantization.md. Real data comes from data/quant_*.json
(extract_quant_data.py, measure_quant_speed.py, and `murmur-lab quality`
for the INT4 quantizer's steps) and data/quality.json (`murmur-lab quality
--json`).

Run from anywhere: python docs/diagrams/make_quant_diagrams.py
"""
import json
import math
import os

from svg_kit import ACCENT, circle, fill_rect, line, path, svg, text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FORMAT_COLOR = {"bf16": ACCENT["gray"], "fp8": ACCENT["teal"], "int4": ACCENT["purple"]}
BUDGET_BYTES = 3584 * 2 ** 20   # the game's model memory: weights + KV cache


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def write(name, height, title, body):
    svg(HERE, name, height, title, body)


def upper(fmt):
    return fmt.upper()


def bytes_and_speed():
    """Each format's weights as a bar of bytes, next to the decode speed it
    gives: the bytes shrink as the speed grows, because one token reads every
    byte once."""
    speed = load("quant_speed.json")
    b = text(40, 22, "Fewer bytes per weight, more tokens per second", "th")
    b += text(110, 52, "bytes read for every token (the weights)", "tx")
    b += text(390, 52, "tokens per second, one conversation", "tx")
    top_gb = max(speed[f]["weights_gb"] for f in speed["formats"])
    top_tps = max(speed[f]["decode_tokens_per_second"] for f in speed["formats"])
    y = 70
    for fmt in speed["formats"]:
        gb, tps = speed[fmt]["weights_gb"], speed[fmt]["decode_tokens_per_second"]
        color = FORMAT_COLOR[fmt]
        b += text(96, y + 13, upper(fmt), "ts", "end")
        w = gb / top_gb * 200
        b += fill_rect(110, y, w, 26, color, 0.85, rx=3) + text(116 + w, y + 13, f"{gb:.2f} GB", "tx")
        w = tps / top_tps * 220
        b += fill_rect(390, y, w, 26, color, 0.85, rx=3) + text(396 + w, y + 13, f"{tps:.0f}", "tx")
        y += 40
    b += text(40, y + 14, "A decode step reads every weight once, so its time is about bytes ÷ bandwidth: half the bytes,", "ts")
    b += text(40, y + 32, "nearly twice the tokens. Measured on the RTX 4080 SUPER, 1,024 tokens of context.", "ts")
    write("quant_bytes_speed.svg", y + 50, "Bytes per weight and decode speed", b)


def importance_spike():
    """How strongly each of 8,960 input channels of one matrix is driven, on
    a log scale: almost all are quiet, a handful are millions of times
    louder. Then what weighting the rounding error by it buys."""
    data = load("quant_importance.json")
    values = data["values"]
    steps = load("quant_int4_steps.json")["steps"]
    X0, X1, Y0, Y1 = 70, 630, 50, 190
    floor = math.log10(max(min(v for v in values if v > 0), 1e-9))
    ceil = math.log10(max(values))
    b = text(40, 22, f"Which weights matter: how hard each input channel of layer {data['layer']}'s down_proj is driven", "th")
    columns = int(X1 - X0)
    per = len(values) / columns
    for c in range(columns):
        chunk = values[int(c * per):int((c + 1) * per)] or [values[-1]]
        v = max(chunk)
        h = (math.log10(max(v, 10 ** floor)) - floor) / (ceil - floor) * (Y1 - Y0)
        b += fill_rect(X0 + c, Y1 - h, 1, h, ACCENT["amber"] if v > 1000 * sorted(values)[len(values) // 2] else ACCENT["teal"], 0.9)
    for p in range(math.ceil(floor), math.floor(ceil) + 1, 2):
        yy = Y1 - (p - floor) / (ceil - floor) * (Y1 - Y0)
        b += line(X0 - 4, yy, X0, yy, None, 0.6) + text(X0 - 8, yy, f"10^{p}", "tx", "end")
    b += text(X0, Y1 + 14, "input channel 0", "tx") + text(X1, Y1 + 14, f"{len(values):,}", "tx", "end")
    b += text(40, Y1 + 40, "Mean x² per channel on calibration text (log scale). A weight in an amber column meets an activation", "ts")
    b += text(40, Y1 + 58, "thousands of times larger than usual, so its rounding error is multiplied by that much in the output.", "ts")

    y = Y1 + 92
    b += text(40, y, "INT4 quality, step by step (mean KL divergence from the original model; shorter is better)", "ts")
    top = max(s["kl_mean"] for s in steps)
    for k, s in enumerate(steps):
        yy = y + 18 + k * 28
        w = s["kl_mean"] / top * 360
        b += text(230, yy + 10, s["label"], "tx", "end")
        b += fill_rect(240, yy, w, 20, ACCENT["purple"], 0.4 + 0.25 * k, rx=3)
        b += text(246 + w, yy + 10, f"{s['kl_mean']:.3f}", "tx")
    write("importance_spike.svg", y + 18 + len(steps) * 28 + 10, "Channel importance and INT4 quality", b)


def quality_size():
    """Every served combination as a point: x is the model's memory for
    weights, y is how far its predictions drift from the original (log
    scale). Moving left costs quality; the drop to INT4 is the big one."""
    q = load("quality.json")
    served = [c for c in q["candidates"] if c["math"] == "fast"]
    X0, X1, Y0, Y1 = 90, 600, 50, 270
    lo, hi = -4.5, 0.0   # log10 KL

    def px(gb):
        return X0 + gb / 3.3 * (X1 - X0)

    def py(kl):
        return Y1 - (math.log10(max(kl, 10 ** lo)) - lo) / (hi - lo) * (Y1 - Y0)

    b = text(40, 22, "What each format costs: memory for the weights against drift from the original model", "th")
    for p in range(-4, 1):
        yy = py(10 ** p)
        b += line(X0, yy, X1, yy, None, 0.4, dash=True) + text(X0 - 6, yy, f"10^{p}", "tx", "end")
    for gb in (1, 2, 3):
        b += text(px(gb), Y1 + 14, f"{gb} GB", "tx", "middle")
    for c in served:
        x, y = px(c["gb"]), py(c["kl_mean"])
        color = FORMAT_COLOR[c["weights"]]
        if c["kv"] == "fp8":
            b += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="none" stroke="{color}" stroke-width="2"/>'
        else:
            b += circle(x, y, 7, color, 0.9)
        label = f"{upper(c['weights'])} + {upper(c['kv'])} cache: perplexity {c['perplexity_prose']:.1f}"
        right_edge = x > 430
        b += text(x - 12 if right_edge else x + 12, y + (12 if c["kv"] == "fp8" else -10), label, "tx",
                  "end" if right_edge else "start")
    b += text(40, (Y0 + Y1) / 2, "KL divergence", "tx", "middle", f' transform="rotate(-90 40 {(Y0 + Y1) / 2})"')
    b += text((X0 + X1) / 2, Y1 + 34, "weights in GPU memory", "ts", "middle")
    ref = q["reference_perplexity"]["prose"]
    b += text(40, Y1 + 62, f"Filled: BF16 cache; ring: FP8 cache. The original model's perplexity on the same text is {ref:.1f}.", "ts")
    b += text(40, Y1 + 80, "KL divergence: how surprised the original would be by each format's next-token odds (0 = identical).", "ts")
    write("quality_size.svg", Y1 + 98, "Quality against size", b)


def key_outliers():
    """The largest key in each of the 28 layers on a real prompt: the first
    two tower over the rest, which is why they keep BF16 in an FP8 cache."""
    keys = load("quant_keys.json")["max_abs_key"]
    fp8_cache = next(c for c in load("quality.json")["candidates"]
                     if (c["weights"], c["kv"], c["math"]) == ("bf16", "fp8", "exact"))
    X0, X1, Y0, Y1 = 70, 630, 50, 210
    top = max(keys)
    bw = (X1 - X0) / len(keys)
    b = text(40, 22, "Why the first two layers keep BF16 in an FP8 cache: their keys are far larger", "th")
    for i, k in enumerate(keys):
        h = k / top * (Y1 - Y0)
        color = ACCENT["amber"] if i < 2 else ACCENT["teal"]
        b += fill_rect(X0 + i * bw + 2, Y1 - h, bw - 4, h, color, 0.85, rx=2)
        if i < 2 or i == len(keys) - 1:
            b += text(X0 + i * bw + bw / 2, Y1 - h - 9, f"{k:.0f}", "tx", "middle")
    b += text(X0, Y1 + 14, "layer 0", "tx") + text(X1, Y1 + 14, "layer 27", "tx", "end")
    b += text(40, Y1 + 42, "The largest |key| per layer on the reference prompt. Layer 0's attention scores swing by ±140, so its", "ts")
    b += text(40, Y1 + 60, "attention picks single tokens; E4M3's 3 mantissa bits move those scores by up to 12, which changes", "ts")
    b += text(40, Y1 + 78, "the pick (mean KL 1.3 with every layer in FP8). With layers 0 and 1 in BF16 it is "
                           f"{fp8_cache['kl_mean']:.2g}.", "ts")
    write("key_outliers.svg", Y1 + 96, "Key magnitude per layer", b)


def bytes_per_page(kv):
    """KV-cache bytes per 16-token page, as PagedKvCache::BytesPerPage."""
    width, layers, heads = 256, 28, 2
    per_value = {"fp32": 4, "bf16": 2, "fp8": 1}

    def token(fmt):
        return 2 * (width * per_value[fmt] + (heads * 4 if fmt == "fp8" else 0))

    return 16 * sum(token("bf16" if kv == "fp8" and layer < 2 else kv) for layer in range(layers))


def memory_budget():
    """The game's model memory as one bar per configuration: the weights
    (left) and the KV cache that fills the rest. The smaller the weights and
    cached values, the more NPC memories fit."""
    speed = load("quant_speed.json")
    configs = (("bf16", "bf16"), ("fp8", "bf16"), ("int4", "bf16"), ("fp8", "fp8"), ("int4", "fp8"))
    X0, X1 = 150, 520
    b = text(40, 22, "One memory budget, 3.75 GB: what the weights leave holds the NPCs' memories", "th")
    y = 52
    for weights, kv in configs:
        weight_bytes = speed[weights]["weights_gb"] * 1e9
        pages = int((BUDGET_BYTES - weight_bytes) // bytes_per_page(kv))
        memories = pages * 16 // 960
        split = X0 + weight_bytes / BUDGET_BYTES * (X1 - X0)
        b += text(X0 - 8, y + 12, f"{upper(weights)} + {upper(kv)} cache", "ts", "end")
        b += fill_rect(X0, y, split - X0, 24, ACCENT["amber"], 0.85, rx=3)
        b += fill_rect(split, y, X1 - split, 24, ACCENT["teal"], 0.45 if kv == "bf16" else 0.8, rx=3)
        b += text(X1 + 10, y + 12, f"{memories} NPC memories", "ts")
        y += 36
    b += fill_rect(X0, y + 4, 12, 12, ACCENT["amber"], 0.85, rx=2) + text(X0 + 18, y + 10, "weights", "tx")
    b += fill_rect(X0 + 90, y + 4, 12, 12, ACCENT["teal"], 0.6, rx=2) + text(X0 + 108, y + 10, "KV cache", "tx")
    b += text(40, y + 38, "An NPC memory is 960 cached tokens (the shared lore, its persona, its latest turns). The market has", "ts")
    b += text(40, y + 56, "32 NPCs: with BF16 weights they don't all fit, and the quiet ones forget sooner.", "ts")
    write("memory_budget.svg", y + 74, "Memory budget per configuration", b)


def format_batch_curves():
    """Total tokens per second against conversations in the step, one line
    per weight format: the narrower format leads at every batch size, most
    of all where steps are bound by reading the weights."""
    speed = load("quant_speed.json")
    sizes = sorted(int(k) for k in speed["bf16"]["batch"])
    X0, X1, Y0, Y1 = 90, 600, 50, 250
    top = max(speed[f]["batch"][str(s)]["tokens_per_second"] for f in speed["formats"] for s in sizes) * 1.1

    def px(i):
        return X0 + i / (len(sizes) - 1) * (X1 - X0)

    def py(v):
        return Y1 - v / top * (Y1 - Y0)

    b = text(40, 22, "Batched decode: every format, 1 to 64 conversations", "th")
    step = 2000
    for v in range(0, int(top), step):
        b += line(X0, py(v), X1, py(v), None, 0.4, dash=True) + text(X0 - 6, py(v), f"{v:,}", "tx", "end")
    for i, s in enumerate(sizes):
        b += text(px(i), Y1 + 14, str(s), "tx", "middle")
    ends = []
    for fmt in speed["formats"]:
        pts = [(px(i), py(speed[fmt]["batch"][str(s)]["tokens_per_second"])) for i, s in enumerate(sizes)]
        b += path("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts), color=FORMAT_COLOR[fmt], width=2, arrow_end=False)
        for x, y in pts:
            b += circle(x, y, 3, FORMAT_COLOR[fmt])
        ends.append([pts[-1][1], fmt])
    ends.sort()
    for k in range(1, len(ends)):   # keep the end labels apart
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 13)
    for y, fmt in ends:
        b += text(X1 + 8, y, upper(fmt), "tx")
    b += text(40, (Y0 + Y1) / 2, "tokens/s", "tx", "middle", f' transform="rotate(-90 40 {(Y0 + Y1) / 2})"')
    b += text((X0 + X1) / 2, Y1 + 32, "conversations generating together", "ts", "middle")
    b += text(40, Y1 + 60, "Up to about 16 conversations a step is bound by reading the weights, so fewer bytes win almost in", "ts")
    b += text(40, Y1 + 78, "proportion. Towards 64 the matrix math starts to dominate and the lines draw closer.", "ts")
    write("format_batch_curves.svg", Y1 + 96, "Batched decode per format", b)


if __name__ == "__main__":
    for make in (bytes_and_speed, quality_size, importance_spike, format_batch_curves, memory_budget, key_outliers):
        make()
    print("quantization diagrams written")
