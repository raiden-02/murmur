"""Diagrams for docs/comparison.md: the three engines' throughput, the GPU
time by kernel kind, one decode step by kernel and the launch gap between
steps. Real data comes from bench/compare/results/*.json (the three engines
on the same workload), data/profile_64.json (Nsight Systems, the 64-request
workload before and after the attention work), data/step_kernels.json
(Nsight Compute, one 64-row BF16 decode step) and data/launch_gap.json
(Nsight Systems, a typical step boundary of the 64-request workload).

Run from anywhere: python docs/diagrams/make_compare_diagrams.py
"""
import json
import math
import os

from svg_kit import ACCENT, circle, fill_rect, line, path, svg, text

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "bench", "compare", "results")
DATA = os.path.join(HERE, "data")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write(name, height, title, body):
    svg(HERE, name, height, title, body)


RUNS = [  # file, label, colour, dashed
    ("murmur_fp8", "Murmur FP8", ACCENT["teal"], False),
    ("vllm_fp8", "vLLM FP8", ACCENT["purple"], False),
    ("murmur_bf16", "Murmur BF16", ACCENT["teal"], True),
    ("vllm_bf16", "vLLM BF16", ACCENT["purple"], True),
    ("llamacpp_q8_0", "llama.cpp Q8_0", ACCENT["amber"], False),
]


def engine_throughput():
    """Tokens per second against requests at once (1, 8, 32, 64) for the same
    workload: a curve per engine and format. Where the curves cross is where
    one engine stops leading."""
    runs = [(load(os.path.join(RESULTS, f + ".json")), label, color, dashed) for f, label, color, dashed in RUNS]
    b = text(40, 22, "Same model, same GPU, same 64 requests: tokens per second as more run at once", "th")
    x0, x1, y0, y1 = 80, 520, 290, 50
    top = math.ceil(max(t["tokens_per_second"] for run, *_ in runs for t in run["throughput"]) / 1000) * 1000
    counts = [1, 8, 32, 64]

    def px(n):
        return x0 + math.log2(n) / 6 * (x1 - x0)

    def py(v):
        return y0 - v / top * (y0 - y1)

    for v in range(0, top + 1, 1000):
        b += line(x0, py(v), x1, py(v), "#8A887F33", 0.6) + text(x0 - 8, py(v), f"{v:,}", "tx", "end")
    for n in counts:
        b += text(px(n), y0 + 16, str(n), "tx", "middle")
    b += text((x0 + x1) / 2, y0 + 34, "requests at once", "ts", "middle")
    labels = []
    for run, label, color, dashed in runs:
        points = [(px(t["requests"]), py(t["tokens_per_second"])) for t in run["throughput"]]
        b += path("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in points), color, 2.0, dash=dashed, arrow_end=False)
        for x, y in points:
            b += circle(x, y, 3, color)
        last = run["throughput"][-1]["tokens_per_second"]
        labels.append([py(last), f"{label}  {last:,.0f}"])
    labels.sort()
    for i in range(1, len(labels)):   # keep the end labels at least a line apart
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 14)
    for y, label in labels:
        b += text(x1 + 10, y, label, "tx")
    write("engine_throughput.svg", 340, "Throughput against requests at once, three engines", b)


def gpu_time():
    """GPU time of the 64-request workload by kind of kernel, one bar per
    step of the work, to the same scale: the two attention segments shrink,
    the matrix multiplies stay."""
    data = load(os.path.join(DATA, "profile_64.json"))
    kinds = [("matrix multiplies", ACCENT["gray"]), ("FP8 activation scaling", "#B4B2A9"),
             ("decode attention", ACCENT["coral"]), ("prompt attention", ACCENT["red"]),
             ("everything else", "#D3D1C7")]
    longest = max(sum(r["ms"].values()) for r in data["runs"])
    scale = 420 / longest
    b = text(40, 22, "Where the GPU's time went at 64 requests (FP8), as the attention work went in", "th")
    y = 54
    for run in data["runs"]:
        b += text(40, y, f"{run['name']}: {sum(run['ms'].values()) / 1e3:.2f} s of GPU time, "
                         f"{run['tokens_per_second']:,} tokens/s", "ts")
        y += 12
        x = 40
        for kind, color in kinds:
            w = run["ms"][kind] * scale
            b += fill_rect(x, y, w, 22, color, 0.9)
            if kind.endswith("attention") and w > 30:
                b += text(x + w / 2, y + 11, f"{run['ms'][kind]:.0f} ms", "tx", "middle")
            x += w
        y += 44
    x = 40
    for kind, color in kinds:
        b += fill_rect(x, y - 5, 10, 10, color, 0.9) + text(x + 14, y, kind, "tx")
        x += 24 + len(kind) * 6.0
    write("gpu_time_64.svg", y + 20, "GPU time by kernel kind, before and after", b)


BIG = {"gate/up multiply": "gate/up", "down multiply": "down", "decode attention": "attention",
       "vocabulary": "vocab"}


def color_of(kind):
    if kind == "decode attention":
        return ACCENT["teal"]
    if "multiply" in kind or kind == "vocabulary":
        return ACCENT["gray"]
    return ACCENT["amber"]


def step_kernels():
    """One decode step as a strip: each kind of kernel as wide as its time,
    filled as high as the share of the memory bandwidth it reached. The empty
    space above the fill is time lost against the floor; the strip after the
    work is shorter where the small kernels were."""
    data = load(os.path.join(DATA, "step_kernels.json"))
    x0, x1 = 90, 600
    scale = (x1 - x0) / max(sum(k["us"] for k in data[s]) for s in ("before", "after"))
    b = text(40, 22, "One 64-row decode step: width = time, fill = share of the memory bandwidth reached", "th")
    for row, (state, label) in enumerate((("before", "before"), ("after", "after"))):
        top, height = 50 + row * 105, 60
        b += text(x0 - 10, top + height / 2, label, "ts", "end")
        x = x0
        small_from = None
        for k in data[state]:
            w = k["us"] * scale
            b += fill_rect(x, top, w, height, color_of(k["kind"]), 0.18)
            b += fill_rect(x, top + height * (1 - k["bandwidth"]), w, height * k["bandwidth"], color_of(k["kind"]))
            b += line(x, top, x, top + height, "#FFFFFF", 0.8)
            if k["kind"] in BIG:
                b += text(x + w / 2, top + height + 14, BIG[k["kind"]], "tx", "middle")
            elif small_from is None:
                small_from = x
            x += w
        b += line(small_from, top + height + 10, x, top + height + 10, ACCENT["amber"], 1.2)
        b += text((small_from + x) / 2, top + height + 24, "small kernels", "tx", "middle")
        total = sum(k["us"] for k in data[state]) / 1e3
        b += text(x + 6, top + height / 2, f"{total:.2f} ms", "ts")
    write("step_kernels.svg", 270, "One decode step by kernel, before and after", b)


def launch_lanes():
    """A typical step boundary on a time axis: the CPU's lane above the
    GPU's. With one graph the GPU idles through the whole launch; in pieces
    it starts after the first."""
    data = load(os.path.join(DATA, "launch_gap.json"))
    x0, x1 = 90, 640
    t0, t1 = -100, 700

    def px(t):
        return x0 + (min(max(t, t0), t1) - t0) / (t1 - t0) * (x1 - x0)

    b = text(40, 22, "The end of one decode step and the start of the next, 64 conversations", "th")
    for panel, state in enumerate(("before", "after")):
        d = data[state]
        top = 50 + panel * 120
        cpu_y, gpu_y, lane = top + 14, top + 50, 20
        b += text(40, top, d["label"], "ts")
        b += text(x0 - 10, cpu_y + lane / 2, "CPU", "tx", "end") + text(x0 - 10, gpu_y + lane / 2, "GPU", "tx", "end")
        b += line(x0, cpu_y + lane / 2, x1, cpu_y + lane / 2, "#8A887F55", 0.6)
        b += line(x0, gpu_y + lane / 2, x1, gpu_y + lane / 2, "#8A887F55", 0.6)
        for c in d["cpu"]:
            if c["to"] < t0:
                continue
            if c["kind"] == "launch":
                b += fill_rect(px(c["from"]), cpu_y, max(1.5, px(c["to"]) - px(c["from"]) - 1), lane, ACCENT["amber"])
            elif c["kind"] == "wait":
                b += fill_rect(px(c["from"]), cpu_y + 6, px(c["to"]) - px(c["from"]), lane - 12, ACCENT["gray"], 0.5)
            else:
                b += fill_rect(px(c["from"]), cpu_y + 3, max(1.0, px(c["to"]) - px(c["from"])), lane - 6,
                               ACCENT["gray"])
        for s, e in d["gpu"]:
            b += fill_rect(px(s), gpu_y, px(e) - px(s), lane, ACCENT["teal"])
        gap = d["gap_us"]
        b += line(px(0), gpu_y + lane + 6, px(gap), gpu_y + lane + 6, ACCENT["red"], 1.4)
        b += text((px(0) + px(gap)) / 2, gpu_y + lane + 19, f"GPU idle {gap:.0f} µs", "tx", "middle")
    for t in range(0, t1 + 1, 100):
        b += text(px(t), 296, f"{t}", "tx", "middle")
    b += text((x0 + x1) / 2, 312, "µs after the previous step's last kernel", "ts", "middle")
    b += fill_rect(x0, 322, 12, 10, ACCENT["amber"]) + text(x0 + 18, 328, "launching a graph", "tx")
    b += fill_rect(x0 + 150, 322, 12, 10, ACCENT["teal"]) + text(x0 + 168, 328, "kernels running", "tx")
    write("launch_lanes.svg", 345, "CPU and GPU at a step boundary, one graph against pieces", b)


if __name__ == "__main__":
    for make in (engine_throughput, gpu_time, step_kernels, launch_lanes):
        make()
    print("comparison diagrams written")
