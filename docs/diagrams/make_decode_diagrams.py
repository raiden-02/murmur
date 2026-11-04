"""Diagrams for docs/decode_performance.md: decode speed stage by stage, the
final time budget and the roofline of the decode kernels. The numbers are the
report's measurements (`murmur-lab decode`, Nsight Systems).

Run from anywhere: python docs/diagrams/make_decode_diagrams.py
"""
import math
import os

from svg_kit import label_in, svg, text

HERE = os.path.dirname(os.path.abspath(__file__))
PEAK_TFLOPS = 52.2   # FP32 math on the RTX 4080 SUPER
PEAK_TBPS = 0.736    # memory bandwidth, TB/s
RIDGE = PEAK_TFLOPS / PEAK_TBPS


def write(name, height, title, body):
    svg(HERE, name, height, title, body)


def cell(x, y, w, h, ramp, label=""):
    """A filled box, with its label centred inside when there is one."""
    box = f'<rect class="b-{ramp}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" rx="3" stroke-width="0.5"/>'
    return box + (label_in(x, y, w, h, ramp, label) if label else "")


def stacked(y, label, segments, scale, x0=150):
    """One horizontal stacked bar: segments are (ms, ramp, text)."""
    s = text(40, y + 14, label, "th")
    x = x0
    for ms, ramp, t in segments:
        w = ms * scale
        s += cell(x, y, w - 2, 28, ramp, t if w > 60 else "")
        x += w
    return s


def log_axes(x0, x1, y0, y1, decades_x, decades_y):
    """Log-log plot mapping: returns px(FLOP/byte) and py(TFLOP/s)."""
    def px(v):
        return x0 + (math.log10(v) - decades_x[0]) / (decades_x[1] - decades_x[0]) * (x1 - x0)

    def py(v):
        return y1 - (math.log10(v) - decades_y[0]) / (decades_y[1] - decades_y[0]) * (y1 - y0)

    return px, py


def roofs(px, py, left):
    """The memory roof from `left` FLOP/byte up to the ridge, then the math roof."""
    return (f'<line x1="{px(left):.1f}" y1="{py(PEAK_TBPS * left):.1f}" x2="{px(RIDGE):.1f}" y2="{py(PEAK_TFLOPS):.1f}" '
            f'stroke="#378ADD" stroke-width="2"/>'
            f'<line x1="{px(RIDGE):.1f}" y1="{py(PEAK_TFLOPS):.1f}" x2="{px(1000):.1f}" y2="{py(PEAK_TFLOPS):.1f}" '
            f'stroke="#D85A30" stroke-width="2"/>')


def dot(x, y):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="b-amber" stroke-width="1"/>'


def frame(x0, x1, y0, y1):
    return f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" class="dash" stroke-width="0.8"/>'


def decode_stages():
    """Tokens per second at three context lengths after each optimization,
    as bars: BF16 doubles the short contexts, split-K the long ones."""
    b = text(40, 22, "Single-stream decode, stage by stage (idle GPU, best of 4 runs)", "th")
    stages = [("first engine: FP32", [88, 66, 38], [74, 56, 33]),
              ("+ BF16, fused matvec", [179, 109, 49], [75, 47, 22]),
              ("+ split-K attention", [185, 180, 171], [78, 77, 77]),
              ("+ CUDA graphs", [193, 188, 180], [81, 80, 81])]
    contexts = ["128", "1,024", "3,800"]
    ramps = ["purple", "teal", "amber"]
    b += text(40, 46, "tokens/s at context length:", "ts")
    for c, (label, ramp) in enumerate(zip(contexts, ramps)):
        b += cell(220 + c * 90, 36, 14, 14, ramp) + text(240 + c * 90, 43, label, "ts")
    for k, (name, tps, pct) in enumerate(stages):
        y = 66 + k * 64
        b += text(40, y + 22, name, "th")
        for c in range(3):
            w = tps[c] * 2.0
            b += cell(220, y + c * 15, w, 13, ramps[c])
            b += text(226 + w, y + c * 15 + 7, f"{tps[c]}  ({pct[c]}%)", "ts")
    b += text(40, 332, "% = share of the bandwidth ceiling: every weight + the KV cache read once per token at 736 GB/s.")
    b += text(40, 350, "BF16 doubled the ceiling, which exposed the old attention kernel at long contexts; split-K fixed that.")
    write("decode_stages.svg", 368, "Decode speed stage by stage", b)


def decode_budget_final():
    """One token's time as a stacked bar by kernel, against the ideal: all
    bytes at full bandwidth."""
    b = text(40, 22, "Where 5.32 ms per token goes now (1,024 tokens of context)", "th")
    segs = [(2.32, "teal", "gate/up 2.32"), (1.18, "teal", "down 1.18"), (0.68, "teal", "lm 0.68"),
            (0.64, "purple", "qkv+o"), (0.27, "amber", ""), (0.23, "gray", "")]
    b += stacked(50, "measured", segs, 100)
    b += stacked(90, "ideal", [(4.28, "gray", "4.28 ms: all bytes at 736 GB/s")], 100)
    b += text(40, 140, "teal: big matrix-vector products at 89–94% of peak bandwidth", "ts")
    b += text(40, 158, "purple: the two small ones (q/k/v 2,048 rows, o_proj 1,536) at ~65%: too short to reach full speed", "ts")
    b += text(40, 176, "amber: split attention 0.27 ms (29% of peak: tiny slices, 16 keys per block at this length)", "ts")
    b += text(40, 194, "gray: combine, RoPE, argmax, embedding 0.13 ms, and ~0.1 ms of host round trip per token", "ts")
    write("decode_budget_final.svg", 212, "Final decode time budget", b)


def decode_roofline():
    """The decode kernels on the roofline: all far left of the ridge, under
    the memory roof."""
    x0, x1, y0, y1 = 110, 600, 52, 272
    px, py = log_axes(x0, x1, y0, y1, (-1, 3), (-2, 2))   # 0.1 .. 1000 FLOP/byte, 0.01 .. 100 TFLOP/s
    b = text(40, 18, "Roofline of the decode kernels: all far left of the ridge, so bytes are what matter", "th")
    b += frame(x0, x1, y0, y1)
    for v, lab in [(0.1, "0.1"), (1, "1"), (10, "10"), (100, "100"), (1000, "1000")]:
        b += text(px(v), y1 + 14, lab, anchor="middle")
    for v, lab in [(0.01, "0.01"), (0.1, "0.1"), (1, "1"), (10, "10"), (100, "100")]:
        b += text(x0 - 8, py(v), lab, anchor="end")
    b += text((x0 + x1) / 2, y1 + 32, "FLOPs per byte read (log)", anchor="middle")
    b += text(x0 - 8, y0 - 12, "TFLOP/s", "ts", anchor="end")
    b += roofs(px, py, 0.1)
    b += text(px(RIDGE), py(PEAK_TFLOPS) + 16, "ridge ≈ 71", anchor="middle")
    points = [(1.0, 0.663, "gate/up, down, lm_head (~1 FLOP/B, 89–94%)", -12),
              (1.0, 0.473, "qkv, o_proj (~65%)", 10),
              (3.0, 0.65, "split attention (~3 FLOP/B, 29%)", 0)]
    for v, t, lab, dy in points:
        b += dot(px(v), py(t))
        b += text(px(v) + 10, py(t) + dy, lab)
    b += text(40, y1 + 56, "Batch-1 decode does ~1 multiply-add per weight loaded: 2 FLOPs per 2-byte BF16 weight. Its ceiling")
    b += text(40, y1 + 74, "is the memory roof (blue), and the gap between a dot and that roof is lost bandwidth. Batching many")
    b += text(40, y1 + 92, "conversations reuses each weight for several tokens and moves the dots to the right.")
    write("decode_roofline.svg", y1 + 110, "Roofline of the decode kernels", b)



if __name__ == "__main__":
    for make in (decode_stages, decode_budget_final, decode_roofline):
        make()
    print("decode diagrams written")
