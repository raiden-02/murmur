"""Diagrams for the ideas the engine is built on: tokenizing, attention and
the residual stream, the paged KV cache, the GPU kernels and what made them
fast, the engine thread, FP8 and INT4 weights, and actions under a grammar.
Each function is named after the SVG it writes.

Real data comes from data/bpe_merges.json, data/whole_picture.json,
data/residual_stream.json, data/logit_lens.json and data/sampling_stages.json
(Qwen2.5-1.5B-Instruct run with Hugging Face), data/quant_weights.json (a row
and a group of the model's weights with what FP8 and INT4 make of them),
data/actions.json (`murmur-lab actions --json`), data/action_logits.json
(extract_action_data.py), data/speculative_fp8.json (`murmur-lab speculative
--json`) and data/deadlock.json (the engine's state when the actions test
froze).

Run from anywhere: python docs/diagrams/make_study_diagrams.py
"""
import json
import math
import os
import re

from svg_kit import (ACCENT, WIDTH, arrow, chip, circle, esc, fill_rect, line, path, polygon, rect, svg, text)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
VERB_COLOR = {"none": ACCENT["gray"], "walk": ACCENT["teal"], "call": ACCENT["blue"], "give": ACCENT["green"],
              "sell": ACCENT["amber"], "quest": ACCENT["purple"], "hack": ACCENT["coral"]}
GRAMMAR_GRAY = "#C9C7BE"
MONO_W = 6.6   # width of one monospace character at 12px


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def write(name, height, title, body):
    svg(HERE, name, height, title, body)


def show_space(s):
    """Make a token's leading space visible."""
    return s.replace(" ", "␣")


def visible(token):
    """Make a token's spaces and newlines visible."""
    return token.replace(" ", "␣").replace("\n", "↵")


def char_width(s, size=12):
    """Rough rendered width of a string at the given font size."""
    return sum(0.62 if c.isalnum() else 0.45 for c in s) * size


def mono(x, y, s, cls="ts", anchor="start"):
    return text(x, y, s, cls + " mono", anchor)


def bpe_merges():
    """Byte-pair merges as a tree: leaves are bytes, each merge joins two
    neighbours into a parent; lower ranks (more common pairs) merge first."""
    data = load("bpe_merges.json")
    leaves = data["symbols"]
    unit = 62
    X0, base_y = 90, 330
    level_h = 40
    b = text(40, 22, "BPE builds “␣tonight” bottom-up from its bytes", "th")
    nodes = {}   # (start, end) -> (x, y, text)
    for k, ch in enumerate(leaves):
        x = X0 + k * unit + unit / 2
        shown = "␣" if ch == "Ġ" else ch
        nodes[(k, k + 1)] = (x, base_y, shown)
        b += chip(x - 18, base_y - 12, 36, 24, "gray", shown, cls="th", mono=True)
    for step, m in enumerate(data["merges"], start=1):
        left = next(s for s in nodes if s[0] == m["start"] and s[1] < m["end"] and (s[1], m["end"]) in nodes)
        right = (left[1], m["end"])
        lx, ly, lt = nodes[left]
        rx, ry, rt = nodes[right]
        x = (lx + rx) / 2
        y = base_y - step * level_h
        textv = lt + rt
        nodes[(m["start"], m["end"])] = (x, y, textv)
        del nodes[left], nodes[right]
        for cx, cy in ((lx, ly), (rx, ry)):
            b += line(cx, cy - 12, x, y + 12, ACCENT["purple"], 1.2)
        last = step == len(data["merges"])
        w = char_width(textv, 14) + 18
        b += chip(x - w / 2, y - 12, w, 24, "amber" if last else "teal", textv, cls="th", mono=True)
        b += text(x + w / 2 + 6, y, f"rank {m['rank']:,}", "tx")
    b += text(40, base_y + 34, "Leaves: the 8 bytes. Each parent: one merge from the learned list; its rank is its position there (lower = merged", "ts")
    b += text(40, base_y + 52, f"earlier, because more common in training text). The root is token {data['id']:,}.", "ts")
    write("bpe_merges.svg", base_y + 70, "BPE merges", b)


def paged_kv():
    """Three conversations' tokens cut into 16-token pages, scattered through
    one pool; each conversation's page table says where its pages are."""
    b = text(40, 22, "Paged KV cache: each conversation's pages live anywhere in one shared pool", "th")
    convos = [("Mei", 5, "coral", [17, 3, 41, 8, 26]), ("Kenji", 3, "teal", [5, 30, 12]),
              ("you", 4, "purple", [44, 0, 21, 35])]
    X0, cw = 110, 30
    y = 56
    for name, pages, ramp, table in convos:
        b += text(X0 - 10, y + 10, name, "ts", "end")
        for p in range(pages):
            b += fill_rect(X0 + p * cw, y, cw - 3, 20, ACCENT[ramp], 0.85, rx=2)
            b += text(X0 + p * cw + (cw - 3) / 2, y + 10, str(p), "tx", "middle", ' style="fill:#fff"')
        b += text(X0 + pages * cw + 8, y + 10, "page table [" + ", ".join(map(str, table)) + "]", "tx")
        y += 30
    b += text(X0, y + 4, "logical pages of 16 tokens each, in order", "ts")

    # The physical pool: 48 pages, 12 x 4.
    PX, PY, pc = 425, 56, 20
    owner = {}
    for name, _, ramp, table in convos:
        for logical, physical in enumerate(table):
            owner[physical] = (ramp, logical)
    for page in range(48):
        x = PX + (page % 12) * pc
        yy = PY + (page // 12) * pc
        if page in owner:
            ramp, logical = owner[page]
            b += fill_rect(x, yy, pc - 3, pc - 3, ACCENT[ramp], 0.85, rx=2)
            b += text(x + (pc - 3) / 2, yy + (pc - 3) / 2, str(logical), "tx", "middle", ' style="fill:#fff"')
        else:
            b += fill_rect(x, yy, pc - 3, pc - 3, ACCENT["gray"], 0.18, rx=2)
    b += text(PX, PY + 4 * pc + 12, "one pool: 48 pages here, 1,170 in the game", "ts")
    # Mei's table, drawn as arrows from two logical pages.
    for logical, physical in ((0, 17), (2, 41)):
        sx = X0 + logical * cw + (cw - 3) / 2
        tx = PX + (physical % 12) * pc + (pc - 3) / 2
        ty = PY + (physical // 12) * pc
        b += path(f"M{sx} 56 C {sx} 30, {tx} 30, {tx} {ty - 1}", color=ACCENT["coral"], width=1.0)
    b += text(40, 186, "Position p of a conversation lives at row  page_table[p / 16] × 16 + p mod 16  of the pool.", "ts")
    b += text(40, 204, "Memory is handed out a page at a time, so nothing is reserved for tokens that never come.", "ts")
    write("paged_kv.svg", 222, "Paged KV cache", b)


def pair_midpoints(centers):
    return [(centers[i] + centers[i + 1]) / 2 for i in range(0, len(centers), 2)]


def reduction_levels():
    """A tiny sum (1+2+…+8) as a pyramid: each sum is centred under the pair
    it replaces, so the rows narrow from 8 values to 4, 2 and 1."""
    size, gap = 44, 14
    pitch = size + gap
    level = list(range(1, 9))
    x0 = (WIDTH - ((len(level) - 1) * pitch + size)) / 2
    centers = [x0 + i * pitch + size / 2 for i in range(len(level))]
    b = text(340, 22, "Each row is one reduction step. The full set is on top.", "th", "middle")
    y = 44
    for label, ramp in zip(("finest", "coarser", "coarser", "coarsest"), ("purple", "teal", "teal", "amber")):
        b += text(24, y + 16, label, "ts")
        for v, cx in zip(level, centers):
            b += chip(cx - size / 2, y, size, 32, ramp, str(v), cls="th", rx=5)
        if len(level) == 1:
            y += 46
            break
        centers = pair_midpoints(centers)
        for mid in centers:
            b += path(f"M{mid} {y + 34} L{mid} {y + 66}")
        y += 68
        level = [level[i] + level[i + 1] for i in range(0, len(level), 2)]
    b += text(40, y, "Each step adds neighbours and keeps half as many. 1+2 is 3, and the pairs combine down to 36.")
    b += text(40, y + 18, "Coarsening is that first add inside one thread: 16 values become 1, then the block's tree runs.")
    b += text(40, y + 36, "One real pass keeps a single sum per block, so 64 Mi values become 16,384, then 4, then 1.")
    write("reduction_levels.svg", y + 54, "Reduction steps from the full set down to one total", b)


def rms_norm():
    """One token's vector entering three layers, before and after RMSNorm,
    the same 48 dimensions on one scale per side. The vector grows layer by
    layer; dividing by its RMS gives every layer the same typical size and
    keeps the pattern. Layers 4, 12 and 20 are ones where these 48 dimensions
    are typical of the whole vector (late layers are dominated by a few
    others)."""
    stream = load("residual_stream.json")
    hidden = 1536
    rows = []
    for layer in (4, 12, 20):
        normalized = stream["values"][layer]
        rms = stream["size"][layer] / math.sqrt(hidden)
        rows.append((layer, rms, [v * rms for v in normalized], normalized))
    raw_peak = max(abs(v) for _, _, raw, _ in rows for v in raw)
    norm_peak = max(abs(v) for _, _, _, norm in rows for v in norm)

    b = text(40, 22, "RMSNorm: the same vector, divided by its own size", "th")
    left, right, width, half = 110, 400, 250, 34
    b += text(left, 50, "entering the layer", "ts") + text(right, 50, "after RMSNorm (before the learned weight)", "ts")
    bar = width / len(rows[0][2])
    for i, (layer, rms, raw, norm) in enumerate(rows):
        mid = 100 + i * (2 * half + 22)
        b += text(left - 12, mid - 6, f"layer {layer}", "ts", "end")
        b += text(left - 12, mid + 10, f"RMS {rms:.2f}", "tx", "end")
        for x0, values, peak in ((left, raw, raw_peak), (right, norm, norm_peak)):
            b += line(x0, mid, x0 + width, mid, None, 0.5)
            for k, v in enumerate(values):
                h = abs(v) / peak * half
                color = ACCENT["teal"] if v >= 0 else ACCENT["coral"]
                b += fill_rect(x0 + k * bar + 0.5, mid - h if v >= 0 else mid, bar - 1, h, color)
    bottom = 100 + len(rows) * (2 * half + 22) - 10
    b += text(40, bottom, "48 of the 1,536 numbers of one real token, each side on one shared scale. On the left the", "ts")
    b += text(40, bottom + 18, "vector grows layer by layer; on the right each row is divided by its RMS: same size, same pattern.", "ts")
    write("rms_norm.svg", bottom + 36, "RMSNorm rescales each vector", b)


def matvec_latency():
    """One lane's timeline with loads issued one at a time and four at once,
    then the bandwidth the matrix-vector kernel reached at each change."""
    b = text(40, 22, "Why batching loads matters: one lane's timeline", "th")
    b += text(40, 50, "one load at a time", "th")
    x = 190
    for _ in range(4):
        b += chip(x, 38, 20, 22, "purple") + chip(x + 20, 38, 90, 22, "gray", "wait ~500 ns")
        x += 112
    b += text(40, 92, "4 loads issued together", "th")
    for k in range(4):
        b += chip(190 + k * 22, 80, 20, 22, "purple")
    b += chip(280, 80, 90, 22, "gray", "wait ~500 ns") + chip(372, 80, 60, 22, "teal", "4 × math")
    b += text(40, 128, "purple = issue a 16-byte load, gray = wait for memory, teal = multiply-adds", "ts")
    b += text(40, 156, "Memory latency is fixed, so throughput comes from overlap: keep many loads in flight (Little's law).", "ts")
    b += text(40, 174, "Issuing a lane's next loads before using the first ones turned 4 waits into 1.", "ts")
    b += text(40, 206, "Measured on the model's shapes, on a GPU shared with other programs:", "th")
    rows = [("before", 202, "coral", "202 GB/s (27%)"), ("loads batched", 452, "amber", "452 GB/s (61%)"),
            ("+ x via cache, rows/warp by size", 609, "teal", "609 GB/s (83%)")]
    x0, width = 250, 380
    for k, (label, gbs, ramp, value_text) in enumerate(rows):
        y = 222 + k * 34
        w = max(3, gbs / 736 * width)
        b += text(40, y + 14, label) + rect(x0, y, w, 28, ramp, rx=4)
        if w > width * 0.55:
            b += text(x0 + 10, y + 14, value_text, f"th k-{ramp}")
        else:
            b += text(x0 + w + 8, y + 14, value_text, "th")
    write("matvec_latency.svg", 330, "Loads in flight in the matrix-vector kernel", b)


def rope_pairs():
    """Two columns. Left is the pair before RoPE, right is the pair after.
    An arrow joins each input dot to its output dot. Values cross the same
    way and do not turn. Angles are the model's, at position 2. Every input
    is drawn at (1, 0) so the output arc equals the angle."""
    position = 2.0
    d = 128

    def theta(i):
        return position * (1e6 ** (-2 * i / d))

    b = text(40, 22, "RoPE rotates pairs of numbers in Q and K. V is not rotated.", "th")
    b += text(40, 42, "One head at position 2. Each circle is one pair, drawn at (1, 0) before RoPE.", "ts")
    r = 26
    in_x, out_x = 280, 530
    rows = (
        ("pair 0", 0, True),
        ("pair 1", 1, True),
        ("pair 8", 8, True),
        ("V", None, False),
    )
    first_cy = 156
    gap = 72
    b += text(in_x, 80, "Q and K before RoPE", "th", "middle", f' style="fill:{ACCENT["amber"]}"')
    b += text(out_x, 80, "Q and K after RoPE", "th", "middle", f' style="fill:{ACCENT["teal"]}"')

    def at(cx, cy, ang, radius):
        return cx + radius * math.cos(ang), cy - radius * math.sin(ang)

    for k, (name, index, turn) in enumerate(rows):
        cy = first_cy + k * gap
        ang = theta(index) if turn else 0.0
        b += text(36, cy, name, "ts", "start")
        for cx in (in_x, out_x):
            b += line(cx - r - 4, cy, cx + r + 4, cy, None, 0.5, dash=True)
            b += line(cx, cy - r - 4, cx, cy + r + 4, None, 0.5, dash=True)
            b += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{ACCENT["gray"]}" stroke-width="1.2"/>'
        ix, iy = at(in_x, cy, 0.0, r)
        b += circle(ix, iy, 4.5, ACCENT["amber"])
        if turn:
            steps = 24
            wedge = [(out_x, cy)] + [at(out_x, cy, ang * s / steps, r - 1) for s in range(steps + 1)]
            b += polygon(wedge, ACCENT["teal"], 0.30)
            ox, oy = at(out_x, cy, ang, r)
            b += circle(ox, oy, 4.5, ACCENT["teal"])
            b += arrow(in_x + r + 10, cy, out_x - r - 10, cy, ACCENT["teal"], 1.4)
            b += text((in_x + out_x) / 2, cy - 12, f"{ang:.2f} rad", "tx", "middle")
        else:
            ox, oy = at(out_x, cy, 0.0, r)
            b += circle(ox, oy, 4.5, ACCENT["amber"])
            b += arrow(in_x + r + 10, cy, out_x - r - 10, cy, ACCENT["gray"], 1.4)
            b += text((in_x + out_x) / 2, cy - 12, "V is not rotated", "tx", "middle")
    b += text(in_x + 12, first_cy + 12, "a", "tx")
    b += text(in_x - r - 6, first_cy - r, "b", "tx", "end")
    b += text(40, first_cy + 3 * gap + r + 20, "The three arrows are pairs inside Q and inside K. Pair 0 turns the most.", "ts")
    b += text(40, first_cy + 3 * gap + r + 38, "The last arrow is V. RoPE leaves it as it is.", "ts")
    write("rope_pairs.svg", first_cy + 3 * gap + r + 58, "RoPE rotates query and key pairs", b)


def attention_softmax():
    """One real attention row as two lines. The stored numbers are the weights.
    Softmax ignores a shift of every score, so the top line is the gaps that
    produce those weights, with the largest set to 0."""
    data = load("whole_picture.json")
    pieces, weights = data["pieces"], data["attention"]
    logs = [math.log(w) for w in weights]
    scores = [v - max(logs) for v in logs]
    n = len(pieces)
    b = text(40, 22, "Softmax turns one row of scores into weights that add to 1", "th")
    b += text(40, 42, "“ is” scoring the five tokens before it (layer 1, head 4). Largest score set to 0.", "ts")
    X0, X1 = 100, 620
    xs = [X0 + i * (X1 - X0) / (n - 1) for i in range(n)]

    def y_at(v, vmin, vmax, top, bot):
        return bot - (v - vmin) / (vmax - vmin) * (bot - top)

    # Top: score gaps. Bottom: the weights those gaps become.
    panels = (
        (68, 148, scores, -14, 1, "score gap", lambda v: f"{v:.1f}"),
        (186, 266, weights, 0, 1, "weight", lambda v: f"{v:.0%}" if v >= 0.005 else "≈ 0"),
    )
    for top, bot, values, vmin, vmax, name, fmt in panels:
        color = ACCENT["coral"] if name == "score gap" else ACCENT["teal"]
        b += text(X0, top - 8, name, "tx")
        for tick in ((0, -7, -14) if name == "score gap" else (0, 0.5, 1)):
            y = y_at(tick, vmin, vmax, top, bot)
            b += line(X0, y, X1, y, None, 0.5, dash=True)
            b += text(X0 - 8, y, f"{tick:.0f}" if name == "score gap" else f"{tick:.1f}", "tx", "end")
        pts = [(xs[i], y_at(values[i], vmin, vmax, top, bot)) for i in range(n)]
        b += path("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts), color=color, width=1.8, arrow_end=False)
        for i, (x, y) in enumerate(pts):
            hot = pieces[i] == " France"
            b += circle(x, y, 4.5 if hot else 3.5, ACCENT["amber"] if hot else color)
            b += text(x, y - 12, fmt(values[i]), "tx", "middle")
    for i, piece in enumerate(pieces):
        b += text(xs[i], 284, show_space(piece), "ts", "middle", ' font-family="Consolas,monospace"')
    b += text(40, 308, "A low score falls to about 0. The high one keeps most of the row. The five weights add to 1.", "ts")
    write("attention_softmax.svg", 328, "Softmax of one attention row", b)


def gqa_heads():
    """12 query heads share 2 key/value heads (fan-in), so the per-token
    memory is 6x smaller than one K/V per query head."""
    b = text(40, 22, "12 query heads, 2 shared key/value heads", "th")
    qy, ky = 64, 160
    for h in range(12):
        x = 50 + h * 30 + (24 if h >= 6 else 0)
        group = h // 6
        b += circle(x, qy, 10, ACCENT["purple"]) + text(x, qy, f"q{h}", "tx", "middle", ' style="fill:#fff"')
        kx = 125 + group * 204
        b += line(x, qy + 10, kx, ky - 16, ACCENT["purple"], 0.9)
    for g in range(2):
        kx = 125 + g * 204
        b += chip(kx - 46, ky - 16, 92, 32, "coral", f"K{g} V{g}", cls="th")
    b += text(40, ky + 44, "Each query head computes its own scores, but every 6 of them read the same keys and values.", "ts")
    # Memory per token, proportional bars.
    bx, by = 460, 52
    b += text(bx, by - 8, "KV cache per token", "ts")
    full = 172
    for k, (name, kb, ramp) in enumerate([("one K,V per query head", 172, "gray"), ("GQA (this model)", 28.7, "coral")]):
        y = by + k * 52
        w = kb / full * 180
        b += rect(bx, y + 4, w, 20, ramp)
        b += text(bx, y + 36, f"{name}: {kb:.0f} KB", "tx")
    write("gqa_heads.svg", ky + 60, "Grouped-query attention", b)


def bytes_or_math():
    """For each kernel, the time its bytes need at peak bandwidth and the
    time its math needs at peak FLOP/s, on a log scale: the longer bar is the
    limit, and the gap shows by how much."""
    kernels = [
        ("vector add", 805.3e6, 67.1e6),
        ("sum (reduction)", 268.4e6, 67.1e6),
        ("LLM decode step", 3.09e9, 3.09e9),
        ("matmul 4,096³", 201.3e6, 137.4e9),
    ]
    b = text(40, 22, "Bytes or math? Time each would take at the hardware's peak; the longer one is the limit", "th")
    X0, X1 = 170, 580
    lo, hi = -6, -2          # 1 us .. 10 ms

    def px(seconds):
        return X0 + (math.log10(seconds) - lo) / (hi - lo) * (X1 - X0)

    for t, label in ((1e-6, "1 µs"), (1e-5, "10 µs"), (1e-4, "100 µs"), (1e-3, "1 ms"), (1e-2, "10 ms")):
        b += line(px(t), 46, px(t), 56 + len(kernels) * 52, None, 0.4, dash=True)
        b += text(px(t), 40, label, "tx", "middle")
    for k, (name, nbytes, flops) in enumerate(kernels):
        y = 60 + k * 52
        t_mem, t_math = nbytes / 736e9, flops / 52.2e12
        mem_wins = t_mem > t_math
        b += text(X0 - 12, y + 14, name, "ts", "end")
        b += fill_rect(X0, y, px(t_mem) - X0, 13, ACCENT["blue"], 0.95 if mem_wins else 0.3, rx=2)
        b += fill_rect(X0, y + 16, px(t_math) - X0, 13, ACCENT["amber"], 0.95 if not mem_wins else 0.3, rx=2)
        ratio = max(t_mem, t_math) / min(t_mem, t_math)
        b += text(max(px(t_mem), px(t_math)) + 6, y + 14, f"{'memory' if mem_wins else 'math'} {ratio:,.0f}×", "tx")
    y = 60 + len(kernels) * 52 + 10
    b += fill_rect(X0, y, 12, 12, ACCENT["blue"], 0.95, rx=2) + text(X0 + 18, y + 6, "bytes ÷ 736 GB/s", "tx")
    b += fill_rect(X0 + 140, y, 12, 12, ACCENT["amber"], 0.95, rx=2) + text(X0 + 158, y + 6, "FLOPs ÷ 52 TFLOP/s", "tx")
    write("bytes_or_math.svg", y + 30, "Memory-bound or compute-bound", b)


def fusion():
    """Every separate kernel writes its result to DRAM and the next reads it
    back: a dip in the zigzag. Fusion keeps intermediates on chip."""
    b = text(40, 22, "Kernel fusion: fewer trips to memory between steps (one layer, one token)", "th")

    def zigzag(y, kernels, color, label):
        top, bottom = y, y + 44
        X0, X1 = 140, 640
        step = (X1 - X0) / kernels
        s = text(40, top + 2, label, "th") + text(40, top + 20, "on chip", "tx") + text(40, bottom, "memory", "tx")
        s += line(X0, bottom + 8, X1, bottom + 8, None, 0.6, dash=True)
        d = f"M{X0} {top}"
        for k in range(kernels):
            x = X0 + k * step
            s += fill_rect(x + 2, top - 6, step * 0.55, 12, color, 0.95, rx=2)
            d += f" L{x + step * 0.55 + 2:.1f} {top} L{x + step * 0.78:.1f} {bottom} L{x + step:.1f} {top}"
        s += path(d, color=ACCENT["coral"], width=1.2, arrow_end=False)
        return s

    b += zigzag(56, 20, ACCENT["gray"], "20 kernels")
    b += zigzag(146, 6, ACCENT["teal"], "6 kernels")
    b += text(40, 222, "Bars: kernels. Each dip: an intermediate result written to global memory and read back by the next kernel,", "ts")
    b += text(40, 240, "plus one more launch. Fused kernels keep the normalized x, the gate and up halves and the partial outputs", "ts")
    b += text(40, 258, "in registers instead (weights still stream from memory inside each kernel: that part can't be avoided).", "ts")
    write("fusion.svg", 276, "Kernel fusion in one layer", b)


def residual_stream():
    """One token's real vector after every layer, as a filmstrip: neighbouring
    columns look alike because each layer only adds to the vector; the vector
    grows; and read out as a word, it only becomes " Paris" near the end."""
    lens = load("logit_lens.json")
    stream = load("residual_stream.json")
    b = text(40, 22, "One token's vector after each layer: every layer adds a little, nothing is overwritten", "th")
    X0, X1 = 120, 650
    cols = len(stream["values"])
    cw = (X1 - X0) / cols

    def cx(layer):
        return X0 + layer * cw

    # Size of the vector: it grows ~400x, which is why each block reads a
    # normalized copy (RMSNorm).
    SY, SH = 44, 40
    biggest = max(stream["size"])
    b += text(X0 - 10, SY + SH - 4, "size", "tx", "end")
    for layer, size in enumerate(stream["size"]):
        h = max(1.0, size / biggest * SH)
        b += fill_rect(cx(layer) + 2, SY + SH - h, cw - 4, h, ACCENT["gray"], 0.8, rx=1)
    # The vector itself: 48 of its 1,536 numbers, each column scaled to the
    # same typical size so the pattern is comparable.
    HY, cell = SY + SH + 10, 3.2
    rows = len(stream["dims"])
    b += text(X0 - 10, HY + rows * cell / 2 - 6, "the vector", "tx", "end")
    b += text(X0 - 10, HY + rows * cell / 2 + 8, "(48 of 1,536)", "ts", "end")
    for layer, values in enumerate(stream["values"]):
        for r, v in enumerate(values):
            color = ACCENT["teal"] if v > 0 else ACCENT["coral"]
            b += fill_rect(cx(layer) + 1, HY + r * cell, cw - 2, cell, color, min(1.0, abs(v) / 3.0))
    # The word the vector reads as if the model stopped there (logit lens).
    WY = HY + rows * cell + 12
    b += text(X0 - 10, WY + 6, "reads as", "tx", "end")
    for p in lens:
        x = cx(p["layer"]) + cw / 2 + 3
        word = show_space(p["top"])
        paris = p["top"] == " Paris"
        style = f' style="fill:{ACCENT["amber"]};font-weight:600"' if paris else ""
        b += (f'<text class="ts" x="{x:.1f}" y="{WY}" text-anchor="end" transform="rotate(-60 {x:.1f} {WY})" '
              f'font-family="Consolas,monospace"{style}>{esc(word)}</text>')
    # The chance of " Paris" at each layer.
    PY, PH = WY + 56, 40
    b += text(X0 - 10, PY + PH - 4, "␣Paris", "tx", "end")
    b += line(X0, PY + PH + 0.5, X1, PY + PH + 0.5, None, 0.6)
    for p in lens:
        h = p["paris"] * PH
        if h > 0.5:
            b += fill_rect(cx(p["layer"]) + 2, PY + PH - h, cw - 4, h, ACCENT["amber"], 0.9, rx=1)
    for p in lens:
        if p["layer"] in (25, 28):
            b += text(cx(p["layer"]) + cw / 2, PY + PH - p["paris"] * PH - 6, f'{p["paris"]:.0%}', "tx", "middle")
    for layer in (0, 7, 14, 21, 28):
        b += text(cx(layer) + cw / 2, PY + PH + 16, "input" if layer == 0 else str(layer), "tx", "middle")
    b += text((X0 + X1) / 2, PY + PH + 32, "after layer", "ts", "middle")
    cy = PY + PH + 58
    b += text(40, cy, "Real values, last token of “The capital of France is”. Each column is the same vector one layer later:", "ts")
    b += text(40, cy + 18, "the pattern carries over (each layer only adds to it), the size keeps growing, and read out as a word", "ts")
    b += text(40, cy + 36, "it is noise until the additions turn it into “Paris” around layer 24.", "ts")
    write("residual_stream.svg", cy + 54, "One token's vector through the residual stream", b)


def sampling_steps():
    """The same real distribution, narrowed stage by stage (computed over the
    full vocabulary): temperature sharpens it, top-k and top-p cut the tail."""
    data = load("sampling_stages.json")
    names = ["the model's odds", "temperature 0.7", "top-k 20", "top-p 0.8"]
    b = text(40, 22, "Choosing the next word: one real distribution, narrowed step by step", "th")
    W, H, Y = 140, 120, 60
    for k, (name, probs) in enumerate(zip(names, data["stages"])):
        X0 = 40 + k * 156
        b += text(X0, Y - 12, name, "ts")
        bw = W / len(probs)
        for i, p in enumerate(probs):
            if p > 0:
                h = p * H / 0.75
                b += fill_rect(X0 + i * bw, Y + H - h, bw - 1, h, ACCENT["amber"] if i == 0 else ACCENT["teal"], 0.9)
            else:
                b += fill_rect(X0 + i * bw, Y + H - 2, bw - 1, 2, ACCENT["gray"], 0.35)
        b += line(X0, Y + H + 1, X0 + W, Y + H + 1)
        b += text(X0, Y + H + 16, f"␣Paris {probs[0]:.0%}", "tx")
        b += text(X0 + W, Y + H + 16, f"{sum(1 for p in probs if p > 0)} left" if k >= 2 else "", "tx", "end")
    b += text(40, Y + H + 46, "Temperature below 1 sharpens the odds; top-k keeps the 20 likeliest; top-p keeps the fewest that make up 80%.", "ts")
    b += text(40, Y + H + 64, "Then one is drawn at random by its odds. Temperature 0 = always the tallest bar. (Top 30 of 151,936 shown.)", "ts")
    write("sampling_steps.svg", Y + H + 82, "Sampling steps", b)


def chat_session():
    """Two turns of a conversation in the KV cache: each row is one kind of
    pass; bright = computed then, faint = read from the cache."""
    segments = [("turn 1: system + question", 42, "purple"), ("Mei's reply", 24, "teal"),
                ("turn 2: ↵ + question", 17, "purple"), ("Mei's reply", 20, "teal")]
    c, X0, Y0 = 4.3, 190, 52
    b = text(40, 22, "A conversation in the KV cache: each turn only computes what's new", "th")
    start = 0
    for r, (name, n, color) in enumerate(segments):
        y = Y0 + r * 26
        b += text(X0 - 10, y + 8, name, "tx", "end")
        if start:
            b += fill_rect(X0, y, start * c - 1, 16, ACCENT["coral"], 0.22, rx=2)
        b += fill_rect(X0 + start * c, y, n * c - 1, 16, ACCENT[color], 0.95, rx=2)
        if color == "teal":
            for t in range(n):
                b += line(X0 + (start + t) * c, y, X0 + (start + t) * c, y + 16, "#1F1E1C", 0.6)
        start += n
    y_end = Y0 + len(segments) * 26
    b += line(X0, y_end + 2, X0 + start * c, y_end + 2, None, 0.8, arrow_end=True)
    b += text(X0, y_end + 14, "tokens in the conversation →", "tx")
    b += text(40, y_end + 40, "Purple rows are one prefill each; teal rows are one decode pass per token (the thin stripes). Faint coral", "ts")
    b += text(40, y_end + 58, "is read, never recomputed: turn 2 costs 17 new tokens, not 103. (Token counts from a real exchange.)", "ts")
    write("chat_session.svg", y_end + 76, "Chat session and the KV cache", b)


def split_attention():
    """The KV cache as a tape: before, one worker per head walks all of it;
    after, many workers take a slice each and their partial results merge."""
    b = text(40, 22, "Decode attention: one block per head over every key, or many slices merged", "th")
    X0, X1 = 140, 640
    # Before: a single block walks the whole tape.
    y = 62
    b += text(40, y + 6, "before", "th")
    b += fill_rect(X0, y, X1 - X0, 14, ACCENT["coral"], 0.3, rx=2)
    b += arrow(X0, y - 10, X1, y - 10, ACCENT["gray"], 1.4)
    b += text(X0, y + 28, "1 block per query head walks every key; 12 blocks for 80 SMs, keys re-read by all 6 heads sharing them", "tx")
    # After: slices, each with its own block, then a merge.
    y = 130
    b += text(40, y + 6, "after", "th")
    slices = 16
    w = (X1 - X0) / slices
    for k in range(slices):
        x = X0 + k * w
        b += fill_rect(x + 1, y, w - 2, 14, ACCENT["coral"], 0.3, rx=2)
        b += arrow(x + 3, y - 10, x + w - 3, y - 10, ACCENT["teal"], 1.2)
        b += line(x + w / 2, y + 16, (X0 + X1) / 2, y + 66, ACCENT["teal"], 0.7)
    b += circle((X0 + X1) / 2, y + 72, 9, ACCENT["amber"])
    b += text((X0 + X1) / 2 + 16, y + 72, "combine: rescale each slice by its max, add", "tx")
    b += text(40, y + 104, "Each slice's block reads its keys once for all 6 query heads of its group and keeps (max, sum, weighted", "ts")
    b += text(40, y + 122, "values); the combine step merges them exactly. Up to 128 slices per KV head keep every SM busy.", "ts")
    write("split_attention.svg", y + 140, "Split-K decode attention", b)


def decode_slices():
    """One sequence of 512 cached keys, as decode attention cuts it: 32
    slices of 16 keys before, 8 of 64 now. Under each slice, what it reads
    (keys and values) and what it writes back (its partial result) to
    scale: with short slices the writes nearly match the reads."""
    keys, kv_bytes_per_key = 512, 2 * 2 * 128   # key + value, one KV head of 128 BF16 values
    partial = 6 * 128 * 4 + 2 * 6 * 4              # 6 query heads' partial outputs, maxes and sums, FP32
    b = text(40, 22, "One sequence's 512 cached keys, cut into slices: what each slice reads and writes", "th")
    y = 56
    for name, per_slice in (("before: 16 keys per slice", 16), ("now: 64 keys per slice", 64)):
        slices = keys // per_slice
        b += text(40, y, f"{name} ({slices} slices)", "ts")
        y += 14
        width = 600 / slices
        read = per_slice * kv_bytes_per_key
        unit = 46 / (64 * kv_bytes_per_key)   # pixels per byte, the same for both
        for s in range(slices):
            x = 40 + s * width
            b += fill_rect(x + 0.5, y, width - 1, read * unit, ACCENT["teal"], 0.85)
            b += fill_rect(x + 0.5, y + read * unit + 2, width - 1, partial * unit, ACCENT["coral"], 0.85)
        total_read = keys * kv_bytes_per_key
        total_written = slices * partial
        y += (read + partial) * unit + 16
        b += text(40, y, f"reads {total_read / 1024:.0f} KB, writes {total_written / 1024:.0f} KB of partials "
                         f"({total_written / total_read:.0%} on top), merged by a second kernel", "tx")
        y += 30
    b += fill_rect(40, y - 5, 10, 10, ACCENT["teal"], 0.85) + text(56, y, "keys and values read", "tx")
    b += fill_rect(200, y - 5, 10, 10, ACCENT["coral"], 0.85) + text(216, y, "partial result written", "tx")
    write("decode_slices.svg", y + 20, "Decode attention slices, before and now", b)


def cuda_graphs():
    """CPU and GPU on one time axis: launching each kernel separately leaves
    the GPU waiting; a recorded graph is launched once and runs gap-free."""
    b = text(40, 22, "CUDA Graphs: the CPU stops being the bottleneck between kernels", "th")
    X0 = 140

    def lanes(y, label, launches, gaps):
        s = text(40, y - 16, label, "th")
        s += text(X0 - 10, y + 6, "CPU", "tx", "end") + text(X0 - 10, y + 36, "GPU", "tx", "end")
        x_cpu, x_gpu = X0, X0
        kernels = [18, 6, 22, 5, 30, 6, 16, 5, 24, 6, 18, 5]
        for k, work in enumerate(kernels):
            if launches == "each" or k == 0:
                s += fill_rect(x_cpu, y, 10, 12, ACCENT["gray"], 0.9, rx=1)
                start_ok = x_cpu + 10
                x_cpu += 12
            else:
                start_ok = 0
            x_gpu = max(x_gpu, start_ok) + (2 if gaps else 0)
            s += fill_rect(x_gpu, y + 30, work, 12, ACCENT["purple"], 0.95, rx=1)
            x_gpu += work
        return s, x_gpu

    before, end_before = lanes(66, "launching each kernel", "each", True)
    after, end_after = lanes(150, "replaying a graph", "once", False)
    b += before + after
    b += line(end_before, 60, end_before, 114, ACCENT["red"], 1.0, dash=True)
    b += line(end_after, 144, end_after, 198, ACCENT["teal"], 1.0, dash=True)
    b += text(40, 216, "Gray: the CPU launching a kernel (a few microseconds each). Purple: GPU work. Short kernels finish before", "ts")
    b += text(40, 234, "the next launch arrives, so the GPU idles. A graph is recorded once and launched with one call per token.", "ts")
    write("cuda_graphs.svg", 252, "CUDA Graphs for decode", b)


def flash_tiles():
    """A 377-token prompt's attention as a grid of query rows against keys,
    in 64 x 64 tiles: the tiles below the diagonal are computed whole, the
    diagonal ones masked, the ones above never loaded. Beside it, the old
    kernel: each row on its own, every key read again for every row."""
    rows = 377
    tile = 64
    cell = 300 / rows
    b = text(40, 22, "One 377-token prompt: which query rows meet which keys", "th")
    x0, y0 = 60, 60
    n = math.ceil(rows / tile)
    for r in range(n):
        for k in range(n):
            if k > r:
                continue
            x = x0 + k * tile * cell
            yy = y0 + r * tile * cell
            w = min(tile, rows - k * tile) * cell
            h = min(tile, rows - r * tile) * cell
            color = ACCENT["teal"] if k < r else ACCENT["green"]
            b += fill_rect(x + 0.6, yy + 0.6, w - 1.2, h - 1.2, color, 0.8 if k < r else 0.55)
    b += path(f"M{x0} {y0} L{x0 + rows * cell} {y0 + rows * cell}", ACCENT["red"], 1.4, arrow_end=False)
    b += text(x0 + rows * cell / 2, y0 - 12, "keys →", "tx", "middle")
    b += text(x0 - 10, y0 + rows * cell / 2, "rows ↓", "tx", "end")
    computed = sum(1 for r in range(n) for k in range(n) if k <= r)
    b += text(400, 80, f"{computed} of {n * n} tiles of 64 x 64 are computed;", "ts")
    b += text(400, 98, "the rest (above the diagonal) are never loaded.", "ts")
    b += fill_rect(400, 122, 12, 12, ACCENT["teal"], 0.8) + text(418, 128, "every key visible", "tx")
    b += fill_rect(400, 142, 12, 12, ACCENT["green"], 0.55) + text(418, 148, "on the diagonal: masked inside", "tx")
    b += text(400, 186, "Each key/value tile is loaded once for 64 rows,", "ts")
    b += text(400, 204, "and its products run on tensor cores.", "ts")
    old_reads = rows * (rows + 1) // 2 * 12
    new_reads = computed * tile * 12
    b += text(400, 240, "Keys and values read, all 12 heads:", "ts")
    b += text(400, 258, f"one row at a time: {old_reads:,} key reads", "tx")
    b += text(400, 274, f"64-row tiles: {new_reads:,} ({old_reads / new_reads:.0f}x fewer)", "tx")
    write("flash_tiles.svg", y0 + rows * cell + 30, "A prompt's attention in tiles", b)


def who_evicts_whom():
    """The engine when the actions test froze: every reply holding its
    pages, in the order they were asked, none free. The oldest needs one
    more page; under the new rule it may evict any newer reply, newest first."""
    data = load("deadlock.json")
    convs = sorted(data["conversations"], key=lambda c: c["order"])
    b = text(40, 22, "When the actions test froze: 31 replies (bar: pages each holds), none free", "th")
    base, scale, bw = 262, 2.4, 16
    top = max(c["pages"] for c in convs)
    for i, c in enumerate(convs):
        x = 60 + i * (bw + 2)
        h = c["pages"] * scale
        color = ACCENT["teal"] if c["generating"] else ACCENT["amber"]
        b += fill_rect(x, base - h, bw, h, color, 0.85 if i else 1.0, rx=2)
    first_x = 60 + bw / 2
    last_x = 60 + (len(convs) - 1) * (bw + 2) + bw / 2
    b += text(first_x, base + 14, "oldest", "tx", "middle") + text(last_x, base + 14, "newest", "tx", "middle")
    b += arrow(first_x + 30, base + 34, last_x - 30, base + 34, ACCENT["gray"], 1.2)
    b += text((first_x + last_x) / 2, base + 48, "asked earlier → later", "tx", "middle")
    peak = base - top * scale - 18
    b += path(f"M{first_x} {base - convs[0]['pages'] * scale - 4} C{first_x + 60} {peak - 40} {last_x - 60} {peak - 40} "
              f"{last_x} {base - convs[-1]['pages'] * scale - 6}", ACCENT["green"], 1.6)
    b += text((first_x + last_x) / 2, peak - 50, "a reply may evict replies asked after it, newest first", "ts", "middle")
    y = base + 74
    b += fill_rect(40, y - 5, 10, 10, ACCENT["amber"], 0.85, rx=2) + text(56, y, "waiting for its first token", "tx")
    b += fill_rect(230, y - 5, 10, 10, ACCENT["teal"], 0.85, rx=2) + text(246, y, "already generating", "tx")
    b += text(400, y, f"free pages: {data['free_pages']}", "tx")
    y += 22
    b += text(40, y, "Evicting freed nothing here: an engine already gone still held the same pages. And under the old rule,", "ts")
    b += text(40, y + 18, "no waiting reply could evict another. Ordered by age, the oldest reply can always make room.", "ts")
    write("who_evicts_whom.svg", y + 40, "Who may evict whom", b)


def engine_thread():
    """The game thread and the engine thread on one time axis."""
    b = text(40, 22, "Game thread and engine thread on one time axis: frames never wait for tokens", "th")
    X0, X1 = 110, 650
    ms = 200.0

    def tx(t):
        return X0 + t / ms * (X1 - X0)

    gy, ey = 60, 130
    b += text(X0 - 10, gy + 14, "game thread", "ts", "end") + text(X0 - 10, ey + 14, "engine thread", "ts", "end")
    t = 0.0
    while t < ms - 16:
        b += fill_rect(tx(t) + 1, gy, tx(t + 16.7) - tx(t) - 2, 28, ACCENT["purple"], 0.8, rx=2)
        t += 16.7
    b += fill_rect(tx(0), ey, tx(65) - tx(0) - 1, 28, ACCENT["amber"], 0.85, rx=2)
    b += text((tx(0) + tx(65)) / 2, ey + 14, "prefill", "tx", "middle", ' style="fill:#fff"')
    t = 65.0
    while t < ms - 12:
        b += fill_rect(tx(t) + 1, ey + 4, tx(t + 12.6) - tx(t) - 2, 20, ACCENT["teal"], 0.85, rx=2)
        frame_after = math.ceil((t + 12.6) / 16.7) * 16.7
        if frame_after < ms:
            b += arrow(tx(t + 12.6), ey - 2, tx(frame_after) + 3, gy + 31, ACCENT["gray"], 0.8)
        t += 12.6
    b += arrow(tx(0) + 3, gy + 31, tx(0) + 3, ey - 2, ACCENT["coral"], 1.2)
    b += text(tx(0) + 8, gy + 44, "Ask()", "tx")
    b += line(X0, ey + 46, X1, ey + 46, None, 0.8, arrow_end=True)
    for v in (0, 50, 100, 150, 200):
        b += text(tx(v), ey + 58, f"{v} ms", "tx", "middle")
    b += text(40, ey + 86, "Purple: frames every 16.7 ms (60 fps). Teal: one token each. Each finished token becomes an event the next", "ts")
    b += text(40, ey + 104, "frame picks up (gray arrows); a frame only takes a short lock to read them, so the frame rate never drops.", "ts")
    write("engine_thread.svg", ey + 122, "Game thread and engine thread", b)


def e4m3_values():
    """Every positive finite E4M3 value."""
    values = [m / 8 * 2.0 ** -6 for m in range(1, 8)]   # subnormals
    for e in range(-6, 9):
        for m in range(8):
            v = (1 + m / 8) * 2.0 ** e
            if v <= 448:
                values.append(v)
    return values


def fp8_grid():
    """A real weight row's values (the gray histogram) and every value FP8
    can store for that row (the ticks): the ticks crowd where the weights
    crowd, near zero, because a float's steps grow with its size."""
    data = load("quant_weights.json")
    row, scale = data["row"], data["fp8_scale"]
    edge = max(abs(v) for v in row)
    X0, X1, Y_AXIS = 60, 620, 210
    b = text(40, 22, f"FP8 for one real weight row (layer {data['layer']}, down_proj): the grid follows the weights", "th")

    def px(v):
        return X0 + (v + edge) / (2 * edge) * (X1 - X0)

    bins = 120
    counts = [0] * bins
    for v in row:
        counts[min(bins - 1, int((v + edge) / (2 * edge) * bins))] += 1
    top = max(counts)
    bw = (X1 - X0) / bins
    for k, c in enumerate(counts):
        h = math.sqrt(c / top) * 110
        b += fill_rect(X0 + k * bw, Y_AXIS - 14 - h, bw - 0.5, h, ACCENT["gray"], 0.55)
    b += line(X0, Y_AXIS, X1, Y_AXIS, None, 0.8)
    for v in e4m3_values():
        for x in (px(v * scale), px(-v * scale)):
            b += line(x, Y_AXIS - 7, x, Y_AXIS + 7, ACCENT["teal"], 0.6)
    b += line(px(0), Y_AXIS + 10, px(0), Y_AXIS + 16, None, 0.8) + text(px(0), Y_AXIS + 26, "0", "tx", "middle")
    b += text(X0, Y_AXIS + 26, f"{-edge:.3f}", "tx", "start") + text(X1, Y_AXIS + 26, f"{edge:.3f}", "tx", "end")
    big_step = 32 * scale
    b += text(X1, Y_AXIS + 40, f"steps of {big_step:.4f} out here", "tx", "end")
    b += text(px(0), Y_AXIS + 40, f"steps of {scale * 2 ** -9:.1e} near 0", "tx", "middle")
    b += text(px(0) + 10, Y_AXIS - 140, f"{len(row):,} weights, most of them here", "tx")
    b += text(40, Y_AXIS + 66, "Gray: how many of the row's weights have each value (square-root scale). Teal: the 253 values an FP8", "ts")
    b += text(40, Y_AXIS + 84, "byte can hold once the row is scaled so its largest weight lands on 448. Each is within 1/16 of the", "ts")
    b += text(40, Y_AXIS + 102, "weight it replaces: the error is relative, small for small weights and larger for the few large ones.", "ts")
    write("fp8_grid.svg", Y_AXIS + 120, "FP8 values for a weight row", b)


def int4_group():
    """A real group of 128 weights (dots) and the 16 levels INT4 can store
    for it, spanning the full range (top) or a clipped one (bottom). Each
    dot's line is its rounding error: clipping lets one outlier saturate and
    gives the other 127 weights finer steps."""
    data = load("quant_weights.json")
    group = data["group"]
    lo, hi = min(group), max(group)
    clip_lo, clip_hi = data["clip_range"]
    X0, X1 = 70, 630

    def px(v):
        return X0 + (v - lo) / (hi - lo) * (X1 - X0)

    b = text(40, 22, "INT4 for one real group of 128 weights: 16 levels, placed to make the error smallest", "th")
    panels = ((f"full range: error {data['full_error'] * 1e3:.2f} (×10⁻³)", lo, hi, data["int4_full"], ACCENT["gray"]),
              (f"clipped: error {data['clipped_error'] * 1e3:.2f} (×10⁻³)", clip_lo, clip_hi, data["int4_clipped"], ACCENT["purple"]))
    y = 52
    for title, a, z, rounded, color in panels:
        b += text(40, y, title, "ts")
        top, bottom = y + 14, y + 104
        for k in range(16):
            x = px(a + (z - a) * k / 15)
            b += line(x, top, x, bottom, color, 1.1)
        for i, (w, r) in enumerate(zip(group, rounded)):
            yy = top + 6 + (i * 37 % 128) / 128 * (bottom - top - 12)
            b += line(px(w), yy, px(r), yy, ACCENT["coral"], 1.0)
            b += circle(px(w), yy, 2.2, ACCENT["blue"])
        y = bottom + 34
    b += text(40, y - 10, "Blue: the weights. Lines down: the 16 levels. Coral: each weight's rounding error (to its nearest level).", "ts")
    b += text(40, y + 8, "Spanning every weight wastes levels on the one far out at the left. Clipping makes that weight saturate", "ts")
    b += text(40, y + 26, "on the last level and gives the other 127 finer steps; the quantizer tries 121 ranges and keeps the best.", "ts")
    write("int4_group.svg", y + 44, "INT4 levels for one group", b)


def logit_mask():
    """The model's real odds for its next token at three moments: right
    after "ACTION: " in the first framing (actions without a leading space),
    right after "ACTION:" now, and after the action's verb. Allowed tokens
    in teal; the ones the grammar forbids grey and struck through. The
    share at the right of each title is how much of the odds the grammar
    allows."""
    data = load("action_logits.json")
    b = text(40, 22, "The model's own odds, and what the grammar lets it pick", "th")
    y = 44
    reply = data["reply"]
    if len(reply) > 74:   # keep the line inside the picture: cut at the last whole word
        reply = reply[:74].rsplit(" ", 1)[0] + " ..."
    b += text(40, y, f"You: “{data['question']}”   Mei: “{reply}”", "tx")
    panels = [("first try: after  ACTION:␣  (actions had no leading space)", data["at_start_after_space"]),
              ("now: after  ACTION:  (actions start with a space)", data["at_start"]),
              (f"then: after  ACTION:{visible(data['verb'])}", data["at_argument"])]
    for title, odds in panels:
        y += 30
        b += text(40, y, title, "ts")
        b += text(640, y, f"allowed: {odds['allowed_mass'] * 100:.2g}% of the odds", "tx", "end")
        y += 4
        rows = odds["top"][:6]
        top = max(r["p"] for r in rows)
        for r in rows:
            y += 17
            w = r["p"] / top * 330
            allowed = r["allowed"]
            b += mono(170, y, visible(r["token"]), "ts" if allowed else "tx", "end")
            b += fill_rect(180, y - 6, w, 12, ACCENT["teal"] if allowed else GRAMMAR_GRAY, 0.9 if allowed else 0.7, rx=2)
            if not allowed:
                b += line(180, y, 180 + w, y, ACCENT["red"], 1.2)
            b += text(186 + w, y, f"{r['p'] * 100:.1f}%", "tx")
        best = odds["best_allowed"]
        if best["rank"] > len(rows):
            y += 19
            b += text(170, y, f"... the grammar's best, “{visible(best['token'])}”, is number {best['rank']}"
                      f" ({best['p'] * 100:.2f}%)", "tx", "start")
        y += 4
    y += 24
    b += fill_rect(40, y - 5, 10, 10, ACCENT["teal"], 0.9, rx=2) + text(56, y, "allowed: the sampler picks among these", "tx")
    b += fill_rect(300, y - 5, 10, 10, GRAMMAR_GRAY, 0.7, rx=2) + line(300, y, 310, y, ACCENT["red"], 1.2)
    b += text(316, y, "masked out: never a candidate", "tx")
    write("logit_mask.svg", y + 20, "The model's odds, and what the grammar allows", b)


def segments(marked):
    """'[quest]([umb]rella' -> [('quest', True), ('(', False), ('umb', True), ('rella', False)]."""
    out, chosen, buf = [], False, ""
    for ch in marked:
        if ch in "[]":
            if buf:
                out.append((buf, chosen))
            buf, chosen = "", ch == "["
        else:
            buf += ch
    if buf:
        out.append((buf, chosen))
    return out


def jump_forward():
    """Real actions byte by byte: what the model chose in the verb's colour,
    what the grammar wrote in grey. Then the model steps per action with and
    without jump-forward."""
    data = load("actions.json")
    examples = data["constrained"]["examples"]
    b = text(40, 22, "Who wrote each byte of real actions: the model (colour) or the grammar (grey)", "th")
    y = 52
    cw = 10
    for verb in ["walk", "call", "sell", "quest", "hack", "none"]:
        if verb not in examples:
            continue
        x = 40
        for part, chosen in segments(examples[verb]):
            for ch in part:
                b += fill_rect(x, y - 9, cw - 1, 18, VERB_COLOR[verb] if chosen else GRAMMAR_GRAY, 0.9 if chosen else 0.55)
                ink = "#FFFFFF" if chosen else "#2C2C2A"
                b += text(x + cw / 2 - 0.5, y, visible(ch), "ts mono", "middle", f' style="fill:{ink}"')
                x += cw
        y += 26
    y += 10
    runs = [("with jump-forward", data["constrained"]), ("without", data["no_jump_forward"])]
    b += text(40, y, "Per action, averaged over 256: tokens the model chose (one model step each) and tokens the grammar wrote", "ts")
    scale = 60
    for name, run in runs:
        y += 26
        chosen = run["sampled_tokens"] / run["actions"]
        written = run["forced_tokens"] / run["actions"]
        b += text(170, y, name, "ts", "end")
        b += fill_rect(180, y - 8, chosen * scale, 16, ACCENT["purple"], 0.9, rx=2)
        b += fill_rect(180 + chosen * scale, y - 8, written * scale, 16, GRAMMAR_GRAY, 0.6, rx=2)
        b += text(186 + (chosen + written) * scale, y, f"{chosen:.1f} model steps" + (f" + {written:.1f} written" if written else ""), "tx")
    write("jump_forward.svg", y + 24, "Who wrote which bytes of real actions", b)


def wrap_pieces(pieces, width_chars):
    """Lays text pieces out in lines of at most width_chars characters,
    breaking only at spaces; returns [(line, [(text, guessed)])]."""
    lines, current, used = [], [], 0
    for piece, guessed in pieces:
        for word in re.split(r"(?<= )", piece):
            if not word:
                continue
            if used + len(word) > width_chars and used > 0:
                lines.append(current)
                current, used = [], 0
            current.append((word, guessed))
            used += len(word)
    if current:
        lines.append(current)
    return lines


def speculation_step():
    """A real greedy reply, piece by piece: the pieces the draft model
    guessed and the big model kept, highlighted; the rest the big model
    wrote itself (one per check, at the first mismatch)."""
    spec = load("speculative_fp8.json")
    run = next(r for r in spec["runs"] if r["draft_tokens"] == 2)
    pieces = [(p[0], p[1]) for p in run["first_reply"]]
    guessed = sum(len(p) for p, g in pieces if g)
    total = sum(len(p) for p, _ in pieces)
    b = text(40, 22, "Mei's reply: what the draft model guessed (teal) and what the big model wrote", "th")
    y = 54
    for row in wrap_pieces(pieces, 78):
        x = 40
        for word, g in row:
            w = len(word) * MONO_W
            if g:
                b += fill_rect(x, y - 10, w, 20, ACCENT["teal"], 0.28, rx=2)
            b += mono(x, y, word, "ts")
            x += w
        y += 24
    y += 8
    b += text(40, y, f"Two guesses per step: {guessed} of {total} characters came from the draft model and were confirmed by the", "ts")
    b += text(40, y + 18, "big model in the same pass. The text is what the big model would have written on its own.", "ts")
    write("speculation_step.svg", y + 40, "A reply: words the draft model guessed and words the model wrote", b)


DIAGRAMS = [rms_norm, bpe_merges, paged_kv, reduction_levels, matvec_latency, rope_pairs, attention_softmax, gqa_heads,
            bytes_or_math, fusion, residual_stream, sampling_steps, chat_session, split_attention, decode_slices,
            cuda_graphs, flash_tiles, who_evicts_whom, engine_thread, fp8_grid, int4_group, logit_mask, jump_forward,
            speculation_step]


if __name__ == "__main__":
    for make in DIAGRAMS:
        make()
    print(f"{len(DIAGRAMS)} study diagrams written")
