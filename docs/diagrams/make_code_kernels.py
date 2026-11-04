"""Code pictures for the math kernels (code tour Part IV and the FP8
quantizer): each kernel drawn as it runs, with its own variable names, which
thread, lane, warp or block touches which element in each phase, and the
lines of src/ that do each part.

The pictures replay the kernels' index expressions and arithmetic in Python
(FP32 rounding included) on real data:
- data/code_rms_norm.json: the embedding of the token " is" (the last token of
  "The capital of France is", so layer 0's input for it) and layer 0's
  input_layernorm weight;
- data/code_rope.json: that token's layer-0 key head 0 before RoPE (the
  normalized embedding times k_proj, plus the bias), its position, and the
  model's head_dim and rope_theta;
- data/quant_weights.json: row 0 of layer 14's down_proj (printed to four
  significant digits, so each value is rounded back to the BF16 it came from).
The first two are refreshed from models/qwen2.5-1.5b-instruct when present.
The matrix-vector and page-table pictures show structure on the model's real
shapes; the page table there is an example.

Run from anywhere: python docs/diagrams/make_code_kernels.py
"""
import functools
import json
import math
import os
import struct
from array import array

from code_lines import lines
from svg_kit import ACCENT, arrow, chip, circle, fill_rect, line, path, rect, svg, text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL = os.path.join(HERE, "..", "..", "models", "qwen2.5-1.5b-instruct")
WARP_COLORS = ("teal", "purple", "coral", "green")   # owner colours: warp 0, 1, 2, 3
FP8_MAX = 448.0


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def save(name, data):
    with open(os.path.join(DATA, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1)


def write(name, height, title, body):
    svg(HERE, name, height, title, body)


def f32(v):
    """v rounded to the nearest FP32, as the GPU holds it."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


def bf16(v):
    """v rounded to the nearest BF16 (ties to even), as an FP32 value."""
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def model_tensor(name, start=0, count=None):
    """BF16 values start .. start + count of a tensor in model.safetensors."""
    with open(os.path.join(MODEL, "model.safetensors"), "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        begin, end = json.loads(f.read(header_size))[name]["data_offsets"]
        count = (end - begin) // 2 - start if count is None else count
        f.seek(8 + header_size + begin + 2 * start)
        halves = array("H")
        halves.frombytes(f.read(2 * count))
    return array("f", array("I", (h << 16 for h in halves)).tobytes()).tolist()


def value_color(v, peak):
    """Fill colour and opacity for a signed value: teal positive, coral negative."""
    return ACCENT["teal"] if v >= 0 else ACCENT["coral"], 0.12 + 0.88 * min(1.0, math.sqrt(abs(v) / peak))


def value_cells(cells, height, peak):
    """Many small value cells (x, y, width, value), drawn as few paths: each
    run of same-coloured neighbours is one stroke, opacity in 8 steps, so a
    row of thousands of weights stays a small file."""
    runs = []
    for x, y, w, v in cells:
        color, opacity = value_color(v, peak)
        key = (color, round(opacity * 8) / 8)
        last = runs[-1] if runs else None
        if last and last[0] == key and last[2] == y and abs(last[1] + last[3] - x) < 0.01:
            last[3] += w
        else:
            runs.append([key, x, y, w])
    paths = {}
    for key, x, y, w in runs:
        paths.setdefault(key, []).append(f"M{x:.1f} {y + height / 2:.1f}h{w:.1f}")
    return "".join(f'<path d="{"".join(d)}" stroke="{color}" stroke-opacity="{opacity:.3f}" '
                   f'stroke-width="{height}" fill="none"/>' for (color, opacity), d in paths.items())


def notes(y, rows, x_text=130):
    """The line ranges and what they do, one per row, as in every code picture."""
    b = ""
    for k, (where, what) in enumerate(rows):
        b += text(24, y + k * 17, where, "tx mono") + text(x_text, y + k * 17, what, "tx")
    return b


# --- Data ------------------------------------------------------------------

TOKEN = "Ġis"     # " is" in tokenizer.json's byte-level spelling
POSITION = 4      # its position in "The capital of France is"


@functools.cache
def token_data():
    """The embedding of " is" and layer 0's input_layernorm weight,
    refreshed from the model when it is present."""
    if os.path.exists(os.path.join(MODEL, "model.safetensors")):
        with open(os.path.join(MODEL, "tokenizer.json"), encoding="utf-8") as f:
            token_id = json.load(f)["model"]["vocab"][TOKEN]
        with open(os.path.join(MODEL, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        hidden = config["hidden_size"]
        save("code_rms_norm.json", {
            "token": TOKEN.replace("Ġ", " "), "id": token_id, "epsilon": config["rms_norm_eps"],
            "x": model_tensor("model.embed_tokens.weight", token_id * hidden, hidden),
            "weight": model_tensor("model.layers.0.input_layernorm.weight")})
    return load("code_rms_norm.json")


@functools.cache
def rope_data():
    """Layer 0's key head 0 for " is" before RoPE: RMSNorm(x) times the first
    head_dim rows of k_proj, plus the bias, as the q/k/v product computes it.
    Refreshed from the model when it is present."""
    if os.path.exists(os.path.join(MODEL, "model.safetensors")):
        with open(os.path.join(MODEL, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        hidden = config["hidden_size"]
        head_dim = hidden // config["num_attention_heads"]
        token = token_data()
        x, weight = token["x"], token["weight"]
        inverse_rms = 1.0 / math.sqrt(sum(v * v for v in x) / hidden + token["epsilon"])
        normed = [w * (v * inverse_rms) for v, w in zip(x, weight)]
        rows = model_tensor("model.layers.0.self_attn.k_proj.weight", 0, head_dim * hidden)
        bias = model_tensor("model.layers.0.self_attn.k_proj.bias", 0, head_dim)
        key = [f32(sum(rows[r * hidden + i] * normed[i] for i in range(hidden)) + bias[r]) for r in range(head_dim)]
        save("code_rope.json", {
            "token": token["token"], "position": POSITION, "layer": 0, "kv_head": 0, "head_dim": head_dim,
            "num_heads": config["num_attention_heads"], "num_kv_heads": config["num_key_value_heads"],
            "rope_theta": config["rope_theta"], "key": key})
    return load("code_rope.json")


def prologue_values(x, threads):
    """Each thread's sum_of_squares in a strided loop over x, as
    MatVecVectorKernel's norm prologue (threads = 128) computes it."""
    sums = []
    for t in range(threads):
        s = 0.0
        for i in range(t, len(x), threads):
            s = f32(s + f32(x[i] * x[i]))
        sums.append(s)
    return sums


def warp_sum_rounds(values):
    """WarpSum on 32 lanes: the values after each round, offsets 16 .. 1."""
    rounds = [list(values)]
    offset = 16
    while offset > 0:
        before = rounds[-1]
        rounds.append([f32(before[lane] + before[lane ^ offset]) for lane in range(32)])
        offset //= 2
    return rounds


def block_sum(values):
    """BlockSum over one value per thread: WarpSum in each warp, then WarpSum
    over the warp totals (lanes past the warp count hold 0)."""
    totals = [warp_sum_rounds(values[w:w + 32])[-1][0] for w in range(0, len(values), 32)]
    return warp_sum_rounds(totals + [0.0] * (32 - len(totals)))[-1][0]


# --- WarpSum ---------------------------------------------------------------

def warp_sum():
    """WarpSum on one real warp: 32 lanes (rows) after each of the 5 rounds
    (columns). Lines show which lane each lane reads with __shfl_xor_sync;
    the teal ones feed lane 0, whose sum gathers all 32 starting values."""
    token = token_data()
    rounds = warp_sum_rounds(prologue_values(token["x"], 128)[:32])
    x0, col_w, gap, top, row = 64, 58, 44, 74, 13
    offsets = [16, 8, 4, 2, 1]

    def cx(c):
        return x0 + c * (col_w + gap)

    def cy(lane):
        return top + lane * row

    b = text(24, 22, "WarpSum: 32 lanes add up their values in 5 rounds of __shfl_xor_sync", "th")
    b += text(24, 42, f'value of lane t: x[t]² + x[t+128]² + … + x[t+1408]², x = the embedding of "{token["token"]}"', "ts")
    for c in range(6):
        b += text(cx(c) + col_w / 2, top - 14, "value" if c == 0 else f"offset {offsets[c - 1]}", "ts mono", "middle")
    for lane in range(32):
        b += text(x0 - 8, cy(lane) + row / 2 - 1, str(lane), "tx mono", "end")
    # Into each lane: its own value (level) and its partner's (slanted).
    # Lane 0's inputs: after round c it depends on lanes 0 .. 32 / 2^(5-c) - 1.
    for c in range(1, 6):
        fan = 2 ** (5 - c)
        for lane in range(32):
            feeds = lane < fan
            color = ACCENT["teal"] if feeds else ACCENT["gray"]
            width, opacity = (1.2, 1.0) if feeds else (0.6, 0.45)
            for source in (lane, lane ^ offsets[c - 1]):
                y1, y2 = cy(source) + row / 2 - 1, cy(lane) + row / 2 - 1
                b += (f'<line x1="{cx(c - 1) + col_w:.1f}" y1="{y1:.1f}" x2="{cx(c):.1f}" y2="{y2:.1f}" '
                      f'stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}"/>')
    for c, values in enumerate(rounds):
        fan = 2 ** (5 - c)
        for lane, v in enumerate(values):
            ramp = "teal" if lane < fan else "gray"
            b += chip(cx(c), cy(lane), col_w, row - 2, ramp, f"{v:.4f}", "tx", mono=True, rx=2)

    y = cy(32) + 14
    total = rounds[-1][0]
    b += text(24, y, f"Each lane adds the value of lane ^ offset to its own. Teal: what lane 0's sum is built from; after 5 rounds", "ts")
    b += text(24, y + 18, f"it holds all 32 starting values, and so does every other lane: {total:.4f} everywhere.", "ts")
    f = "engine/kernels/block_reduce.cuh"
    b += notes(y + 44, [
        (lines(f, "__device__ __forceinline__ float WarpSum(float value) {",
               "value += __shfl_xor_sync(kFullWarpMask, value, offset);"),
         "offset = 16, 8, 4, 2, 1; every lane reads value from lane ^ offset (a register of that lane) and adds it"),
    ])
    write("code_warp_sum.svg", y + 44 + 17 + 6, "WarpSum round by round", b)


# --- BlockReduce -----------------------------------------------------------

def block_reduce():
    """BlockSum in a 128-thread block (MatVecVectorKernel's norm prologue) on
    real values: each warp sums its lanes, lane 0 of each warp writes
    warp_results[warp], and after the barrier every warp sums those, so
    every thread returns the block's total."""
    token = token_data()
    values = prologue_values(token["x"], 128)
    warp_totals = [warp_sum_rounds(values[32 * w:32 * w + 32])[-1][0] for w in range(4)]
    block_total = block_sum(values)
    x0, cell, stride = 40, 4.5, 152
    peak = max(values)

    def wx(w, lane=0):
        return x0 + w * stride + lane * cell

    def band(y, code, what):
        return text(24, y, code, "ts mono") + text(24 + 6.6 * len(code) + 12, y, what, "tx")

    def barrier(y):
        return line(24, y, 656, y, ACCENT["gray"], 1, dash=True) + text(656, y - 8, "__syncthreads()", "tx mono", "end")

    b = text(24, 22, "BlockReduce (as BlockSum) in a block of 128 threads: 4 warps of 32 lanes", "th")
    b += text(24, 42, f'value of thread t: its sum of squares of x[t], x[t+128], …, x = the embedding of "{token["token"]}"', "ts")

    y = 66
    b += band(y, "value", "one per thread")
    for t, v in enumerate(values):
        h = v / peak * 30
        b += fill_rect(wx(t // 32, t % 32) + 0.4, y + 44 - h, cell - 0.8, h, ACCENT[WARP_COLORS[t // 32]], 0.9)
    for w in range(4):
        b += text(wx(w) + 72, y + 56, f"warp {w}: threads {32 * w}–{32 * w + 31}", "tx", "middle")

    y = 146
    b += band(y, "value = warp_op(value);", "WarpSum inside each warp: all 32 lanes get the warp's total")
    for w in range(4):
        b += chip(wx(w), y + 12, 32 * cell, 18, WARP_COLORS[w], f"{warp_totals[w]:.4f}", "tx", mono=True)
    b += barrier(y + 44)

    y = 206
    b += band(y, "if (lane == 0) warp_results[warp] = value;", "shared memory, 32 floats")
    for w in range(4):
        b += rect(wx(w, 1), y + 12, 31 * cell, 10, "gray", rx=1, extra=' fill-opacity="0.3"')
        b += fill_rect(wx(w) + 0.4, y + 12, cell - 0.8, 10, ACCENT[WARP_COLORS[w]])
    b += text(wx(0, 2), y + 32, "lane 0 of each warp", "tx")
    slot_y, slot_w = y + 66, 600 / 32
    for w in range(4):
        b += arrow(wx(w) + cell / 2, y + 24, x0 + w * slot_w + slot_w / 2, slot_y - 2, ACCENT[WARP_COLORS[w]], 1.2)
    for s in range(32):
        b += chip(x0 + s * slot_w, slot_y, slot_w - 1, 16, "amber" if s < 4 else "gray", str(s), "tx", mono=True, rx=2)
    b += text(x0 + 4 * slot_w, slot_y + 28, "warp_results[4..31]: not written (blockDim.x / 32 = 4 warps)", "tx")
    b += barrier(slot_y + 44)

    y = slot_y + 62
    b += band(y, "value = lane < warp_count ? warp_results[lane] : identity;", "in every warp")
    for w in range(4):
        for lane in range(4):
            b += fill_rect(wx(w, lane) + 0.4, y + 12, cell - 0.8, 18, ACCENT[WARP_COLORS[lane]], 0.9)
        b += chip(wx(w, 4), y + 12, 28 * cell, 18, "gray", "identity = 0", "tx")
    b += text(wx(0), y + 40, "lanes 0–3 read warp_results[0..3]", "tx")

    y += 64
    b += band(y, "return warp_op(value);", "WarpSum again, in every warp")
    for w in range(4):
        b += chip(wx(w), y + 12, 32 * cell, 18, "amber", f"{block_total:.4f}", "tx", mono=True)

    y += 48
    b += text(24, y, f"All 128 threads return {block_total:.4f}, the sum of the 128 values. Every warp repeats the second WarpSum,", "ts")
    b += text(24, y + 18, "so no thread has to send the total to the others.", "ts")
    f = "engine/kernels/block_reduce.cuh"
    b += notes(y + 44, [
        (lines(f, "const int lane = threadIdx.x % kWarpSize;", "const int warp_count = blockDim.x / kWarpSize;"),
         "each thread's lane and warp, and how many warps the block has"),
        (lines(f, "value = warp_op(value);\n    __syncthreads();", "warp_results[warp] = value;"),
         "each warp combines its lanes; the first barrier waits until a previous call has read warp_results"),
        (lines(f, "value = lane < warp_count ? warp_results[lane] : identity;", "return warp_op(value);\n}"),
         "after the second barrier, every warp combines the warp results (identity fills the unused lanes)"),
    ])
    write("code_block_reduce.svg", y + 44 + 3 * 17 + 6, "BlockReduce phase by phase", b)


# --- RmsNormKernel ---------------------------------------------------------

def rms_norm_items(width, threads=256, max_items=16):
    """LaunchRmsNorm's choice: the fewest values per thread that cover width."""
    for items in (4, 8):
        if width <= items * threads:
            return items
    return max_items


def rms_norm():
    """RmsNormKernel on one real row: which thread holds which element in
    values[k], each thread's sum of squares, the block's sum, inverse_rms,
    and thread 3's values through to the stored y."""
    token = token_data()
    x, weight, epsilon = token["x"], token["weight"], token["epsilon"]
    width, threads = len(x), 256
    items = rms_norm_items(width)
    values = [[x[t + k * threads] if t + k * threads < width else 0.0 for k in range(items)] for t in range(threads)]
    sums = []
    for t in range(threads):
        s = 0.0
        for k in range(items):
            s = f32(s + f32(values[t][k] * values[t][k]))
        sums.append(s)
    total = block_sum(sums)
    inverse_rms = f32(1.0 / math.sqrt(f32(total / width + epsilon)))
    shown = 3
    x0, cell, top, row = 110, 2.0, 82, 13
    peak = max(abs(v) for v in x)

    b = text(24, 22, f"RmsNormKernel on one row: 256 threads × Items = {items} values, width {width:,}", "th")
    b += text(24, 42, f'x = the embedding of "{token["token"]}", weight = layer 0 input_layernorm. '
                      f"LaunchRmsNorm picks Items = {items}: {width:,} > 4 × 256.", "ts")
    b += text(x0, top - 18, "threadIdx.x = t", "tx")
    b += text(x0 + threads * cell, top - 18, "values[k] = x[t + 256 × k]", "tx mono", "end")
    for t in (0, 64, 128, 192, 255):
        b += text(x0 + t * cell + cell / 2, top - 6, str(t), "tx mono", "middle")
    cells = []
    for k in range(items):
        y = top + k * row
        b += text(x0 - 6, y + row / 2 - 1, f"values[{k}]", "tx mono", "end")
        if k * threads >= width:
            b += chip(x0, y, threads * cell, row - 2, "gray", f"i = t + {k * threads:,} ≥ width: 0, never stored", "tx", rx=1)
        else:
            cells += [(x0 + t * cell, y, cell, values[t][k]) for t in range(threads)]
    b += value_cells(cells, row - 2, peak)
    grid_bottom = top + items * row
    b += (f'<rect x="{x0 + shown * cell - 0.6:.1f}" y="{top - 1.5:.1f}" width="{cell + 1.2:.1f}" '
          f'height="{grid_bottom - top + 1:.1f}" fill="none" stroke="{ACCENT["amber"]}" stroke-width="1.2"/>')

    y = grid_bottom + 8
    bars_h = 24
    sum_peak = max(sums)
    for t, s in enumerate(sums):
        h = s / sum_peak * bars_h
        b += fill_rect(x0 + t * cell, y + bars_h - h, cell - 0.3, h, ACCENT["purple"], 0.85)
    b += text(x0 - 6, y + bars_h / 2, "sum_of_squares", "tx mono", "end")
    b += text(x0 + threads * cell + 6, y + bars_h / 2, "per thread", "tx")

    y += bars_h + 22
    b += text(24, y, f"BlockSum → {total:.4f} in every thread", "ts mono")
    b += text(24, y + 18, f"inverse_rms = 1 / sqrtf({total:.4f} / {width} + {epsilon:g}) = {inverse_rms:.3f}", "ts mono")

    # Thread 3's values through the store.
    y += 62
    b += text(24, y - 20, f"Thread {shown} (the amber column) through both passes:", "ts")
    cols = [(24, "k"), (64, "i"), (130, "values[k]"), (230, "weight[i]"), (330, "y[i] = weight[i] * (values[k] * inverse_rms)")]
    for cx, head in cols:
        b += text(cx, y, head, "tx mono")
    for k in range(items):
        ry = y + 10 + k * 16
        i = shown + k * threads
        b += text(24, ry + 7, str(k), "tx mono") + text(64, ry + 7, f"{i:,}", "tx mono")
        if i >= width:
            b += text(130, ry + 7, "0  (i ≥ width: skipped)", "tx mono")
            continue
        out = f32(weight[i] * f32(values[shown][k] * inverse_rms))
        color, opacity = value_color(values[shown][k], peak)
        b += fill_rect(126, ry, 70, 14, color, opacity * 0.5, rx=2) + text(130, ry + 7, f"{values[shown][k]:+.4f}", "tx mono")
        b += text(230, ry + 7, f"{weight[i]:+.4f}", "tx mono")
        b += chip(326, ry, 70, 14, "amber", f"{out:+.4f}", "tx", mono=True, rx=2)

    y += 10 + items * 16 + 16
    b += text(24, y, "Amber: stored. Store rounds each y to the output type (FP32, BF16 or FP16).", "ts")
    f = "engine/kernels/rms_norm.cu"
    b += notes(y + 24, [
        (lines(f, "float values[Items];", "values[k] = i < width ? x[i] : 0.0f;"),
         "thread t loads elements t, t + 256, …; past the end it holds 0"),
        (lines(f, "float sum_of_squares = 0.0f;\n#pragma unroll", "const float inverse_rms = 1.0f / sqrtf(sum_of_squares / width + epsilon);"),
         "each thread sums its squares, BlockSum adds the 256 sums, every thread computes inverse_rms"),
        (lines(f, "#pragma unroll\n    for (int k = 0; k < Items; ++k) {\n        const int i = threadIdx.x + k * kThreadsPerBlock;\n"
                  "        if (i < width) {", "Store(&y[i], weight[i] * (values[k] * inverse_rms));"),
         "the second pass reuses values[] from registers: scale, multiply by weight, store"),
        (lines(f, "if (width <= 4 * kThreadsPerBlock) {", "launch(RmsNormKernel<Out, kMaxItems>);"),
         "LaunchRmsNorm picks Items = 4, 8 or 16 from the width"),
    ])
    write("code_rms_norm.svg", y + 24 + 4 * 17 + 6, "RmsNormKernel on one row", b)


# --- MatVecVectorKernel ----------------------------------------------------

def rows_per_warp(rows, swiglu):
    """RowsPerWarp for BF16 weights (narrowness 1)."""
    base = 2 if swiglu else 4 if rows >= 16384 else 2 if rows >= 4096 else 1
    return min(8, base)


def matvec():
    """MatVecVectorKernel's BF16 path on gate_up: which warp owns which rows,
    which 8-weight chunks of a row each lane loads in which pass, lane 3's
    work phase by phase (norm prologue, batched loads, Dot8, WarpSum, SwiGLU
    epilogue), and what one Dot8 multiplies."""
    hidden, intermediate = 1536, 8960
    out_rows, in_features = 2 * intermediate, hidden
    rpw = rows_per_warp(out_rows, swiglu=True)
    bits_bytes = 16                         # sizeof(uint4): 8 BF16 weights
    loads_in_flight = 128 // bits_bytes // rpw
    vectors = in_features // 8
    step = 32 * loads_in_flight
    passes = [[lane0 + k * 32 for k in range(loads_in_flight)] for lane0 in range(0, vectors, step)]
    lane = 3

    b = text(24, 22, f"MatVecVectorKernel, BF16, on gate_up ({out_rows:,} × {in_features:,}) with norm and SwiGLU", "th")
    b += text(24, 42, f"RowsPerWarp = {rpw} · kLoadsInFlight = 128 B / 16 B / {rpw} = {loads_in_flight} · "
                      f"vectors = {in_features:,} / 8 = {vectors} · 128 threads per block", "ts")

    # Rows to warps to outputs (block 0).
    top, bar_h = 74, 11
    b += text(24, top - 12, "block 0: warp w owns rows 2w, 2w+1", "tx")
    for r in range(8):
        w = r // rpw
        y = top + r * (bar_h + 3)
        b += text(24, y + bar_h / 2, f"row {r}", "tx mono")
        b += chip(62, y, 88, bar_h, WARP_COLORS[w], f"gate {w}" if r % 2 == 0 else f"up {w}", "tx", rx=2)
    for w in range(4):
        y = top + w * 2 * (bar_h + 3)
        b += path(f"M152 {y + bar_h / 2:.1f} C164 {y + bar_h / 2:.1f} 164 {y + bar_h + 3 + bar_h / 2:.1f} 152 "
                  f"{y + bar_h + 3 + bar_h / 2:.1f}", ACCENT[WARP_COLORS[w]], 1, arrow_end=False)
        b += arrow(162, y + bar_h + 1.5, 176, y + bar_h + 1.5, ACCENT[WARP_COLORS[w]], 1)
        b += chip(178, y + 2, 34, bar_h + 6, WARP_COLORS[w], f"y[{w}]", "tx", mono=True, rx=2)
    b += text(24, top + 8 * (bar_h + 3) + 8, f"… {out_rows // 2:,} warps in {out_rows // 2 // 4:,} blocks", "tx")

    # One row's chunks: lane = column, v = base + k * 32.
    gx, cw, ch = 300, 9, 11
    b += text(236, top - 12, "one row: chunk v (8 BF16 weights, one uint4) is read by lane v % 32", "tx")
    b += text(gx - 4, top - 1, "lane", "tx", "end")
    for l in (0, 8, 16, 24, 31):
        b += text(gx + l * cw + cw / 2, top - 1, str(l), "tx mono", "middle")
    y = top + 6
    for p, chunks in enumerate(passes):
        first = y
        for v0 in chunks:
            b += text(gx - 4, y + ch / 2, f"lane + {v0}", "tx mono", "end")
            for l in range(32):
                if v0 < vectors:
                    b += rect(gx + l * cw, y, cw - 1, ch, "purple" if p == 0 else "teal", rx=1)
                else:
                    b += rect(gx + l * cw, y, cw - 1, ch, "gray", rx=1, extra=' stroke-dasharray="2 2" fill-opacity="0.3"')
            y += ch + 2
        b += text(gx + 32 * cw + 6, (first + y - 2) / 2, f"pass {p + 1}", "tx")
        y += 6
    ghosts = sum(1 for v0 in passes[-1] if v0 >= vectors)
    b += text(gx + 16 * cw, y - 6 - ghosts * (ch + 2) / 2 - 1, "v ≥ vectors: Bits{}, no load", "tx", "middle")
    b += (f'<rect x="{gx + lane * cw - 1:.1f}" y="{top + 5:.1f}" width="{cw + 1:.1f}" height="{y - top - 10:.1f}" '
          f'fill="none" stroke="{ACCENT["amber"]}" stroke-width="1.4"/>')

    # Lane 3's work, phase by phase.
    lane_chunks = [[v0 + lane for v0 in chunks] for chunks in passes]
    y = 232
    b += text(24, y, f"Lane {lane} of warp 0 (threadIdx.x = {lane}) in order; the amber outline above is its chunks, "
                     f"the same in both rows.", "ts")
    lx, chip_h = 150, 18

    def phase(y, title, detail):
        return text(24, y + chip_h / 2, title, "ts") + text(24, y + chip_h / 2 + 14, detail, "tx mono")

    y += 16
    b += phase(y, "norm prologue", "all 128 threads")
    xs = [lane + 128 * j for j in range(in_features // 128)]
    for j in range(len(xs)):
        b += chip(lx + j * 19, y, 18, chip_h, "gray", "", rx=2)
    b += text(lx, y + chip_h + 10, f"sum_of_squares += x[i] * x[i], i = {xs[0]}, {xs[1]}, …, {xs[-1]}", "tx mono")
    bx = lx + len(xs) * 19 + 8
    b += chip(bx, y, 70, chip_h, "purple", "BlockSum", "tx", mono=True)
    b += arrow(bx + 72, y + chip_h / 2, bx + 92, y + chip_h / 2)
    b += chip(bx + 94, y, 86, chip_h, "purple", "inverse_rms", "tx", mono=True)
    b += line(24, y + chip_h + 22, 656, y + chip_h + 22, ACCENT["gray"], 1, dash=True)

    for p, chunks in enumerate(lane_chunks):
        y += chip_h + 34
        ramp = "purple" if p == 0 else "teal"
        real = [v for v in chunks if v < vectors]
        b += phase(y, f"pass {p + 1}: base = {chunks[0]}", f"{len(real) * rpw * 16} B in flight")
        x = lx
        for k, v in enumerate(chunks):          # issue order: k outer, r inner
            for r in range(rpw):
                if v < vectors:
                    b += chip(x, y, 40, chip_h, ramp, f"[{r}][{k}]", "tx", mono=True, rx=2)
                else:
                    b += rect(x, y, 40, chip_h, "gray", rx=2, extra=' stroke-dasharray="2 2" fill-opacity="0.3"')
                x += 42
        b += text(lx, y + chip_h + 10, "bits[r][k] = __ldcs(rows[r] + v), all issued first", "tx mono")
        x += 10
        b += text(x, y + chip_h + 10, "then 2 Dot8 per v", "tx mono")
        for v in real:
            b += chip(x, y, 30, chip_h, "gray", f"v{v}", "tx", mono=True, rx=2)
            x += 32

    y += chip_h + 34
    b += line(24, y - 12, 656, y - 12, ACCENT["gray"], 1, dash=True)
    b += phase(y, "warp sum", "sums[0], sums[1]")
    for r in range(rpw):
        b += chip(lx + r * 112, y, 108, chip_h, "gray", f"WarpSum(sums[{r}])", "tx", mono=True)
    y += chip_h + 22
    b += phase(y, "SwiGLU epilogue", "lane 0 only")
    b += chip(lx, y, 220, chip_h, "amber", "y[0] = Silu(sums[0]) * sums[1]", "tx", mono=True)

    # One Dot8: the uint4's 4 words, two BF16 each, times a and b.
    dy = y + 46
    b += text(24, dy, "Dot8(w, a, b): uint4 w = 4 words of two BF16 weights; a, b = x[8v .. 8v+7], normalized; products summed", "ts")
    dx, dw = 24, 78
    acts = ["a.x", "a.y", "a.z", "a.w", "b.x", "b.y", "b.z", "b.w"]
    for j in range(8):
        cx = dx + j * (dw + 1)
        word = "xyzw"[j // 2]
        b += chip(cx, dy + 16, dw, 16, WARP_COLORS[0], f"Bf16{'High' if j % 2 else 'Low'}(w.{word})", "tx", mono=True, rx=2)
        b += text(cx + dw / 2, dy + 39, "×", "tx", "middle")
        b += chip(cx, dy + 46, dw, 16, "gray", acts[j], "tx", mono=True, rx=2)
    for word in range(4):
        cx = dx + word * 2 * (dw + 1)
        b += line(cx, dy + 11, cx + 2 * dw + 1, dy + 11, ACCENT["teal"], 1.4)

    y = dy + 84
    f = "engine/kernels/matvec.cu"
    b += notes(y, [
        (lines(f, "if constexpr (kNorm) {\n        float sum_of_squares = 0.0f;",
               "inverse_rms = 1.0f / sqrtf(BlockSum(sum_of_squares) / w.in_features + norm_epsilon);"),
         "norm prologue: the block sums x's squares (every block the same) and computes inverse_rms"),
        (lines(f, "const int warp_index = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpSize;",
               "rows[r] = reinterpret_cast<const Bits*>(w.data) + static_cast<std::size_t>(row_index[r]) * vectors;"),
         "warp_index picks the warp's RowsPerWarp rows"),
        (lines(f, "for (int base = lane; base < vectors; base += kWarpSize * kLoadsInFlight) {",
               "bits[r][k] = v < vectors ? __ldcs(rows[r] + v) : Bits{};"),
         "each pass issues every load first: kLoadsInFlight chunks of each row"),
        (lines(f, "float4 a = __ldg(x_vectors + 2 * v);", "sums[r] += Dot8(bits[r][k], a, b);"),
         "then reads x's 8 values per chunk, normalizes them and adds a Dot8 for each row"),
        (lines(f, "sums[r] = WarpSum(sums[r]);", "y[(first_row + r) / 2] = Silu(sums[r]) * sums[r + 1];"),
         "WarpSum gives every lane the row's sum; lane 0 stores SiLU(gate) × up"),
    ], x_text=120)
    write("code_matvec.svg", y + 5 * 17 + 6, "MatVecVectorKernel on gate_up", b)


# --- PagedRow --------------------------------------------------------------

PAGE_SHIFT = 4                 # 16 positions per page (TransformerOptions::page_size)
PAGE_TABLE = [3, 8, 2, 6]      # an example sequence's pages, logical page 0 .. 3
POSITIONS = [5, 16, 37, 50]


def paged_row_of(pages, page_shift, position):
    """PagedRow: the physical page's first row, OR the offset in the page."""
    page = pages[position >> page_shift]
    return (page << page_shift) | (position & ((1 << page_shift) - 1))


def paged_row():
    """PagedRow on four positions: the position's bits split by the shift
    and the mask, the high part replaced by the page table entry, and where
    the resulting rows sit among the cache's pages."""
    size = 1 << PAGE_SHIFT
    bits, bw = 8, 17
    logical_ramps = ["teal", "purple", "coral", "green"]

    def bit_row(x, y, value, high_ramp, low_ramp):
        out = ""
        for k in range(bits):
            bit = (value >> (bits - 1 - k)) & 1
            ramp = high_ramp if k < bits - PAGE_SHIFT else low_ramp
            out += chip(x + k * bw, y, bw - 1, 18, ramp, str(bit), "tx", mono=True, rx=2)
        return out

    b = text(24, 22, f"PagedRow: page_shift = {PAGE_SHIFT}, so each page holds {size} positions", "th")
    b += text(24, 42, "pages (an example page table):", "ts")
    for k, page in enumerate(PAGE_TABLE):
        b += chip(214 + k * 56, 32, 52, 20, logical_ramps[k], f"[{k}] = {page}", "tx", mono=True)

    b += text(24, 78, "position", "tx") + text(110, 78, "bits", "tx")
    b += text(270, 78, "page = pages[position >> 4]", "tx mono")
    b += text(470, 78, "(page << 4) | (position & 15)", "tx mono")
    for r, p in enumerate(POSITIONS):
        y = 92 + r * 34
        logical = p >> PAGE_SHIFT
        page = PAGE_TABLE[logical]
        row = paged_row_of(PAGE_TABLE, PAGE_SHIFT, p)
        b += text(24, y + 9, str(p), "ts mono")
        b += bit_row(80, y, p, "gray", "teal")
        b += arrow(80 + bits * bw + 4, y + 9, 266, y + 9)
        b += chip(270, y, 160, 18, logical_ramps[logical], f"pages[{logical}] = {page}", "tx", mono=True)
        b += arrow(434, y + 9, 446, y + 9)
        b += bit_row(450, y, row, "amber", "teal")
        b += text(450 + bits * bw + 8, y + 9, f"= {row}", "ts mono")
    y = 92 + len(POSITIONS) * 34 + 2
    b += text(80, y, "high 4 bits: which page", "tx") + text(80, y + 14, "low 4 bits: offset in it (teal)", "tx")
    b += text(450, y, "amber: the physical page", "tx") + text(450, y + 14, "offset unchanged", "tx")

    # The cache: physical pages of 16 rows; this sequence's pages coloured.
    y += 40
    pages_shown = max(PAGE_TABLE) + 2
    pw = (656 - 24) / pages_shown
    rw = (pw - 4) / size
    b += text(24, y, "cache rows, page by page (row = page × 16 + offset); amber: the four rows above", "ts")
    y += 30
    for page in range(pages_shown):
        x = 24 + page * pw
        owner = PAGE_TABLE.index(page) if page in PAGE_TABLE else None
        ramp = logical_ramps[owner] if owner is not None else "gray"
        b += rect(x, y, pw - 4, 26, ramp, rx=2, extra="" if owner is not None else ' fill-opacity="0.35"')
        b += text(x + (pw - 4) / 2, y + 38, f"page {page}", "tx", "middle")
        b += text(x, y + 52, str(page * size), "tx mono")
    for p in POSITIONS:
        row = paged_row_of(PAGE_TABLE, PAGE_SHIFT, p)
        x = 24 + (row >> PAGE_SHIFT) * pw + (row & (size - 1)) * rw + rw / 2
        b += line(x, y - 4, x, y + 30, ACCENT["amber"], 2)
        b += text(x, y - 10, str(row), "tx mono", "middle")

    y += 76
    b += text(24, y, "Positions 0–15 of the sequence are in page 3, 16–31 in page 8, and so on; the pages need not be in order.", "ts")
    f = "engine/kernels/paged_kv.cuh"
    b += notes(y + 24, [
        (lines(f, "const int page = pages[position >> page_shift];"),
         "position >> page_shift is position / 16: the page table entry; it holds the physical page"),
        (lines(f, "return (static_cast<std::size_t>(page) << page_shift) | static_cast<std::size_t>(position & ((1 << page_shift) - 1));"),
         "page << page_shift is the page's first row; | (position & 15) adds position % 16"),
        ("transformer.cpp " + lines("engine/transformer.cpp",
                                     "host[layout_.row_cache + row] = (sequence.pages[position >> page_shift_] << page_shift_) | (position & mask);"),
         "the CPU computes the same row for each new token's cache_rows"),
    ], x_text=180)
    write("code_paged_row.svg", y + 24 + 3 * 17 + 6, "PagedRow on four positions", b)


# --- RopeAndCacheQkvKernel and RotateHead ----------------------------------

def rope():
    """RopeAndCacheQkvKernel's blocks for one row (12 query, 2 key and 2
    value slots) and where each writes, then RotateHead on a real key head:
    thread i turns the pair (i, i + 64) by position × inverse_frequencies[i]."""
    data = rope_data()
    heads, kv_heads, head_dim = data["num_heads"], data["num_kv_heads"], data["head_dim"]
    half = head_dim // 2
    slots = heads + 2 * kv_heads
    position = data["position"]
    key = data["key"]
    inverse_frequencies = [f32(1.0 / data["rope_theta"] ** (2.0 * i / head_dim)) for i in range(half)]
    angles = [f32(position * inverse_frequencies[i]) for i in range(half)]
    rotated = key[:]
    for i in range(half):
        a, c = key[i], key[i + half]
        rotated[i] = a * math.cos(angles[i]) - c * math.sin(angles[i])
        rotated[i + half] = c * math.cos(angles[i]) + a * math.sin(angles[i])

    b = text(24, 22, f"RopeAndCacheQkvKernel: one block of 64 threads per (row, slot), {slots} slots per row", "th")
    b += text(24, 42, f"one row of qkv: {heads} query heads, {kv_heads} key heads, {kv_heads} value heads, "
                      f"head_dim = {head_dim}; blockIdx.x = row × {slots} + slot", "ts")
    x0, sw, top = 24, (656 - 24) / slots, 70

    def kind(slot):
        if slot < heads:
            return "purple", f"q{slot}"
        if slot < heads + kv_heads:
            return "teal", f"k{slot - heads}"
        return "coral", f"v{slot - heads - kv_heads}"

    for slot in range(slots):
        ramp, name = kind(slot)
        b += text(x0 + slot * sw + sw / 2, top - 8, str(slot), "tx mono", "middle")
        b += chip(x0 + slot * sw + 1, top, sw - 2, 22, ramp, name, "tx", mono=True)
    # Where each group goes.
    q_end = x0 + heads * sw
    k_mid = x0 + (heads + kv_heads / 2) * sw
    v_mid = x0 + (heads + 1.5 * kv_heads) * sw
    dest = top + 64
    b += arrow(x0 + heads * sw / 2, top + 26, x0 + heads * sw / 2, dest - 2, ACCENT["purple"])
    b += text(x0 + heads * sw / 2 + 6, top + 42, "RotateHead(in, out, …)", "tx mono")
    b += chip(x0, dest, heads * sw - 2, 20, "purple", f"q row: out = q + (row × {heads} + slot) × {head_dim}", "tx", mono=True)
    b += arrow(k_mid, top + 26, k_mid, dest - 2, ACCENT["teal"])
    b += arrow(v_mid, top + 26, v_mid, dest - 2, ACCENT["coral"])
    b += text(k_mid - 6, top + 42, "RotateHead", "tx mono", "end")
    b += text(v_mid + 6, top + 42, "copy", "tx mono")
    b += chip(q_end + 1, dest, kv_heads * sw - 2, 20, "teal", "staged", "tx", mono=True)
    b += chip(q_end + kv_heads * sw + 1, dest, kv_heads * sw - 2, 20, "coral", "staged", "tx", mono=True)
    b += text(656, dest + 34, "__syncthreads(), then StoreHead writes staged (BF16 by default) to", "tx", "end")
    b += text(656, dest + 48, "cache.keys / cache.values at row cache_rows[row], head slot − 12 / − 14", "tx mono", "end")

    # RotateHead on the real key head: column i = thread i.
    y = dest + 76
    b += text(24, y, f'RotateHead on k0 of "{data["token"]}" at position {position} (layer {data["layer"]}): '
                     f"thread i turns the pair (in[i], in[i + {half}])", "ts")
    cx0, cw = 120, (656 - 120) / half
    peak = max(abs(v) for v in key + rotated)
    rows = [("in[i]", key[:half]), (f"in[i + {half}]", key[half:]),
            ("out[i]", rotated[:half]), (f"out[i + {half}]", rotated[half:])]
    ry = y + 22
    b += text(cx0 - 6, ry - 8, "thread i", "tx", "end")
    for i in (0, 16, 32, 48, 63):
        b += text(cx0 + i * cw + cw / 2, ry - 8, str(i), "tx mono", "middle")
    for r, (name, vals) in enumerate(rows):
        yy = ry + r * 15 + (6 if r >= 2 else 0)
        b += text(cx0 - 6, yy + 6, name, "tx mono", "end")
        for i, v in enumerate(vals):
            color, opacity = value_color(v, peak)
            b += fill_rect(cx0 + i * cw + 0.3, yy, cw - 0.6, 13, color, opacity)

    # Dials: the pair as a 2D point before (gray) and after (amber).
    picks = [0, 1, 2, 4, 8, 16, 32, 63]
    dial_y = ry + 4 * 15 + 6 + 52
    dial_x0, dial_step, radius = 60, (656 - 60) / len(picks), 22
    for n, i in enumerate(picks):
        cx = dial_x0 + n * dial_step
        b += line(cx0 + i * cw + cw / 2, ry + 4 * 15 + 8, cx, dial_y - radius - 6, ACCENT["gray"], 0.6)
        b += f'<circle cx="{cx:.1f}" cy="{dial_y:.1f}" r="{radius}" fill="none" class="ln" stroke-width="0.6"/>'
        before = math.atan2(key[i + half], key[i])
        after = before + angles[i]
        p0 = (cx + radius * math.cos(before), dial_y - radius * math.sin(before))
        p1 = (cx + radius * math.cos(after), dial_y - radius * math.sin(after))
        b += line(cx, dial_y, *p0, ACCENT["gray"], 1.4) + circle(*p0, 2.5, ACCENT["gray"])
        b += line(cx, dial_y, *p1, ACCENT["amber"], 1.6) + circle(*p1, 2.5, ACCENT["amber"])
        sweep = angles[i] % (2 * math.pi)
        if sweep > 0.02:
            r_arc = radius * 0.55
            a0 = (cx + r_arc * math.cos(before), dial_y - r_arc * math.sin(before))
            a1 = (cx + r_arc * math.cos(after), dial_y - r_arc * math.sin(after))
            large = 1 if sweep > math.pi else 0
            b += path(f"M{a0[0]:.1f} {a0[1]:.1f} A{r_arc:.1f} {r_arc:.1f} 0 {large} 0 {a1[0]:.1f} {a1[1]:.1f}",
                      ACCENT["amber"], 1, arrow_end=False)
        b += text(cx, dial_y + radius + 12, f"i = {i}", "tx mono", "middle")
        b += text(cx, dial_y + radius + 26, f"{angles[i]:.3g} rad", "tx mono", "middle")
    y = dial_y + radius + 48
    b += text(24, y, f"angle = position × inverse_frequencies[i], with inverse_frequencies[i] = "
                     f"{data['rope_theta']:,.0f}^(−2i / {head_dim}).", "ts")
    b += text(24, y + 18, f"Gray: (a, b) = (in[i], in[i + {half}]); amber: (a cos − b sin, b cos + a sin).", "ts")
    b += text(24, y + 36, f"Pair 0 turns {angles[0]:g} rad; the last pairs, which hold the largest values, barely move.", "ts")
    y += 18
    f = "engine/kernels/rope.cu"
    b += notes(y + 44, [
        (lines(f, "const int slots = num_heads + 2 * num_kv_heads;", "const float* in = qkv + static_cast<std::size_t>(row) * slots * head_dim"),
         "blockIdx.x gives the row and the slot; in points at that head's 128 values"),
        (lines(f, "if (slot < num_heads) {\n        float* out", "return;\n    }\n    const bool key"),
         "a query slot rotates into q and is done"),
        (lines(f, "const bool key = slot < num_heads + num_kv_heads;", "cache_row, num_kv_heads, head);\n}"),
         "a key slot rotates into staged, a value slot copies; after the barrier StoreHead writes the cache row"),
        (lines(f, "const int half = head_dim / 2;\n    for (int i = threadIdx.x;", "out[i + half] = b * cos_angle + a * sin_angle;"),
         "RotateHead: 64 threads, one pair each (half = 64)"),
    ])
    write("code_rope.svg", y + 44 + 4 * 17 + 6, "RopeAndCacheQkvKernel and RotateHead", b)


# --- QuantizeToFp8Kernel ---------------------------------------------------

def e4m3_round(v):
    """v rounded to the nearest E4M3 value (ties to even), saturating at 448
    like __nv_cvt_float_to_fp8(..., __NV_SATFINITE, __NV_E4M3)."""
    magnitude = abs(v)
    if magnitude == 0:
        return 0.0
    exponent = max(math.floor(math.log2(magnitude)), -6)   # below 2^-6: subnormal steps of 2^-9
    step = 2.0 ** (exponent - 3)
    rounded = min(round(magnitude / step) * step, FP8_MAX)
    return math.copysign(rounded, v)


def e4m3_neighbours(v):
    """The E4M3 values just below and above |v|, and the step between them."""
    magnitude = abs(v)
    exponent = max(math.floor(math.log2(magnitude)), -6) if magnitude else -6
    step = 2.0 ** (exponent - 3)
    low = math.floor(magnitude / step) * step
    return low, min(low + step, FP8_MAX), step


def quantize_fp8():
    """QuantizeToFp8Kernel on one real row: which thread reads which weight,
    each thread's largest |w|, the block's maximum and the scale, then six
    weights divided by the scale and rounded to the nearest E4M3 value."""
    data = load("quant_weights.json")
    row = [bf16(v) for v in data["row"]]
    cols, threads = len(row), 256
    per_thread = cols // threads
    largest = [max(abs(row[i]) for i in range(t, cols, threads)) for t in range(threads)]
    block_max = max(largest)
    scale = f32(block_max / FP8_MAX)
    at_max = max(range(cols), key=lambda i: abs(row[i]))
    q = [e4m3_round(f32(w / scale)) for w in row]

    b = text(24, 22, f"QuantizeToFp8Kernel on one row: layer {data['layer']} down_proj, row 0 (blockIdx.x = 0), "
                     f"cols = {cols:,}", "th")
    b += text(24, 42, f"256 threads; thread t reads i = t, t + 256, … ({per_thread} weights each). Colour: the weight's sign and size.", "ts")
    x0, cell, top, rh = 70, 2.2, 74, 2.6
    b += text(x0, top - 16, "threadIdx.x", "tx")
    for t in (0, 64, 128, 192, 255):
        b += text(x0 + t * cell + cell / 2, top - 5, str(t), "tx mono", "middle")
    b += value_cells([(x0 + (i % threads) * cell, top + (i // threads) * rh, cell, w) for i, w in enumerate(row)], rh,
                     block_max)
    grid_bottom = top + per_thread * rh
    b += text(x0 - 6, top + 4, "i = t", "tx mono", "end")
    b += text(x0 - 6, grid_bottom - 4, f"t + {(per_thread - 1) * threads:,}", "tx mono", "end")
    mx, my = x0 + (at_max % threads) * cell + cell / 2, top + (at_max // threads) * rh + rh / 2
    b += f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="5" fill="none" stroke="{ACCENT["amber"]}" stroke-width="1.6"/>'
    b += text(x0 + threads * cell, top - 16, f"circled: the largest |w|, i = {at_max:,} (thread {at_max % threads})", "tx", "end")

    y = grid_bottom + 8
    for t, v in enumerate(largest):
        h = v / block_max * 22
        b += fill_rect(x0 + t * cell, y + 22 - h, cell - 0.3, h, ACCENT["purple"], 0.85)
    b += text(x0 - 6, y + 11, "largest", "tx mono", "end")
    y += 44
    b += text(24, y, f"BlockMax → {block_max:.5g};  scale = {block_max:.5g} / 448 = {scale:.4e}", "ts mono")
    b += chip(470, y - 10, 186, 20, "amber", f"row_scales[0] = {scale:.4e}", "tx", mono=True)

    # Six weights through the division and the rounding.
    picks = [at_max, 0, 1, 2, 3, min((i for i in range(cols) if row[i]), key=lambda i: abs(row[i]))]
    y += 30
    heads = [(24, "i"), (66, "w"), (140, "w / scale"), (236, "E4M3 values around |w / scale|"), (420, "stored"),
             (480, "× scale"), (560, "error")]
    for hx, head in heads:
        b += text(hx, y, head, "tx mono" if head in ("i", "w", "w / scale") else "tx")
    strip_x, strip_w = 236, 160
    for n, i in enumerate(picks):
        ry = y + 14 + n * 24
        w = row[i]
        scaled = f32(w / scale)
        stored = q[i]
        back = stored * scale
        b += text(24, ry + 8, f"{i:,}", "tx mono") + text(66, ry + 8, f"{w:+.5f}" if abs(w) >= 1e-3 else f"{w:+.1e}", "tx mono")
        b += text(140, ry + 8, f"{scaled:+.3f}", "tx mono")
        low, high, step = e4m3_neighbours(scaled)
        lo2, hi2 = max(low - step, 0.0), high + step

        def sx(v):
            return strip_x + (v - lo2) / (hi2 - lo2) * strip_w

        b += line(strip_x, ry + 8, strip_x + strip_w, ry + 8, None, 0.6)
        for g in (lo2, low, high, hi2):
            if g <= FP8_MAX:
                chosen = abs(g - abs(stored)) < 1e-12
                b += line(sx(g), ry + 2, sx(g), ry + 14, ACCENT["amber"] if chosen else ACCENT["gray"], 2 if chosen else 1)
        b += circle(sx(abs(scaled)), ry + 8, 3, ACCENT["teal"])
        b += chip(416, ry, 54, 16, "amber", f"{stored:+g}", "tx", mono=True, rx=2)
        b += text(480, ry + 8, "0" if back == 0 else f"{back:+.5f}" if abs(back) >= 1e-3 else f"{back:+.1e}", "tx mono")
        b += text(560, ry + 8, f"{back - w:+.1e}", "tx mono")
    y += 14 + len(picks) * 24 + 8
    b += text(24, y, "Ticks: E4M3 values around |w / scale| (dot); amber: the one stored. The largest weight lands on 448. Steps", "ts")
    b += text(24, y + 18, "grow with the size, so the error is at most 1/16 of the weight, down to 2^-6; below that, steps are 2^-9.", "ts")
    f = "engine/kernels/quantize.cu"
    b += notes(y + 44, [
        (lines(f, "float largest = 0.0f;\n    for (int i = threadIdx.x; i < cols; i += blockDim.x) {",
               "largest = fmaxf(largest, fabsf(Bf16ToFloat(row[i])));"),
         "each thread's largest |w| over its share of the row"),
        (lines(f, "BlockMax(largest);\n    const float scale = largest > 0.0f ? largest / kFp8Max : 1.0f;\n"
                  "    for (int i = threadIdx.x; i < cols",
               "largest / kFp8Max : 1.0f;\n    for (int i = threadIdx.x; i < cols"),
         "BlockMax gives every thread the row's largest; scale maps it to 448"),
        (lines(f, "row_out[i] = __nv_cvt_float_to_fp8(Bf16ToFloat(row[i]) / scale, __NV_SATFINITE, __NV_E4M3);"),
         "every weight divided by the scale, rounded to the nearest E4M3 byte"),
        (lines(f, "row_scales[blockIdx.x] = scale;\n    }\n}\n\n// The 4-bit code"),
         "thread 0 stores the scale; MatVec multiplies each row's sum by it"),
    ])
    write("code_quantize_fp8.svg", y + 44 + 4 * 17 + 6, "QuantizeToFp8Kernel on one row", b)


DIAGRAMS = [warp_sum, block_reduce, rms_norm, matvec, paged_row, rope, quantize_fp8]

if __name__ == "__main__":
    for make in DIAGRAMS:
        make()
    print(f"{len(DIAGRAMS)} code pictures written")
