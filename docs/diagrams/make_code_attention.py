"""Code pictures for the attention kernels (code tour, attention chapters):
CausalAttentionKernel, SplitAttentionKernel + CombineSplitsKernel and
FlashAttentionKernel's online softmax, each drawn as it runs on real
queries, keys and values, with its own variables after every step and the
lines of src/ that do each part.

The pictures replay the kernels in Python, step for step: same thread,
warp and slice mappings, same constants, same order of operations.
data/code_attention.json holds layer 1's rotated queries and its cached keys
and values (BF16, like the default cache) for two inputs: the sentence
"The capital of France is" and Mei's chat (the system prompt of
murmur-lab chat, "What's good tonight?" and her greedy reply from
`murmur-lab generate`). It is refreshed from models/qwen2.5-1.5b-instruct
when the model is present and numpy and tokenizers are installed
(reference/.venv has both): the script then runs layers 0 and 1 itself,
FP32 math on the BF16 weights, as the engine does.

Run from anywhere: python docs/diagrams/make_code_attention.py
"""
import base64
import functools
import importlib.util
import json
import math
import os
import re
import struct

from code_lines import lines
from svg_kit import ACCENT, fill_rect, label_in, line, rect, svg, text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ROOT = os.path.join(HERE, "..", "..")
MODEL = os.path.join(ROOT, "models", "qwen2.5-1.5b-instruct")
PERSONAS = os.path.join(ROOT, "src", "game", "personas.hpp")

LAYER = 1
SENTENCE = "The capital of France is"
SENTENCE_HEAD = 4   # the head data/whole_picture.json shows
CHAT_HEAD = 2       # its largest score moves between key tiles, so the rescaling is visible
QUESTION = "What's good tonight?"
REPLY = ("Tonight, I've got a special dish for you: a bowl of miso soup with a side of seaweed salad. "
         "It's the perfect start to a night out in the neon-lit night market.")
FLASH_ROW = 140     # a prompt row in the last row tile, so its key tile holds the causal diagonal

# The kernels' constants (attention.cu, decode_attention.cu) and the engine defaults they meet.
THREADS = 128            # kThreadsPerBlock
WARP = 32                # kWarpSize
WARPS = THREADS // WARP  # kWarpsPerBlock
PAGE_SHIFT = 4           # page_size 16 (TransformerOptions)
TILE = 64                # kTile (decode slices) and kFlashKeys (flash key tiles)
FLASH_ROWS = 64          # kFlashRows
BLOCKS_PER_SM = 8        # kBlocksPerSm
SM_COUNT = 80            # RTX 4080 SUPER (murmur-lab devices), the GPU the docs measure on
MAX_CONTEXT = 4096       # TransformerOptions::max_context
LOG2_E = 1.4426950408889634

WARP_RAMPS = ["teal", "purple", "green", "gray"]
SLICE_RAMPS = ["teal", "purple", "green"]


def write(name, height, title, body):
    svg(HERE, name, height, title, body)


# --- Data --------------------------------------------------------------------

def bf16_bits(np, values):
    """FP32 values rounded to the nearest BF16 (ties to even), as uint16 bits."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")


def bf16_row(encoded):
    """One cached row: base64 of little-endian BF16 bits, back to floats."""
    raw = base64.b64decode(encoded)
    return [struct.unpack("<f", struct.pack("<I", b << 16))[0] for b in struct.unpack(f"<{len(raw) // 2}H", raw)]


def layer_qkv(np, config, ids):
    """Layer LAYER's rotated queries, keys and values for the token ids, from
    the model's weights: layers before it run whole (attention, MLP), FP32
    math on BF16 weights. Shapes (tokens, heads, head_dim)."""
    path = os.path.join(MODEL, "model.safetensors")
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_size)

    def weight(name, rows=None):
        begin, end = header[name]["data_offsets"]
        bits = raw[begin:end].view("<u2").reshape(header[name]["shape"])
        bits = bits if rows is None else bits[rows]
        return (bits.astype(np.uint32) << 16).view(np.float32)

    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    dim = config["hidden_size"] // heads
    count = len(ids)
    angles = np.arange(count)[:, None] * config["rope_theta"] ** (-np.arange(0, dim, 2) / dim)
    cos, sin = np.cos(angles)[:, None, :].astype(np.float32), np.sin(angles)[:, None, :].astype(np.float32)

    def norm(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + config["rms_norm_eps"]) * w

    def rope(x):   # pairs (i, i + dim/2), like the model's rotate_half
        a, b = x[..., : dim // 2], x[..., dim // 2:]
        return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)

    x = weight("model.embed_tokens.weight", ids)
    for n in range(LAYER + 1):
        prefix = f"model.layers.{n}."

        def w(name):
            return weight(prefix + name)

        h = norm(x, w("input_layernorm.weight"))
        q, k, v = (h @ w(f"self_attn.{p}_proj.weight").T + w(f"self_attn.{p}_proj.bias") for p in "qkv")
        q, k, v = (rope(q.reshape(count, heads, dim)), rope(k.reshape(count, kv_heads, dim)),
                   v.reshape(count, kv_heads, dim))
        if n == LAYER:
            return q, k, v
        group = heads // kv_heads
        scores = np.einsum("thd,shd->hts", q, np.repeat(k, group, axis=1)) / np.sqrt(dim)
        scores += np.triu(np.full((count, count), -np.inf, dtype=np.float32), 1)
        weights = np.exp(scores - scores.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        x = x + np.einsum("hts,shd->thd", weights, np.repeat(v, group, axis=1)).reshape(count, -1) @ w(
            "self_attn.o_proj.weight").T
        h = norm(x, w("post_attention_layernorm.weight"))
        gate = h @ w("mlp.gate_proj.weight").T
        x = x + (gate / (1 + np.exp(-gate)) * (h @ w("mlp.up_proj.weight").T)) @ w("mlp.down_proj.weight").T


def model_snapshot():
    """What the pictures need from the model: per input, the pieces, the
    shown head's query rows (FP32) and its KV head's keys and values (BF16)."""
    import numpy as np
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
    with open(os.path.join(MODEL, "config.json")) as f:
        config = json.load(f)
    with open(PERSONAS, encoding="utf-8") as f:
        literal = re.search(r'kNoodleVendor =\s*((?:"[^"]*"\s*)+);', f.read()).group(1)
    system = "".join(re.findall(r'"([^"]*)"', literal))
    chat = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{QUESTION}<|im_end|>\n"
            f"<|im_start|>assistant\n{REPLY}")

    def case(prompt, head, rows):
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        q, k, v = layer_qkv(np, config, ids)
        kv_head = head // (q.shape[1] // k.shape[1])
        rows = {name: (len(ids) - 1 if row is None else row) for name, row in rows.items()}
        return {"pieces": [tokenizer.decode([i]) for i in ids], "head": head, "kv_head": kv_head,
                "rows": rows, "q": {name: [float(f"{x:.7g}") for x in q[row, head]] for name, row in rows.items()},
                "keys": [base64.b64encode(bf16_bits(np, r).tobytes()).decode() for r in k[:, kv_head]],
                "values": [base64.b64encode(bf16_bits(np, r).tobytes()).decode() for r in v[:, kv_head]]}

    return {"layer": LAYER, "num_heads": config["num_attention_heads"],
            "num_kv_heads": config["num_key_value_heads"],
            "head_dim": config["hidden_size"] // config["num_attention_heads"],
            "encoding": "keys/values: one base64 string per row, little-endian BF16 bits",
            "sentence": case(SENTENCE, SENTENCE_HEAD, {"query": None}),
            "chat": case(chat, CHAT_HEAD, {"decode": None, "flash": FLASH_ROW})}


@functools.cache
def attention_data():
    """data/code_attention.json, refreshed from the model when it can be,
    with keys and values decoded to floats."""
    path = os.path.join(DATA, "code_attention.json")
    if os.path.exists(MODEL) and all(importlib.util.find_spec(m) for m in ("numpy", "tokenizers")):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(model_snapshot(), f, ensure_ascii=False, indent=1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for case in (data["sentence"], data["chat"]):
        case["keys"] = [bf16_row(r) for r in case["keys"]]
        case["values"] = [bf16_row(r) for r in case["values"]]
    return data


# --- Drawing helpers ---------------------------------------------------------

def heat(x, y, values, cell, h, peak=None):
    """A vector as a row of cells: blue for positive, coral for negative,
    darker for larger magnitude (relative to `peak`, by default the row's own)."""
    peak = peak or max(abs(v) for v in values) or 1.0
    b = rect(x - 1, y - 1, len(values) * cell + 2, h + 2, "gray", rx=2)
    for i, v in enumerate(values):
        b += fill_rect(x + i * cell, y, cell, h, ACCENT["blue" if v >= 0 else "coral"], min(abs(v) / peak, 1.0) ** 0.5)
    return b


def notes(y, items):
    """The 'lines a–b: what they do' footer; returns (svg, bottom y)."""
    b = ""
    for k, (where, what) in enumerate(items):
        b += text(24, y + k * 17, where, "tx mono") + text(130, y + k * 17, what, "tx")
    return b, y + len(items) * 17


def barrier(y, label="__syncthreads()"):
    return line(24, y, 560, y, dash=True) + text(656, y, label, "tx mono", "end")


def fmt(v, digits=2):
    return "−∞" if v == -math.inf else f"{v:.{digits}f}".replace("-", "−")


# --- CausalAttentionKernel -----------------------------------------------------

def replay_causal(q, keys, values, position, scale):
    """CausalAttentionKernel for one block (one head, one query row)."""
    head_dim = len(q)
    # 1. Scores: warp w takes keys j = w, w + WARPS, ...; lane l adds dims l, l + 32, ...
    lane_sums = [[sum(q[d] * keys[j][d] for d in range(lane, head_dim, WARP)) for lane in range(WARP)]
                 for j in range(position + 1)]
    scores = [sum(lanes) * scale for lanes in lane_sums]   # WarpSum, then lane 0 writes weights[j]
    # 2. Softmax: thread t owns weights[t], weights[t + 128], ...
    max_score = max(scores)
    exps = [math.exp(s - max_score) for s in scores]
    total = sum(exps)
    weights = [e / total for e in exps]
    # 3. Mix: thread d sums weights[j] * v_j[d] over j.
    out = [sum(weights[j] * values[j][d] for j in range(position + 1)) for d in range(head_dim)]
    return {"lane_sums": lane_sums, "scores": scores, "max_score": max_score, "exps": exps, "total": total,
            "weights": weights, "out": out}


def causal_attention():
    """CausalAttentionKernel for the block (head 4, row 4) of "The capital of
    France is": phase 1 scores (warps over keys, lanes over dims), phase 2
    softmax (threads over keys), phase 3 mix (threads over dims)."""
    data = attention_data()
    case = data["sentence"]
    head, kv_head, dim = case["head"], case["kv_head"], data["head_dim"]
    position = case["rows"]["query"]
    q = case["q"]["query"]
    r = replay_causal(q, case["keys"], case["values"], position, 1 / math.sqrt(dim))
    group = data["num_heads"] // data["num_kv_heads"]
    pieces = case["pieces"]

    sx, cell = 132, 2.75                  # strips: x of dim 0, width of one dim
    pitch = 18                            # height of one key row in phase 1
    bx, bar = 494, 3                      # lane sums: x of lane 0, width of one lane
    vx, vw = 604, 52                      # weights[j]
    b = text(24, 22, f'CausalAttentionKernel, block (head {head}, row {position}): "{pieces[position].strip()}" '
                     f"attends to keys 0–{position}", "th")
    b += text(24, 42, f'Layer {data["layer"]} on "{SENTENCE}", real values, BF16 cache. {THREADS} threads = {WARPS} '
                      f"warps, head_dim {dim}.", "ts")
    b += text(24, 60, f"kv_head = {head} / ({data['num_heads']} / {data['num_kv_heads']}) = {kv_head}, "
                      f"so keys and values come from KV head {kv_head} (shared by heads "
                      f"{kv_head * group}–{kv_head * group + group - 1}).", "tx")
    b += text(24, 76, f"PagedRow(pages, {PAGE_SHIFT}, j) = pages[j >> {PAGE_SHIFT}] · {1 << PAGE_SHIFT} + "
                      f"(j & {(1 << PAGE_SHIFT) - 1}): keys 0–{position} are rows 0–{position} of page pages[0].",
              "tx mono")

    # Phase 1: scores.
    y = 98
    b += text(24, y, "1  Scores: warp w takes keys j = w, w + 4, …", "ts")
    b += text(sx, y + 18, "query[d] and k_j[d], d = 0..127", "tx")
    b += text(bx, y + 18, "lane sums 0..31", "tx") + text(vx + vw, y + 18, "weights[j]", "tx", "end")
    qy = y + 34
    b += text(24, qy + 5, "query", "tx mono") + heat(sx, qy, q, cell, 10)
    for d in range(0, dim, WARP):
        cx = sx + d * cell + cell / 2
        b += line(cx, qy - 3, cx, qy + 16 + (position + 1) * pitch, width=0.8, dash=True)
    peak = max(abs(s) for lanes in r["lane_sums"] for s in lanes)
    for j in range(position + 1):
        ky = qy + 20 + j * pitch
        warp = j % WARPS
        ramp = WARP_RAMPS[warp]
        b += text(24, ky + 6, f"{j} {pieces[j].strip()}", "tx mono")
        b += rect(84, ky, 42, 13, ramp) + label_in(84, ky, 42, 13, ramp, f"warp {warp}", "tx")
        b += heat(sx, ky + 1, case["keys"][j], cell, 11)
        mid = ky + 6.5
        b += line(bx, mid, bx + WARP * bar, mid, width=0.5)
        for lane, s in enumerate(r["lane_sums"][j]):
            hgt = math.copysign(math.sqrt(abs(s) / peak), s) * 7   # square root: small lanes stay visible
            b += fill_rect(bx + lane * bar, min(mid, mid - hgt), bar - 0.6, abs(hgt), ACCENT[ramp])
        b += rect(vx, ky, vw, 13, "amber") + label_in(vx, ky, vw, 13, "amber", fmt(r["scores"][j]), "tx", True)
    cy = qy + 20 + (position + 1) * pitch + 6
    b += text(24, cy, "Lane l adds query[d] · k_j[d] for d = l, l+32, l+64, l+96 (dashed: lane 0). WarpSum adds the "
                      "32 lane sums;", "tx")
    b += text(24, cy + 15, f"lane 0 writes weights[j] = dot · scale, scale = 1/√{dim}. Key {position} is warp 0's "
                           f"second key (j = 0 + 4).", "tx")
    y = cy + 28
    b += barrier(y)

    # Phase 2: softmax.
    y += 20
    b += text(24, y, f"2  Softmax: thread j owns weights[j]; threads {position + 1}–{THREADS - 1} hold no key", "ts")
    cw, gap = 66, 4
    for j in range(position + 1):
        b += text(sx + j * (cw + gap) + cw / 2, y + 18, f"thread {j}", "tx", "middle")
    rows = [("weights[j]", r["scores"], "teal", f"BlockMax: max_score = {fmt(r['max_score'])}", 2),
            ("expf(w − max)", r["exps"], "amber", f"BlockSum: {fmt(r['total'], 4)}", 4),
            ("× inverse_sum", r["weights"], "amber", f"inverse_sum = 1 / {fmt(r['total'], 4)}", 4)]
    for k, (label, vals, ramp, result, digits) in enumerate(rows):
        ry = y + 28 + k * 22
        b += text(24, ry + 9, label, "tx mono")
        for j, v in enumerate(vals):
            x = sx + j * (cw + gap)
            if k == 2:   # the probabilities as filled fractions of their cells
                b += rect(x, ry, cw, 18, "gray") + fill_rect(x, ry, cw * v, 18, ACCENT["amber"], 0.55)
                b += text(x + cw / 2, ry + 9, fmt(v, digits), "ts mono", "middle")
            else:
                b += rect(x, ry, cw, 18, ramp) + label_in(x, ry, cw, 18, ramp, fmt(v, digits), "tx", True)
        b += text(sx + (position + 1) * (cw + gap) + 6, ry + 9, result, "tx mono")
    y += 28 + 3 * 22 + 6
    b += barrier(y)

    # Phase 3: mix.
    y += 20
    b += text(24, y, "3  Mix: thread d sums weights[j] · v_j[d] over j = 0..4; threads d and d+1 read neighbouring floats",
              "ts")
    shown_d = 77
    top = y + 26
    vpeak = max(abs(v) for row in case["values"][: position + 1] for v in row)
    for j in range(position + 1):
        ry = top + j * 15
        b += text(sx - 6, ry + 6, f"{fmt(r['weights'][j], 3)} × v_{j}", "tx mono", "end")
        b += heat(sx, ry + 1, case["values"][j], cell, 11, vpeak)
    oy = top + (position + 1) * 15 + 6
    b += text(sx - 6, oy + 6, "out_row", "tx mono", "end") + heat(sx, oy, r["out"], cell, 12, vpeak)
    b += rect(sx - 2, oy - 2, dim * cell + 4, 16, "amber", extra=' fill-opacity="0"')
    hx = sx + shown_d * cell
    b += f'<rect x="{hx - 1:.1f}" y="{top - 1:.1f}" width="{cell + 2:.1f}" height="{oy + 13 - top:.1f}" fill="none" ' \
         f'stroke="{ACCENT["amber"]}" stroke-width="1.2"/>'
    b += text(hx + cell / 2, top - 8, f"thread {shown_d}", "tx", "middle")
    terms = " + ".join(f"{r['weights'][j]:.3f}·{case['values'][j][shown_d]:.2f}" for j in range(position + 1))
    b += text(24, oy + 30, f"out_row[{shown_d}] = {terms} = {r['out'][shown_d]:.3f}".replace("-", "−"),
              "tx mono")

    f = "engine/kernels/attention.cu"
    foot, bottom = notes(oy + 50, [
        (lines(f, "rows.sequences[row]) * kv.page_table_stride;", "query[d] = q_row[d];"),
         "pick the KV head and the sequence's page table; copy the query row to shared memory"),
        (lines(f, "// 1. Scores.", "weights[j] = dot * scale;"),
         "warps over keys, lanes over dims, WarpSum; one score per visible key in shared memory"),
        (lines(f, "// 2. Softmax.", "weights[j] *= inverse_sum;"),
         "block max, e^(s − max), block sum, scale by 1 / sum"),
        (lines(f, "// 3. Weighted mix of the values.", "out_row[d] = acc;"),
         "one output dim per thread, summed over every visible key's value"),
    ])
    write("code_causal_attention.svg", bottom + 4, "CausalAttentionKernel step by step", b + foot)


# --- SplitAttentionKernel + CombineSplitsKernel ---------------------------------

def splits_for(context):
    return (context + TILE - 1) // TILE


def blocks_per_sequence(batch, num_kv_heads, max_context):
    """BlocksPerSequence: enough blocks to fill the GPU, at most one per slice."""
    pairs = batch * num_kv_heads
    want = (BLOCKS_PER_SM * SM_COUNT + pairs - 1) // pairs
    return max(1, min(want, splits_for(max_context)))


def replay_split(q, keys, values, context, scale):
    """SplitAttentionKernel's partial result per slice, then CombineSplitsKernel."""
    head_dim = len(q)
    slices = []
    for s in range(splits_for(context)):
        begin = s * TILE
        n = min(TILE, context - begin)
        scores = [sum(a * b for a, b in zip(q, keys[begin + j])) * scale for j in range(n)]
        slice_max = max(scores)
        p = [math.exp(x - slice_max) for x in scores]
        partial = [sum(p[j] * values[begin + j][d] for j in range(n)) for d in range(head_dim)]
        slices.append({"begin": begin, "n": n, "scores": scores, "p": p, "max": slice_max, "sum": sum(p),
                       "partial": partial})
    overall_max = max(s["max"] for s in slices)
    scales = [math.exp(s["max"] - overall_max) for s in slices]
    total = sum(c * s["sum"] for c, s in zip(scales, slices))
    weights = [c / total for c in scales]
    out = [sum(w * s["partial"][d] for w, s in zip(weights, slices)) for d in range(head_dim)]
    all_scores = [x for s in slices for x in s["scores"]]
    m = max(all_scores)
    e = [math.exp(x - m) for x in all_scores]
    exact = [sum(e[j] * values[j][d] for j in range(context)) / sum(e) for d in range(head_dim)]
    return {"slices": slices, "overall_max": overall_max, "scales": scales, "total": total, "weights": weights,
            "out": out, "exact": exact}


def split_attention():
    """One decode step of Mei's chat for head 2: the grid, each slice's max,
    sum and partial output, and CombineSplitsKernel merging them into the
    softmax over all keys."""
    data = attention_data()
    case = data["chat"]
    head, kv_head, dim = case["head"], case["kv_head"], data["head_dim"]
    position = case["rows"]["decode"]
    context = position + 1
    r = replay_split(case["q"]["decode"], case["keys"], case["values"], context, 1 / math.sqrt(dim))
    slices = r["slices"]
    group = data["num_heads"] // data["num_kv_heads"]

    b = text(24, 22, f"SplitAttentionKernel, then CombineSplitsKernel: one decode step, head {head}", "th")
    b += text(24, 42, f'Layer {data["layer"]} of Mei\'s chat: the new token "{case["pieces"][position].strip()}" at '
                      f"position {position} attends to {context} keys: SplitsFor({context}) = {len(slices)} slices "
                      f"of {TILE}.", "ts")

    # The grid.
    cx, kw = 132, 3.3     # keys and dims: x of index 0, width of one
    gc = 6                # grid: width of one cell
    one = blocks_per_sequence(1, data["num_kv_heads"], MAX_CONTEXT)
    many = blocks_per_sequence(64, data["num_kv_heads"], MAX_CONTEXT)
    y = 64
    b += text(24, y, "grid = (num_kv_heads, BlocksPerSequence, batch); block (x, y, z) takes slices y, y + gridDim.y, … "
                     "of sequence z.", "tx")
    b += text(24, y + 15, f"BlocksPerSequence = ⌈{SM_COUNT} SMs × {BLOCKS_PER_SM} / (batch × {data['num_kv_heads']})⌉, "
                          f"at most SplitsFor({MAX_CONTEXT}) = {splits_for(MAX_CONTEXT)}: {one} for batch 1, "
                          f"{many} for batch 64.", "tx")
    rows = [("batch 1: blocks y", [SLICE_RAMPS[i] if i < len(slices) else None for i in range(one)],
             f"{len(slices)}–{one - 1}: no slice"),
            ("batch 64: slices", ["teal" if i % many == 0 else None for i in range(splits_for(MAX_CONTEXT))],
             "teal: block y = 0")]
    for k, (label, ramps, note) in enumerate(rows):
        ry = y + 26 + k * 16
        b += text(24, ry + 5, label, "tx mono")
        for i, ramp in enumerate(ramps):
            b += rect(cx + i * gc, ry, gc - 1, 10, ramp or "gray", rx=1, extra="" if ramp else ' fill-opacity="0.3"')
        b += text(cx + len(ramps) * gc + 8, ry + 5, note, "tx")
    b += text(24, y + 70, f"Each block scores query heads {kv_head * group}–{kv_head * group + group - 1} together "
                          f"(they share kv_head {kv_head}), reading each key once. Shown: head {head}.", "tx")

    # One slice per block: scores, then e^(s - max_s).
    top, sh, ph = 166, 56, 30
    lo, hi = min(min(s["scores"]) for s in slices), r["overall_max"]
    b += text(24, top + sh / 2, "scores[h][j]", "tx mono") + text(24, top + sh + 8 + ph / 2, "e^(s − max_s)",
                                                                   "tx mono")
    for i, s in enumerate(slices):
        x0, ramp = cx + s["begin"] * kw, SLICE_RAMPS[i]
        b += fill_rect(x0, top - 18, s["n"] * kw - 2, sh + ph + 26, ACCENT[ramp], 0.08, rx=3)
        b += text(x0 + 3, top - 9, f"slice {i}: keys {s['begin']}–{s['begin'] + s['n'] - 1}", "tx")
        for j, (score, p) in enumerate(zip(s["scores"], s["p"])):
            x = x0 + j * kw
            hgt = (score - lo) / (hi - lo) * sh
            b += fill_rect(x, top + sh - hgt, kw - 0.8, hgt, ACCENT[ramp], 0.85)
            b += fill_rect(x, top + sh + 8 + ph - p * ph, kw - 0.8, p * ph, ACCENT[ramp], 0.85)
    # Each slice's stored results, under its keys.
    sy = top + sh + ph + 22
    stat_rows = [("partial_maxes", [s["max"] for s in slices], 2),
                 ("partial_sums", [s["sum"] for s in slices], 3)]
    combine_rows = [(f"e^(max_s − {fmt(r['overall_max'])})", r["scales"], 3),
                    ("weights[s]", r["weights"], 4)]
    for k, (label, vals, digits) in enumerate(stat_rows):
        ry = sy + k * 18
        b += text(24, ry + 7, label, "tx mono")
        for i, (s, v) in enumerate(zip(slices, vals)):
            x0 = cx + s["begin"] * kw
            b += text(x0 + (s["n"] * kw - 2) / 2, ry + 7, fmt(v, digits), "tx mono", "middle")
    y = sy + 2 * 18 + 10
    b += barrier(y, "second launch")
    y += 18
    b += text(24, y, f"CombineSplitsKernel, block (head {head}, sequence 0): thread s reads slice s", "ts")
    b += text(24, y + 18, f"BlockMax(partial_maxes) = {fmt(r['overall_max'])};  "
                          f"BlockSum(e^(max_s − max) · partial_sums) = {fmt(r['total'], 3)} = total", "tx mono")
    for k, (label, vals, digits) in enumerate(combine_rows):
        ry = y + 32 + k * 20
        b += text(24, ry + 8, label, "tx mono")
        for i, (s, v) in enumerate(zip(slices, vals)):
            x0, w = cx + s["begin"] * kw, s["n"] * kw - 2
            b += rect(x0 + w / 2 - 34, ry, 68, 16, "amber") + label_in(x0 + w / 2 - 34, ry, 68, 16, "amber",
                                                                        fmt(v, digits), "tx", True)
    b += text(24, y + 32 + 2 * 20 + 4, "weights[s] = e^(max_s − max) / total", "tx")

    # out = sum_s weights[s] * partial_s.
    oy = y + 32 + 2 * 20 + 22
    for i, s in enumerate(slices):
        ry = oy + i * 15
        b += text(cx - 6, ry + 6, f"{fmt(r['weights'][i], 4)} × partial {i}", "tx mono", "end")
        b += heat(cx, ry + 1, s["partial"], kw, 11)
    ey = oy + len(slices) * 15 + 6
    b += text(cx - 6, ey + 6, "out", "tx mono", "end") + heat(cx, ey, r["out"], kw, 12)
    b += rect(cx - 2, ey - 2, dim * kw + 4, 16, "amber", extra=' fill-opacity="0"')
    b += text(cx - 6, ey + 24, f"softmax, {context} keys", "tx", "end") + heat(cx, ey + 18, r["exact"], kw, 12)
    diff = max(abs(a - b_) for a, b_ in zip(r["out"], r["exact"]))
    b += text(cx + dim * kw + 8, ey + 15, "equal:", "tx") + text(cx + dim * kw + 8, ey + 29,
                                                                f"max diff {diff:.0e}", "tx")

    f = "engine/kernels/decode_attention.cu"
    foot, bottom = notes(ey + 50, [
        (lines(f, "for (int slice = blockIdx.y; slice < SplitsFor(context); slice += gridDim.y) {"),
         "block y walks slices y, y + gridDim.y, …; each slice is keys [slice · 64, slice · 64 + 64)"),
        (lines(f, "// 2. Softmax, a warp per head: the slice's max and e^(s - max).", "slice_sum[h] = sum;"),
         "the slice's own max and sum of e^(s − max_s)"),
        (lines(f, "// 3. Values: this thread's 8 dimensions over every kKeyLanes-th key.",
               "partial_sums[(head_base + threadIdx.x) * kMaxSplits + slice] = slice_sum[threadIdx.x];"),
         "partial output Σ e^(s − max_s) · v, its max and its sum, stored per slice"),
        (lines(f, "const float max_score = s < splits ? maxes[s] : -FLT_MAX;", "weights[s] = scale / total;"),
         "combine: overall max, rescale each slice, one weight per slice"),
        (lines(f, "for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {\n        float acc = 0.0f;\n"
                  "        for (int t = 0;", "out[head * head_dim + d] = acc;"),
         "thread d sums weights[t] · partial_t[d] over the slices"),
    ])
    write("code_split_attention.svg", bottom + 4, "Split attention and the combine step", b + foot)


# --- FlashAttentionKernel: online softmax ---------------------------------------

def fp16(x):
    """x rounded to FP16 (round to nearest even), as __floats2half2_rn does."""
    return struct.unpack("<e", struct.pack("<e", x))[0]


def replay_flash_row(q, keys, values, position, last_key, scale_log2):
    """FlashAttentionKernel's steps 1-3 for one query row, tile by tile: Q, K,
    V and P in FP16, sums in FP32 (here: float64), scores in the log2 domain."""
    head_dim = len(q)
    q16 = [fp16(x) for x in q]
    running_max, running_sum, o = -math.inf, 0.0, [0.0] * head_dim
    tiles = []
    for key0 in range(0, last_key + 1, TILE):
        rows = [min(key0 + j, last_key) for j in range(TILE)]   # keys past last_key reload it
        k16 = [[fp16(x) for x in keys[row]] for row in rows]
        v16 = [[fp16(x) for x in values[row]] for row in rows]
        s = [sum(a * b for a, b in zip(q16, k16[j])) * scale_log2 if key0 + j <= position else -math.inf
             for j in range(TILE)]
        tile_max = max(s)
        new_max = max(running_max, tile_max)
        base = 0.0 if new_max == -math.inf else new_max
        shrink = 2 ** (running_max - base)
        p = [2 ** (x - base) for x in s]
        tile_sum = sum(p)
        old_sum = running_sum
        running_sum = running_sum * shrink + tile_sum
        p16 = [fp16(x) for x in p]
        o = [o[d] * shrink + sum(p16[j] * v16[j][d] for j in range(TILE)) for d in range(head_dim)]
        tiles.append({"key0": key0, "scores": s, "tile_max": tile_max, "old_max": running_max, "max": new_max,
                      "shrink": shrink, "old_sum": old_sum, "tile_sum": tile_sum, "sum": running_sum, "o": o})
        running_max = new_max
    out = [x / running_sum for x in o]
    return tiles, out


def online_softmax():
    """FlashAttentionKernel's online softmax for one prompt row of Mei's chat:
    per key tile of 64, the masked scores, the running max and sum, the
    rescale factor and the unnormalized output o."""
    data = attention_data()
    case = data["chat"]
    head, dim = case["head"], data["head_dim"]
    position = case["rows"]["flash"]
    row_count = len(case["pieces"])
    first_row = position // FLASH_ROWS * FLASH_ROWS
    last_key = min(first_row + FLASH_ROWS, row_count) - 1    # one sequence: the run's last position
    r_in_tile = position - first_row
    warp, lanes = r_in_tile // 16, (r_in_tile % 16) % 8 * 4
    half = "hi" if r_in_tile % 16 >= 8 else "lo"
    scale_log2 = LOG2_E / math.sqrt(dim)
    tiles, out = replay_flash_row(case["q"]["flash"], case["keys"], case["values"], position, last_key, scale_log2)
    exact = replay_causal(case["q"]["flash"], case["keys"], case["values"], position, 1 / math.sqrt(dim))["out"]

    b = text(24, 22, f"FlashAttentionKernel's online softmax: prompt row {position}, head {head}, "
                     f"key tiles of {TILE}", "th")
    b += text(24, 42, f"Layer {data['layer']} of Mei's chat ({row_count} tokens) as one prompt. Block (head {head}, "
                      f"blockIdx.y {first_row // FLASH_ROWS}) holds rows {first_row}–{last_key}; last_key = {last_key}.",
              "ts")
    b += text(24, 60, f"Row {position} is warp {warp}'s row_{half} for lanes {lanes}–{lanes + 3}: each holds 16 of the "
                      f"tile's 64 scores; two __shfl_xor_sync combine their max and sum.", "tx")
    b += text(24, 76, "Scores are s · scale_log2 (scale_log2 = log2(e) / √128), so exp2 replaces e^x. Q, K, V, P in "
                      "FP16, sums in FP32.", "tx")

    bx, kw, bh = 110, 3.5, 44
    visible = [x for t in tiles for x in t["scores"] if x != -math.inf]
    lo, hi = min(visible), max(visible)
    height = lambda v: (v - lo) / (hi - lo) * (bh - 10) + 4   # 6 px above the highest bar
    y = 100
    for k, t in enumerate(tiles):
        top = y + k * 98
        b += text(24, top + 8, f"key0 = {t['key0']}", "ts mono")
        b += text(24, top + 24, f"keys {t['key0']}–{t['key0'] + TILE - 1}", "tx")
        b += rect(bx - 1, top - 1, TILE * kw + 2, bh + 2, "gray", rx=2, extra=' fill-opacity="0.3"')
        masked = [j for j, s in enumerate(t["scores"]) if s == -math.inf]
        for j, s in enumerate(t["scores"]):
            if s != -math.inf:
                hgt = height(s)
                b += fill_rect(bx + j * kw, top + bh - hgt, kw - 0.8, hgt, ACCENT["teal"], 0.85)
        if masked:
            x0 = bx + masked[0] * kw
            b += fill_rect(x0, top, len(masked) * kw, bh, ACCENT["gray"], 0.25)
            b += text(x0 + len(masked) * kw / 2, top + bh / 2 - 7, "−∞", "ts", "middle")
            b += text(x0 + len(masked) * kw / 2, top + bh / 2 + 8,
                      f"keys {t['key0'] + masked[0]}–{t['key0'] + TILE - 1} > {position}", "tx", "middle")
        changed = t["max"] != t["old_max"]
        my = top + bh - height(t["max"])
        b += line(bx, my, bx + TILE * kw, my, ACCENT["amber"] if changed else ACCENT["gray"], 1.2, dash=True)
        rx = bx + TILE * kw + 16
        old_max = "−∞" if t["old_max"] == -math.inf else fmt(t["old_max"])
        amber = "tx mono s-amber"
        b += text(rx, top + 4, f"tile_{half} = {fmt(t['tile_max'])}", "tx mono")
        b += text(rx, top + 19, f"max_{half}: {old_max} → {fmt(t['max'])}", amber if changed else "tx mono")
        b += text(rx, top + 34, f"shrink_{half} = exp2({old_max} − {fmt(t['max'])}) = {fmt(t['shrink'], 3)}",
                  amber if t["shrink"] != 1 else "tx mono")
        b += text(rx, top + 49, f"sum_{half} = {fmt(t['old_sum'], 3)} · {fmt(t['shrink'], 3)} + "
                                f"{fmt(t['tile_sum'], 3)} = {fmt(t['sum'], 3)}", amber)
        oy = top + bh + 14
        b += text(bx - 8, oy + 6, "o", "tx mono", "end") + heat(bx, oy, t["o"], kw, 12)
        b += text(bx + dim * kw + 8, oy + 6, f"o · {fmt(t['shrink'], 3)} + P·V", "tx mono")
    y += len(tiles) * 98 + 4
    b += text(bx - 8, y + 6, "out", "tx mono", "end") + heat(bx, y, out, kw, 12)
    b += rect(bx - 2, y - 2, dim * kw + 4, 16, "amber", extra=' fill-opacity="0"')
    b += text(bx + dim * kw + 8, y + 6, f"o · (1 / sum_{half})", "tx mono")
    b += text(bx - 8, y + 24, "exact", "tx mono", "end") + heat(bx, y + 18, exact, kw, 12)
    diff = max(abs(a - e) for a, e in zip(out, exact))
    b += text(bx + dim * kw + 8, y + 24, f"max diff {diff:.0e}", "tx")
    b += text(24, y + 46, "exact: CausalAttention in FP32. Bars: one score per key, same scale in every tile; dashed: "
                          "the running max after the tile.", "tx")

    f = "engine/kernels/attention.cu"
    foot, bottom = notes(y + 72, [
        (lines(f, "const std::size_t row = PagedRow(pages, kv.page_shift, min(key0 + j, last_key));"),
         "load the tile's keys and values; keys past last_key reload key last_key"),
        (lines(f, "// 2. Online softmax (log2 domain: exp2 is one instruction).",
               "tile_hi = fmaxf(tile_hi, __shfl_xor_sync(kFullWarpMask, tile_hi, 2));"),
         "mask keys past the row's position to −∞, take the tile max across the 4 lanes"),
        (lines(f, "const float new_lo = fmaxf(max_lo, tile_lo);", "const float shrink_hi = exp2f(max_hi - base_hi);"),
         "new running max; shrink = exp2(old max − new max)"),
        (lines(f, "float tile_sum_lo = 0.0f, tile_sum_hi = 0.0f;", "o[n][3] *= shrink_hi;"),
         "P = exp2(s − max); sum = sum · shrink + tile sum; o *= shrink"),
        (lines(f, "// 3. O += P . V, 16 keys at a time.", "Mma(o[n + 1], p, b[2], b[3]);"),
         "o += P · V on tensor cores, P rounded to FP16"),
        (lines(f, "const float inverse_lo = sum_lo > 0.0f ? 1.0f / sum_lo : 0.0f;",
               "make_float2(o[n][2] * inverse_hi, o[n][3] * inverse_hi);"),
         "after the last tile: divide o by the sum and write the row"),
    ])
    write("code_online_softmax.svg", bottom + 4, "Online softmax step by step", b + foot)


DIAGRAMS = [causal_attention, split_attention, online_softmax]

if __name__ == "__main__":
    for make in DIAGRAMS:
        make()
    print(f"{len(DIAGRAMS)} code pictures written")
