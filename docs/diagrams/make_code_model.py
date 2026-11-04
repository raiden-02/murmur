"""Code pictures for the model on the GPU (code tour Part V): which path
Linear::Apply takes, what Transformer::StageInputs writes for one step and
what ForwardLayer does with it, how Sampler::Pick chooses a token, and what
ChatSession::Reply forwards for one reply. Each picture replays the C++
logic in Python, step for step, on real data:

- data/code_forward_layer.json holds the model's shapes, refreshed from
  models/qwen2.5-1.5b-instruct/config.json when it is present.
- data/schedule.json is the engine trace of `murmur-lab schedule`; the step
  drawn, and the pages each conversation holds at that step, are replayed
  from it. data/code_stage_inputs.json holds that step's token ids (the
  model's tokenizer and greedy replies).
- data/code_sampler_pick.json holds next-token logits of the reference model
  (reference/.venv, Hugging Face) and the draw `murmur-lab generate --sample`
  used for that token.
- data/code_chat_reply.json holds the token counts of `murmur-lab generate
  --sample` and of a next message (the model's tokenizer).

Run from anywhere: python docs/diagrams/make_code_model.py
"""
import json
import math
import os

from code_lines import lines
from svg_kit import ACCENT, arrow, chip, fill_rect, label_in, line, rect, svg, text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CONFIG = os.path.join(HERE, "..", "..", "models", "qwen2.5-1.5b-instruct", "config.json")

# TransformerOptions defaults and constants of src/engine, as the programs above ran them.
MAX_CHUNK, MAX_BATCH, MAX_LOGIT_ROWS, MAX_CONTEXT, PAGE_SIZE = 512, 64, 80, 4096, 16
GEMM_F16_ROWS = 48   # kGemmF16Rows


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def write(name, height, title, body):
    svg(HERE, name, height, title, body)


def shown(token):
    """A token's text with its spaces and newlines made visible."""
    return token.replace(" ", "␣").replace("\n", "↵")


def notes(y, items):
    """Rows of 'lines a–b   what they do', as in the other code pictures."""
    b = ""
    for k, (where, what) in enumerate(items):
        b += text(24, y + k * 17, where, "tx mono") + text(130, y + k * 17, what, "tx")
    return b


# --- Linear::Apply ---------------------------------------------------------

FORMATS = ["BF16", "FP8", "INT4"]
ROW_BANDS = [("1", 1), ("2–48", 2), ("49–512", 49)]   # rows per call, up to max_chunk; one sample of each band
PATHS = {   # function Apply calls: (ramp, what it runs)
    "MatVec": ("gray", "once per row: the fused matrix-vector kernel"),
    "ApplyTensorCores": ("teal", "RmsNormToF16 or ConvertF32ToF16, then GemmF16 with the epilogue fused"),
    "ApplyCublasBf16": ("purple", "RmsNormToBf16 or ConvertF32ToBf16, blas_.LinearBf16, then Finish"),
    "ApplyFp8TensorCores": ("coral", "QuantizeRowsToFp8, blas_.LinearFp8, then FinishFp8Products"),
    "ApplyExact": ("green", "RmsNorm, DequantizeRows (whole matrix to FP32), blas_.Linear (SGEMM), Finish"),
}
OTHER_MATHS = [("exact", "kExactFp32"), ("gemm", "kTensorCores"), ("cublas", "kCublasBf16"),
               ("fp8", "kFp8TensorCores")]   # MultiRowMathName and the enum, besides "fast" (kFastest)


def apply_path(math_name, weight_format, rows, lm_head=False):
    """The function Linear::Apply calls for a model matrix. Every Qwen matrix
    passes GemmF16Supports and the FP8 path's % 16 checks; the only matrix
    bigger than the exact path's scratch is the vocabulary projection."""
    exact = math_name == "exact"
    if rows == 1 or (exact and lm_head):
        return "MatVec"
    if exact:
        return "ApplyExact"
    fastest = math_name == "fast" and rows > GEMM_F16_ROWS
    if weight_format == "BF16" and (math_name == "cublas" or fastest):   # UsesCublasBf16
        return "ApplyCublasBf16"
    if weight_format == "FP8" and (math_name == "fp8" or fastest):
        return "ApplyFp8TensorCores"
    return "ApplyTensorCores"


def linear_apply():
    """Linear::Apply's decision as a grid: rows per call × the weight's
    format, each cell coloured by the function it calls; a large grid for
    the default MultiRowMath, small ones for the other four."""
    b = text(24, 22, "Linear::Apply: the function it calls, by the number of rows and the weight's format", "th")
    b += text(24, 44, "Default math_ = MultiRowMath::kFastest:", "ts")
    gx, gy, cw, ch = 96, 92, 186, 34
    for c, (band, _) in enumerate(ROW_BANDS):
        b += text(gx + c * cw + cw / 2, gy - 22, f"rows = {band}", "ts mono", "middle")
    for r, fmt in enumerate(FORMATS):
        y = gy + r * (ch + 4)
        b += text(gx - 12, y + ch / 2, fmt, "ts mono", "end")
        for c, (_, rows) in enumerate(ROW_BANDS):
            name = apply_path("fast", fmt, rows)
            b += chip(gx + c * cw, y, cw - 4, ch, PATHS[name][0], name, "ts", mono=True)
    split = gx + 2 * cw - 2
    b += line(split, gy - 12, split, gy + 3 * (ch + 4) + 4, ACCENT["amber"], 1.4, dash=True)
    b += text(split, gy + 3 * (ch + 4) + 14, "kGemmF16Rows = 48", "tx mono", "middle")
    b += text(gx - 12, gy - 22, "Format()", "tx mono", "end")

    # The other MultiRowMath values, same axes, colour only.
    sy, scw, sch = 262, 26, 13
    b += text(24, sy - 16, "The other MultiRowMath values (Transformer::SetMultiRowMath), same rows and formats:", "ts")
    for k, (name, enum) in enumerate(OTHER_MATHS):
        x0 = 24 + k * 160
        b += text(x0, sy + 6, f'"{name}"', "ts mono") + text(x0, sy + 20, enum, "tx mono")
        for r, fmt in enumerate(FORMATS):
            for c, (_, rows) in enumerate(ROW_BANDS):
                ramp = PATHS[apply_path(name, fmt, rows)][0]
                b += rect(x0 + c * (scw + 2), sy + 32 + r * (sch + 2), scw, sch, ramp)
        if name == "exact":
            y = sy + 32 + 3 * (sch + 2) + 4
            for c, (_, rows) in enumerate(ROW_BANDS):
                b += rect(x0 + c * (scw + 2), y, scw, sch, PATHS[apply_path(name, "BF16", rows, lm_head=True)][0])
            b += text(x0 + 3 * (scw + 2) + 4, y + sch / 2, "lm_head", "tx mono")
    b += text(24, sy + 104, "Exact math decodes the whole matrix into widened_, sized for the largest layer matrix; "
                           "the 151,936 × 1,536", "tx")
    b += text(24, sy + 119, "vocabulary projection does not fit, so with more than one row it runs MatVec once per row.",
              "tx")

    ly = sy + 146
    for k, (name, (ramp, what)) in enumerate(PATHS.items()):
        y = ly + k * 22
        b += chip(24, y, 150, 18, ramp, name, "tx", mono=True) + text(184, y + 9, what, "tx")

    f = "engine/linear.cpp"
    y = ly + len(PATHS) * 22 + 16
    b += notes(y, [
        (lines(f, "if (rows == 1 || (exact && weight.ElementCount()", "MatVec(weight.View(), x + static_cast"),
         "one row (or a matrix the multi-row path can't take): MatVec for each row"),
        (lines(f, "if (exact) {\n        ApplyExact(", "ApplyTensorCores(x, weight, y, rows, options);\n    }"),
         "then, in order: exact, cuBLAS for BF16, FP8 tensor cores for FP8, else GemmF16"),
        (lines(f, "bool Linear::UsesCublasBf16(const WeightMatrix& weight, int rows) const {",
               "GemmF16Supports(weight.View()) && (math_ == MultiRowMath::kCublasBf16 || fastest);"),
         "UsesCublasBf16: BF16 weights, and math \"cublas\" or \"fast\" with more than 48 rows"),
    ])
    write("code_linear_apply.svg", y + 3 * 17 + 6, "Linear::Apply paths", b)


# --- StageInputs -------------------------------------------------------------

def plan_order(trace, step):
    """The conversations in `step` of the trace, in Engine::Plan's order:
    higher priority first, then decoding before prompts, then arrival."""
    counts = trace["steps"][step]["tokens"]
    return sorted((i for i, n in enumerate(counts) if n > 0), key=lambda i: (-trace["priority"][i], counts[i] != 1, i))


def replay_sequences(trace, step):
    """Each conversation's KvSequence (length and page table) at the start of
    `step` of the engine trace, its token count in that step and its
    `logits` (1 when the step ends its uncached tokens: a decode token, or a
    prompt chunk followed by decoding). Engine::Plan reserves pages in plan
    order, and PagePool hands out free pages 0, 1, 2, ... in turn."""
    names = trace["names"]
    sequences = {name: {"length": 0, "pages": []} for name in names}
    next_page = 0
    for s in range(step + 1):
        counts = trace["steps"][s]["tokens"]
        if s < step and trace["steps"][s]["finished"]:
            raise ValueError("replay_sequences: a reply finished before the step drawn")
        for i in plan_order(trace, s):
            seq = sequences[names[i]]
            while len(seq["pages"]) < -(-(seq["length"] + counts[i]) // PAGE_SIZE):
                seq["pages"].append(next_page)
                next_page += 1
            if s < step:
                seq["length"] += counts[i]
    counts, following = trace["steps"][step]["tokens"], trace["steps"][step + 1]["tokens"]
    return [(names[i], sequences[names[i]], counts[i], int(counts[i] == 1 or following[i] == 1))
            for i in plan_order(trace, step)]


def stage_inputs_replay(entries):
    """Transformer::StageInputs on `entries` (name, sequence, tokens, logits):
    the filled part of every layout_ array, and the StepShape."""
    slots = [(e, 0, 1) for e, (_, _, tokens, _) in enumerate(entries) if len(tokens) == 1]
    decode_rows = len(slots)
    slots += [(e, 0, len(tokens)) for e, (_, _, tokens, _) in enumerate(entries) if len(tokens) > 1]
    shift, mask = PAGE_SIZE.bit_length() - 1, PAGE_SIZE - 1
    rows = {"row_tokens": [], "row_positions": [], "row_cache": [], "row_sequences": []}
    contexts, page_tables, first_row, prefill_context = [], [], {}, 0
    for s, (e, first, count) in enumerate(slots):
        _, seq, tokens, _ = entries[e]
        start = seq["length"] + first
        end = start + count
        page_tables.append(seq["pages"][: -(-end // PAGE_SIZE)])
        contexts.append(end)
        first_row.setdefault(e, len(rows["row_tokens"]))
        for t in range(count):
            position = start + t
            rows["row_tokens"].append(tokens[first + t])
            rows["row_positions"].append(position)
            rows["row_cache"].append((seq["pages"][position >> shift] << shift) | (position & mask))
            rows["row_sequences"].append(s)
        if count > 1:
            prefill_context = max(prefill_context, end)
    logit_rows = []
    for e, (_, _, tokens, logits) in enumerate(entries):
        last_row = first_row[e] + len(tokens) - 1
        logit_rows += [last_row - k for k in range(logits - 1, -1, -1)]
    shape = {"rows": len(rows["row_tokens"]), "decode_rows": decode_rows, "sequences": len(slots),
             "logit_count": len(logit_rows), "contiguous_logits": logit_rows == list(range(len(logit_rows))),
             "prefill_context": prefill_context}
    return rows, contexts, page_tables, logit_rows, shape


def layout_offsets():
    """layout_ as the Transformer constructor sets it (ints into inputs_host_)."""
    slots = MAX_BATCH + MAX_LOGIT_ROWS
    stride = -(-MAX_CONTEXT // PAGE_SIZE)
    o = {"row_tokens": 0, "row_positions": MAX_CHUNK, "row_cache": 2 * MAX_CHUNK, "row_sequences": 3 * MAX_CHUNK,
         "contexts": 4 * MAX_CHUNK}
    o["logit_rows"] = o["contexts"] + slots
    o["page_tables"] = o["logit_rows"] + MAX_LOGIT_ROWS
    o["total"] = o["page_tables"] + slots * stride
    return o, stride


ENTRY_RAMPS = ["teal", "purple", "coral"]


def stage_step():
    """The step drawn: entries in Engine order with their sequences, tokens
    and token texts."""
    data = load("code_stage_inputs.json")
    trace = load("schedule.json")
    entries, texts = [], []
    for name, seq, count, logits in replay_sequences(trace, data["step"]):
        tokens = data["tokens"][name]
        if len(tokens) != count:
            raise ValueError(f"code_stage_inputs.json: {name} has {len(tokens)} tokens, the trace {count}")
        entries.append((name, seq, [t[0] for t in tokens], logits))
        texts.append([t[1] for t in tokens])
    return data["step"], entries, texts


def stage_inputs():
    """StageInputs on a real step: the pinned buffer to scale, one line per
    step row with its four per-row values, the KV-cache rows those values
    point at, and the per-slot and logits arrays."""
    step, entries, texts = stage_step()
    rows, contexts, page_tables, logit_rows, shape = stage_inputs_replay(entries)
    offsets, stride = layout_offsets()
    n = shape["rows"]
    row_arrays = ["row_tokens", "row_positions", "row_cache", "row_sequences"]

    b = text(24, 22, "Transformer::StageInputs: one step's inputs in the pinned buffer inputs_host_", "th")
    b += text(24, 42, f"Engine step {step} of murmur-lab schedule: "
                      + ", ".join(f"{name} {'decodes 1 token' if len(tok) == 1 else f'runs {len(tok)} prompt tokens'}"
                                  for name, _, tok, _ in entries) + ".", "ts")

    # The buffer, its first 2,272 ints to scale; page_tables broken off. Amber: what this step fills.
    bx, by, bh = 24, 74, 14
    unit = 440 / offsets["page_tables"]
    sections = row_arrays + ["contexts", "logit_rows", "page_tables"]
    used = {"contexts": len(contexts), "logit_rows": len(logit_rows)}
    x = bx
    for k, name in enumerate(sections[:-1]):
        w = (offsets[sections[k + 1]] - offsets[name]) * unit
        b += rect(x, by, w - 1, bh, "gray", rx=1) + rect(x, by, max(used.get(name, n) * unit, 2), bh, "amber", rx=1)
        b += text(x, by - 8, str(offsets[name]), "tx mono")
        if name in row_arrays:
            b += text(x + w / 2, by + bh + 10, name, "tx mono", "middle")
        x += w
    b += text(x - (offsets["page_tables"] - offsets["logit_rows"]) * unit - 2, by + bh + 23, "contexts", "tx mono", "end")
    b += text(x - (offsets["page_tables"] - offsets["logit_rows"]) * unit + 2, by + bh + 23, "logit_rows", "tx mono")
    b += text(x + 6, by + bh / 2, "//", "tx", "middle")
    px, pw = x + 12, 656 - x - 12
    b += rect(px, by, pw, bh, "gray", rx=1) + rect(px, by, 2, bh, "amber", rx=1)
    b += text(px, by - 8, str(offsets["page_tables"]), "tx mono")
    b += text(px + pw, by - 8, f"{offsets['total']:,} ints", "tx mono", "end")
    b += text(px + pw / 2, by + bh + 10, f"page_tables: {MAX_BATCH + MAX_LOGIT_ROWS} slots × {stride}", "tx mono",
              "middle")

    # One line per step row.
    ty, rh = 150, 12.5
    cols = {"token": 152, "row": 178, "row_tokens": 222, "row_positions": 300, "row_cache": 378, "row_sequences": 456}
    head = ty - 12
    b += text(24, head, "entry", "tx") + text(cols["token"], head, "token", "tx", "end")
    b += text(cols["row"], head, "row", "tx", "end")
    for name in row_arrays:
        b += text(cols[name], head, name, "tx mono", "middle")
    row = 0
    for e, (name, seq, tokens, _) in enumerate(entries):
        top = ty + row * rh
        b += rect(24, top, 4, len(tokens) * rh - 2, ENTRY_RAMPS[e], rx=1) + text(32, top + 5, name, "tx")
        if len(tokens) > 1:
            b += text(32, top + 18, f"{len(tokens)} rows", "tx")
        for tok_text in texts[e]:
            y = ty + row * rh + 5
            b += text(cols["token"], y, shown(tok_text), "tx mono", "end")
            b += text(cols["row"], y, str(row), "tx mono", "end")
            for col in row_arrays:
                b += text(cols[col] + 18, y, str(rows[col][row]), "ts mono" if col == "row_cache" else "tx mono", "end")
            row += 1
    cut = ty + shape["decode_rows"] * rh - 1
    b += line(24, cut, 470, cut, ACCENT["amber"], 1, dash=True)
    b += text(482, cut, f"decode_rows = {shape['decode_rows']}", "tx mono")

    # The KV-cache pages the step's sequences hold, and the cache rows row_cache points at.
    kx, ky, cell = 520, ty + 52, 8.5
    b += text(kx - 14, ky - 10, "KV cache pages, 16 rows each", "tx")
    written = set(rows["row_cache"])
    for e, (_, seq, _, _) in enumerate(entries):
        for k, page in enumerate(seq["pages"]):
            y = ky + page * 11
            b += text(kx - 6, y + 4.5, str(page), "tx mono", "end")
            for offset in range(PAGE_SIZE):
                x0 = kx + offset * cell
                if (page << 4 | offset) in written:
                    b += rect(x0, y, cell - 1, 9, "amber", rx=1)
                elif k * PAGE_SIZE + offset < seq["length"]:
                    b += rect(x0, y, cell - 1, 9, ENTRY_RAMPS[e], rx=1)
                else:
                    b += rect(x0, y, cell - 1, 9, "gray", rx=1, extra=' fill-opacity="0.2" stroke-opacity="0.4"')
    pages = 1 + max(p for _, seq, _, _ in entries for p in seq["pages"])
    y = ky + pages * 11 + 8
    b += text(kx - 14, y, "colour: cached before this step", "tx") + text(kx - 14, y + 13, "amber: written this step", "tx")

    # Per-slot arrays, then logit_rows.
    sy = y + 40
    b += text(kx - 14, sy, "s", "tx mono") + text(kx + 44, sy, "contexts", "tx mono", "end")
    b += text(kx + 54, sy, "page_tables", "tx mono")
    for s, (ctx, table) in enumerate(zip(contexts, page_tables)):
        y = sy + 16 + s * 16
        b += text(kx - 14, y, str(s), "tx mono") + text(kx + 44, y, str(ctx), "tx mono", "end")
        owner = next(e for e, (_, seq, _, _) in enumerate(entries) if seq["pages"][:len(table)] == table)
        for k, page in enumerate(table):
            b += chip(kx + 54 + k * 22, y - 6, 20, 12, ENTRY_RAMPS[owner], str(page), "tx", mono=True, rx=2)
    y = sy + 16 + len(contexts) * 16 + 8
    b += text(kx - 14, y, "logit_rows", "tx mono") + text(kx + 60, y, "  ".join(map(str, logit_rows)), "tx mono")

    y = ty + n * rh + 16
    b += text(24, y, "row_cache[row] = pages[position >> 4] << 4 | (position & 15): e.g. row 2, position 18: "
                     f"pages[1] = {entries[2][1]['pages'][1]}, so {entries[2][1]['pages'][1]} << 4 | 2 = "
                     f"{rows['row_cache'][2]}.", "tx")
    fields = [f"{field} {str(shape[field]).lower()}" for field in
              ["rows", "decode_rows", "sequences", "logit_count", "contiguous_logits", "prefill_context"]]
    b += text(24, y + 16, "StepShape: " + ", ".join(fields[:4]) + ",", "tx mono")
    b += text(90, y + 30, f"{fields[4]} (logit_rows is not 0, 1, 2), {fields[5]}", "tx mono")
    f = "engine/transformer.cpp"
    y += 54
    b += notes(y, [
        (lines(f, "if (count == 1 || entries[i].decode_each_token) {", "if (static_cast<int>(slots.size()) > MaxSlots()) {"),
         "one slot per decode row first, then one slot per prompt chunk"),
        (lines(f, "const Slot& slot = slots[s];", "host[layout_.row_sequences + row] = static_cast<int>(s);"),
         "per slot: its page table and context; per row: token, position, cache row, slot"),
        (lines(f, "const int last_row = first_row[i] + static_cast<int>(entries[i].tokens.size()) - 1;",
               "++shape.logit_count;"), "logit_rows: the last `logits` rows of each entry, in entry order"),
    ])
    write("code_stage_inputs.svg", y + 3 * 17 + 6, "StageInputs for one step", b)


# --- ForwardLayer ------------------------------------------------------------

def config_data():
    """The model shapes ForwardLayer's buffers are sized by, refreshed from
    the model's config.json when it is present."""
    path = os.path.join(DATA, "code_forward_layer.json")
    if os.path.exists(CONFIG):
        with open(CONFIG, encoding="utf-8") as f:
            config = json.load(f)
        keys = ["hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
                "num_hidden_layers", "vocab_size"]
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({k: config[k] for k in keys}, f, indent=1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def forward_layer():
    """ForwardLayer's six calls on the step above: one band per call, one
    lane per buffer (its columns to scale, one pixel per row); each call's
    reads are grey, its writes amber."""
    config = config_data()
    _, entries, _ = stage_step()
    _, contexts, _, _, shape = stage_inputs_replay(entries)
    rows, decode = shape["rows"], shape["decode_rows"]
    hidden = config["hidden_size"]
    head_dim = hidden // config["num_attention_heads"]
    q_width, kv_width = config["num_attention_heads"] * head_dim, config["num_key_value_heads"] * head_dim
    inter = config["intermediate_size"]

    scale, gap, broken_width = 1 / 24, 12, 120   # px per column; activation_ is drawn broken
    lanes = [("hidden_", hidden), ("qkv_", q_width + 2 * kv_width), ("q_", q_width), ("cache", 2 * kv_width),
             ("attention_", q_width), ("activation_", inter)]
    xs, x = {}, 176
    for name, width in lanes:
        w = broken_width if name == "activation_" else width * scale
        xs[name] = (x, w)
        x += w + gap

    b = text(24, 22, f"Transformer::ForwardLayer on that step: {rows} rows through six calls", "th")
    b += text(24, 42, f"One lane per buffer: columns to scale (1 px = 24 floats), 1 px per row; "
                      f"each is allocated for max_chunk = {MAX_CHUNK} rows.", "ts")
    hy = 82
    for name, width in lanes:
        x0, w = xs[name]
        if name == "cache":
            b += text(x0 + w / 2, hy - 16, f"cache layer: K | V, {kv_width} each", "tx", "middle")
            continue
        b += text(x0, hy, name, "tx mono")
        sizes = {"qkv_": f"{q_width}|{kv_width}|{kv_width}", "activation_": f"× {width:,} (broken)"}
        b += text(x0, hy + 13, sizes.get(name, f"× {width:,}"), "tx mono")

    band, top = rows + 20, hy + 30
    calls = [
        ("1 linear_.Apply", "w.qkv_proj, attn_norm, bias", ["hidden_"], {"qkv_": "w"}),
        ("2 RopeAndCacheQkv", "row_positions, row_cache", ["qkv_"], {"q_": "w", "cache": "w"}),
        ("3 DecodeAttention", f"rows 0–{decode - 1}, contexts", ["q_", "cache"], {"attention_": "decode"}),
        ("4 FlashAttention", f"rows {decode}–{rows - 1}", ["q_", "cache"], {"attention_": "prompt"}),
        ("5 linear_.Apply", "w.o_proj, accumulate", ["attention_"], {"hidden_": "+="}),
        ("6 linear_.FeedForward", "gate_up_proj, SwiGLU", ["hidden_"], {"activation_": "w"}),
        ("", "down_proj, accumulate", ["activation_"], {"hidden_": "+="}),
    ]
    for k, (name, detail, reads, writes) in enumerate(calls):
        y = top + k * band
        b += text(24, y + rows / 2 - 6, name, "ts mono") + text(38, y + rows / 2 + 8, detail, "tx")
        for lane, (x0, w) in xs.items():
            b += rect(x0, y, w, rows, "gray", rx=1, extra=' fill-opacity="0.15" stroke-opacity="0.3"')
        for lane in reads:
            x0, w = xs[lane]
            b += rect(x0, y, w, rows, "gray", rx=1)
        for lane, how in writes.items():
            x0, w = xs[lane]
            first, count = {"decode": (0, decode), "prompt": (decode, rows - decode)}.get(how, (0, rows))
            b += rect(x0, y + first, w, count, "amber", rx=1)
            if how == "+=":
                b += label_in(x0, y, w, rows, "amber", "+=", "tx", mono=True)
        # Splits inside a lane: q | k | v, K | V, and the break in activation_.
        x0, w = xs["qkv_"]
        for split in (q_width, q_width + kv_width):
            b += line(x0 + split * scale, y, x0 + split * scale, y + rows, None, 0.6)
        x0, w = xs["cache"]
        b += line(x0 + w / 2, y, x0 + w / 2, y + rows, None, 0.6)
        x0, w = xs["activation_"]
        b += f'<rect class="bg" x="{x0 + w / 2 - 5:.1f}" y="{y - 1}" width="10" height="{rows + 2}"/>'
        b += text(x0 + w / 2, y + rows / 2, "//", "tx", "middle")
        src = xs[reads[-1]]
        dst = xs[next(iter(writes))]
        if dst[0] > src[0]:
            b += arrow(src[0] + src[1] + 2, y + rows / 2, dst[0] - 3, y + rows / 2, ACCENT["gray"], 1.2)
        else:
            b += arrow(src[0] - 2, y + rows / 2, dst[0] + dst[1] + 3, y + rows / 2, ACCENT["gray"], 1.2)

    y = top + len(calls) * band + 2
    caption = [
        "Grey: read. Amber: written. +=: the product is added onto hidden_, the residual stream.",
        "RopeAndCacheQkv writes k and v at each row's row_cache row; attention reads the cache up to each slot's",
        f"context ({', '.join(map(str, contexts))}). {q_width // head_dim} query heads and {kv_width // head_dim} "
        f"key/value heads of {head_dim} columns each.",
        f"With {rows} rows (≤ 48) every linear_.Apply here runs ApplyTensorCores; SwiGLU runs in GemmF16's epilogue.",
    ]
    for k, caption_line in enumerate(caption):
        b += text(24, y + k * 14, caption_line, "tx")
    f = "engine/transformer.cpp"
    y += len(caption) * 14 + 12
    b += notes(y, [
        (lines(f, "linear_.Apply(hidden_.Data(), w.qkv_proj, qkv_.Data(), rows,", "q_.Data(), cache, stream);"),
         "1–2: q, k, v from the normalized residual; rotate, then cache k and v"),
        (lines(f, "DecodeAttention(q_.Data(), cache, kv, DeviceInputs(layout_.contexts),", "heads, stream);\n        }\n    }"),
         "3–4: decode rows, then prompt rows from offset (CausalAttention in exact mode)"),
        (lines(f, "linear_.Apply(attention_.Data(), w.o_proj, hidden_.Data(), rows, {.accumulate = true});",
               "linear_.FeedForward(hidden_.Data(), w.gate_up_proj, w.down_proj, activation_.Data(), rows,"),
         "5–6: each adds its product onto hidden_"),
    ])
    write("code_forward_layer.svg", y + 3 * 17 + 6, "ForwardLayer data flow", b)


# --- Sampler::Pick -----------------------------------------------------------

TEMPERATURE, TOP_K, TOP_P, PENALTY = 0.7, 20, 0.8, 1.1   # SamplingParams defaults


def replay_pick(logits, unit_draw):
    """Sampler::Pick on (id, text, logit, in_history) candidates. Returns
    the TopK candidates with every stage's value, keep, and the draw walk."""
    staged = []
    for token_id, token_text, logit, in_history in logits:
        penalized = (logit * PENALTY if logit < 0 else logit / PENALTY) if in_history else logit
        staged.append({"id": token_id, "text": token_text, "raw": logit, "penalized": penalized,
                       "in_history": in_history, "scaled": penalized / TEMPERATURE})
    candidates = sorted(staged, key=lambda c: (-c["scaled"], c["id"]))[:TOP_K]
    if min(c["raw"] for c in staged) >= candidates[-1]["penalized"]:
        raise ValueError("replay_pick: too few logits stored to know the top k")
    dropped_by_top_k = [c for c in sorted(staged, key=lambda c: -c["raw"])[:TOP_K] if c not in candidates]
    max_logit = candidates[0]["scaled"]
    weights = [math.exp(c["scaled"] - max_logit) for c in candidates]
    for c, w in zip(candidates, weights):
        c["p"] = w / sum(weights)
    keep, dropped = len(candidates), 0.0
    while keep > 1 and dropped + candidates[keep - 1]["p"] <= 1.0 - TOP_P:
        dropped += candidates[keep - 1]["p"]
        keep -= 1
    kept_mass = sum(c["p"] for c in candidates[:keep])
    draw, walk, chosen = unit_draw * kept_mass, [], keep - 1
    for i in range(keep - 1):
        walk.append((draw, draw - candidates[i]["p"]))
        draw -= candidates[i]["p"]
        if draw < 0:
            chosen = i
            break
    return candidates, dropped_by_top_k, keep, kept_mass, walk, chosen


def sampler_pick():
    """Sampler::Pick on a real next-token distribution: one line per TopK
    candidate with its logit after each step, the probabilities, the cut
    NucleusSize makes, and the draw walking the kept candidates."""
    data = load("code_sampler_pick.json")
    unit_draw = data["unit_draw"]
    candidates, dropped_by_top_k, keep, kept_mass, walk, chosen = replay_pick(data["logits"], unit_draw)
    if candidates[chosen]["text"] != data["chosen_by_engine"]:
        raise ValueError("sampler_pick: the replay picks a different token than the engine did")

    so_far = "".join(data["reply_so_far"])
    b = text(24, 22, f'Sampler::Pick for the token after "{so_far}" (Mei\'s reply), unit_draw = {unit_draw:.3f}', "th")
    b += text(24, 42, f"history = context_, {data['history_length']} tokens. temperature {TEMPERATURE}, top_k {TOP_K}, "
                      f"top_p {TOP_P}, repetition_penalty {PENALTY}.", "ts")

    ty, rh = 100, 15
    cx = {"i": 34, "token": 104, "raw": 176, "pen": 246, "temp": 316, "p": 346, "walk": 506}
    for key, step, label in [("raw", "", "logits[id]"), ("pen", "1 penalty", "÷ 1.1"), ("temp", "2 temperature", "÷ 0.7")]:
        b += text(cx[key], ty - 14, label, "tx mono", "end")
        b += text(cx[key] - 18, ty - 30, step, "tx", "middle")
    b += text(cx["i"], ty - 14, "i", "tx mono", "end") + text(cx["token"], ty - 14, "token", "tx", "end")
    b += text(cx["p"], ty - 30, "3 TopK, 4 Probabilities", "tx") + text(cx["p"], ty - 14, "probabilities[i]", "tx mono")
    b += text(cx["walk"], ty - 30, "6 the draw", "tx") + text(cx["walk"], ty - 14, "draw −= probabilities[i]", "tx mono")
    bar_max = 108 / candidates[0]["p"]
    for i, c in enumerate(candidates):
        y = ty + i * rh
        kept = i < keep
        b += text(cx["i"], y + 6, str(i), "tx mono", "end")
        b += text(cx["token"], y + 6, shown(c["text"]), "ts mono" if kept else "tx mono", "end")
        if c["in_history"]:
            b += rect(cx["token"] + 5, y + 1, 10, 11, "coral", rx=2)
        b += text(cx["raw"], y + 6, f"{c['raw']:.2f}", "tx mono", "end")
        if c["penalized"] != c["raw"]:
            b += rect(cx["pen"] - 36, y, 40, 13, "amber", rx=2)
            b += text(cx["pen"], y + 6, f"{c['penalized']:.2f}", "tx mono k-amber", "end")
        else:
            b += text(cx["pen"], y + 6, f"{c['penalized']:.2f}", "tx mono", "end")
        b += text(cx["temp"], y + 6, f"{c['scaled']:.2f}", "tx mono", "end")
        w = c["p"] * bar_max
        color = ACCENT["amber"] if i == chosen else ACCENT["teal"] if kept else ACCENT["gray"]
        b += fill_rect(cx["p"], y + 1, w, 11, color, 0.9 if kept else 0.45, rx=1)
        b += text(cx["p"] + w + 4, y + 6, f"{c['p']:.3f}", "tx mono")
        if i < len(walk):
            before, after = walk[i]
            result = f"{before:.3f} − {c['p']:.3f} " + (f"= {after:.3f}" if after >= 0 else "< 0")
            b += text(cx["walk"], y + 6, result, "ts mono" if i == chosen else "tx mono")
    b += text(cx["walk"], ty + 6 + len(walk) * rh, f"return candidates[{chosen}].id", "tx mono")
    cut = ty + keep * rh - 2
    b += line(24, cut, 656, cut, ACCENT["amber"], 1, dash=True)
    for k, nucleus_line in enumerate(["5 NucleusSize: keep = " + str(keep),
                                      f"the {len(candidates) - keep} below hold {1 - kept_mass:.3f},",
                                      f"within 1 − top_p = {1 - TOP_P:.1f}"]):
        b += text(cx["walk"], cut + 10 + k * 14, nucleus_line, "tx")

    # The draw, measured along the kept candidates laid end to end.
    sy = ty + len(candidates) * rh + 20
    sx, sw = 24, 632
    b += text(24, sy, f"draw = unit_draw × kept_mass = {unit_draw:.3f} × {kept_mass:.3f} = {unit_draw * kept_mass:.3f}; "
                      "the kept candidates end to end, each as wide as its probability:", "ts")
    x = sx
    for i, c in enumerate(candidates[:keep]):
        w = c["p"] / kept_mass * sw
        ramp = "amber" if i == chosen else "teal"
        b += rect(x, sy + 12, w - 2, 22, ramp)
        if w > 7 * len(c["text"]) + 8:
            b += label_in(x, sy + 12, w - 2, 22, ramp, shown(c["text"]), "tx", True)
        x += w
    px = sx + unit_draw * sw
    b += line(px, sy + 8, px, sy + 40, ACCENT["coral"], 2)
    b += text(px, sy + 50, f"unit_draw {unit_draw:.3f}", "tx mono", "middle")

    raw_top = max(candidates, key=lambda c: c["raw"])
    caption = [f"Coral square: the token is in history, so its logit is divided by {PENALTY} (a negative one would be "
               f"multiplied). \"{shown(raw_top['text'])}\" had the highest logit;",
               f"after the penalty \"{shown(candidates[0]['text'])}\" is first."
               + (" Penalized out of the top 20: " + ", ".join(f'"{shown(c["text"])}"' for c in dropped_by_top_k) + "."
                  if dropped_by_top_k else "")]
    y = sy + 70
    for k, caption_line in enumerate(caption):
        b += text(24, y + k * 14, caption_line, "tx")
    f = "engine/sampler.cpp"
    y += len(caption) * 14 + 12
    b += notes(y, [
        (lines(f, "std::vector<bool> seen(logits.size(), false);", "logits[id] = logits[id] < 0.0f ? "
               "logits[id] * penalty : logits[id] / penalty;"), "1: each token in history once, over all 151,665 logits"),
        (lines(f, "for (float& logit : logits) {", "const std::size_t keep = NucleusSize(probabilities, params_.top_p);"),
         "2–5: temperature, the 20 highest sorted, softmax over them, the nucleus"),
        (lines(f, "const double kept_mass = std::accumulate(", "return candidates[keep - 1].id;"),
         "6: subtract each kept probability from the draw until it goes below zero"),
    ])
    write("code_sampler_pick.svg", y + 3 * 17 + 6, "Sampler::Pick step by step", b)


# --- ChatSession::Reply ------------------------------------------------------

def chat_reply():
    """One reply and the next message on a token axis, one line per
    Transformer::Forward call in time order: the prompt in one call, one
    call per generated token (a staircase), the stop token that is never
    forwarded, and the closing tokens kept in pending_ for the next message."""
    data = load("code_chat_reply.json")
    system, turn, gen = data["system_tokens"], data["first_turn_tokens"], data["generation_prompt_tokens"]
    reply = data["reply"]
    prompt = system + turn + gen
    after = len(data["after_reply"])
    next_prompt = after + data["next_turn_tokens"] + gen
    replied = prompt + len(reply)
    total = replied + next_prompt

    x0, x1, mx = 160, 508, 520   # the token axis, and the margin for labels
    unit = (x1 - x0) / total

    def px(position):
        return x0 + position * unit

    b = text(24, 22, f'ChatSession::Reply("{data["first_message"]}"), then the next message', "th")
    b += text(24, 42, "One line per Transformer::Forward call, in order; x is the position in context_ and in the KV "
                      "cache.", "ts")
    ay = 66
    for position in range(0, total + 1, 20):
        b += line(px(position), ay, px(position), ay + 4) + text(px(position), ay - 8, str(position), "tx mono", "middle")
    b += line(x0, ay + 4, x1, ay + 4)

    # Reply's first Forward: pending_ (the system message) + the user turn + the generation prompt.
    y, h = 80, 16
    b += text(24, y + h / 2, "Forward(prompt, 0)", "tx mono")
    b += chip(px(0), y, system * unit - 1, h, "purple", f"pending_: the system message, {system}", "tx")
    b += chip(px(system), y, (turn + gen) * unit - 1, h, "teal")
    b += text(px(prompt) + 4, y + h / 2, str(prompt), "tx mono")
    b += text(mx, y + 2, f"user turn {turn} +", "tx") + text(mx, y + 15, f"generation prompt {gen}", "tx")
    b += text(px(0), y + h + 12, f"{prompt} rows ≤ max_chunk {MAX_CHUNK}: one RunStep", "tx")

    # Then one Forward per generated token.
    sy, step_h = y + h + 28, 6
    for n in range(len(reply)):
        yy = sy + n * step_h
        b += fill_rect(px(0), yy + 1, (prompt + n) * unit - 1, step_h - 2, ACCENT["gray"], 0.25)
        b += fill_rect(px(prompt + n), yy, max(unit - 0.5, 1.5), step_h - 1, ACCENT["amber"], 0.95)
    b += text(24, sy + 6, "NextToken, then", "tx") + text(24, sy + 20, "Forward(&next, 1),", "tx mono")
    b += text(24, sy + 34, f"{len(reply)} times", "tx")
    b += text(px(0) + 6, sy + len(reply) * step_h / 2, "grey: already in context_, read from the cache", "tx")
    mid = sy + len(reply) * step_h / 2
    b += text(mx, mid - 7, f"{len(reply)} reply tokens,", "tx") + text(mx, mid + 7, "one Forward each", "tx")

    # The stop token: picked, never forwarded.
    yy = sy + len(reply) * step_h + 4
    b += rect(px(replied), yy, max(unit, 3), 10, "coral", rx=1, extra=' fill-opacity="0.15" stroke-dasharray="2 1"')
    b += text(px(replied) - 6, yy + 5, f"NextToken = {data['stop']}: break", "tx mono", "end")
    b += text(mx, yy + 5, "stop token: not forwarded", "tx")

    # pending_ after the reply: the closing tokens, kept for the next message.
    py = yy + 24
    b += text(24, py + 7, "pending_ =", "tx mono") + text(24, py + 21, f"context_.size() = {replied}", "tx mono")
    b += chip(px(replied), py, after * unit, 14, "coral")
    b += text(px(replied) - 6, py + 7, "<|im_end|> ↵", "tx mono", "end")
    b += text(mx, py + 7, "the turn's closing tokens,", "tx") + text(mx, py + 21, "kept for the next message", "tx")

    # The next message's Forward.
    ny = py + 40
    b += text(24, ny + h / 2, f"Forward(prompt, {replied})", "tx mono")
    b += fill_rect(px(0), ny + 3, replied * unit - 1, h - 6, ACCENT["gray"], 0.25)
    b += chip(px(replied), ny, after * unit, h, "coral")
    b += chip(px(replied + after), ny, (next_prompt - after) * unit - 1, h, "teal")
    b += text(mx, ny + 2, f"{after} + user turn {data['next_turn_tokens']}", "tx")
    b += text(mx, ny + 15, f"+ {gen} = {next_prompt} rows", "tx")
    b += text(px(0), ny + h + 14, f'"{data["next_message"]}"; its reply then starts the same way, at position {total}.',
              "tx")

    reply_text = "".join(reply)
    y = ny + h + 38
    b += text(24, y, f'Reply: "{reply_text}"', "tx")
    f, t = "engine/chat_session.cpp", "engine/transformer.cpp"
    y += 22
    b += notes(y, [
        (lines(f, "std::vector<TokenId> prompt = std::exchange(pending_, {});", "Append(context_, prompt);"),
         "pending_ + the user turn, forwarded at position context_.size()"),
        (lines(t, "for (std::size_t begin = 0; begin < tokens.size(); begin += chunk) {",
               "solo_.length += static_cast<int>(part.size());"), "Forward: one RunStep per max_chunk tokens"),
        (lines(f, "for (int n = 0; n < max_reply_tokens_ && !stop.stop_requested(); ++n) {", "context_.push_back(next);"),
         "pick a token; stop at the end of turn; else stream it, forward it, append it"),
        (lines(f, "pending_.push_back(template_.EndOfTurn());", "Append(pending_, template_.EncodeAfterReply());"),
         "the closing tokens wait in pending_ for the next message"),
    ])
    write("code_chat_reply.svg", y + 4 * 17 + 6, "ChatSession::Reply timeline", b)


DIAGRAMS = [linear_apply, stage_inputs, forward_layer, sampler_pick, chat_reply]

if __name__ == "__main__":
    for make in DIAGRAMS:
        make()
    print(f"{len(DIAGRAMS)} code pictures written")
