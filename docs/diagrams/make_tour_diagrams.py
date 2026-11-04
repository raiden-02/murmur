"""Diagrams for docs/CODE_TOUR.md: the running system, the source files in reading order, the
startup load path, one message end to end, and one reply's action. The
action figure reads data/actions.json (`murmur-lab actions --json`) and
data/action_logits.json (extract_action_data.py).

Run from anywhere: python docs/diagrams/make_tour_diagrams.py
"""
import json
import os

from svg_kit import ACCENT, WIDTH, arrow, chip, fill_rect, label_in, line, path, rect, svg, text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GRAMMAR_GRAY = "#C9C7BE"
MONO_W = 6.6   # width of one monospace character at 12px


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def write(name, height, title, body):
    svg(HERE, name, height, title, body)


def char_width(s, size=12):
    """Rough rendered width of a string at the given font size."""
    return sum(0.62 if c.isalnum() else 0.45 for c in s) * size


def mono(x, y, s, cls="ts", anchor="start"):
    return text(x, y, s, cls + " mono", anchor)


def visible(token):
    return token.replace(" ", "␣").replace("\n", "↵")


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


# The tour's parts, bottom to top, and the files each one reads, with whether
# that file belongs to a core (starred) chapter. A file read in several
# chapters is listed once, under its main one. Generated tables are left out.
READING_ORDER = [
    ("I", "files", [("model/mapped_file", False), ("model/safetensors", False), ("model/model_config", False),
                    ("model/unicode", False), ("model/pre_tokenizer", False), ("model/tokenizer", True),
                    ("model/chat_template", False)]),
    ("II", "pages", [("engine/page_pool", False), ("engine/prefix_cache", False),
                     ("engine/kv_cache_manager", True)]),
    ("III", "GPU", [("common/cuda_check", False), ("common/device_buffer", False), ("common/pinned_buffer", False),
                    ("common/cuda_stream", False), ("common/cuda_graph", False)]),
    ("IV", "kernels", [("engine/kernels/block_reduce", False), ("engine/kernels/rms_norm", False),
                       ("engine/kernels/elementwise", False), ("engine/kernels/matvec", True),
                       ("engine/kernels/rope", False), ("engine/kernels/attention", True),
                       ("engine/kernels/gemm", False), ("engine/kernels/sampling", False)]),
    ("V", "model", [("engine/weight_matrix", False), ("engine/linear", True), ("engine/model_weights", False),
                    ("engine/kv_cache", False), ("engine/transformer", True), ("engine/sampler", False),
                    ("engine/gpu_sampler", False), ("engine/chat_session", True)]),
    ("VI", "faster", [("engine/kernels/decode_attention", True)]),
    ("VII", "serving", [("engine/engine", True), ("engine/inference_service", True)]),
    ("VIII", "formats", [("engine/kernels/quantize", True), ("engine/channel_importance", False),
                         ("lab/quality", False)]),
    ("IX", "acting", [("engine/grammar", False), ("engine/speculator", True)]),
    ("X", "game", [("game/action_language", False), ("game/street", False), ("game/post_fx", False),
                   ("game/market", False), ("game/dialogue", False), ("game/hud", False), ("game/main", False)]),
]
SRC = os.path.join(HERE, "..", "..", "src")
# Shorter labels, so the narrow core blocks can carry one.
SHORT = {"engine/inference_service": "service", "engine/kernels/decode_attention": "split attn",
         "engine/kv_cache_manager": "kv pages", "engine/chat_session": "chat", "engine/speculator": "draft"}


def source_lines(stem):
    """Lines in a file's header and source together (.hpp/.cuh and .cpp/.cu)."""
    total = 0
    for ext in (".hpp", ".cuh", ".cpp", ".cu"):
        path = os.path.join(SRC, stem + ext)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                total += sum(1 for _ in f)
    if total == 0:
        raise FileNotFoundError(stem)
    return total


# The game's model memory (kModelMemoryBudget in src/game/main.cpp), the FP8
# weights' size (`murmur-lab quality`) and one BF16 KV page (16 tokens x 28
# layers x 2 x 256 values x 2 bytes).
MEMORY_BUDGET = 3584 << 20
FP8_WEIGHT_BYTES = 1.55e9
KV_PAGE_BYTES = 16 * 28 * 2 * 256 * 2


def tour_system():
    """The running system: one column per place code runs (the game thread,
    the worker thread, the GPU), each part in the column where it runs and
    tagged with its tour chapters, the four things that cross between columns
    every step, and the GPU's model memory to scale. The files on disk feed
    it once, at startup."""
    b = text(24, 22, "The whole system: where each part runs, and what crosses between them", "th")
    top, bottom = 50, 404
    game, worker, gpu = (16, 136), (246, 186), (528, 136)   # (x, width) of each column
    for (x, w), title, sub, ramp in ((game, "game thread", "CPU, 60 frames a second", "purple"),
                                     (worker, "worker thread", "CPU, one engine step after another", "teal"),
                                     (gpu, "GPU", "one stream, in order", "coral")):
        b += fill_rect(x, top, w, bottom - top, ACCENT[ramp], 0.08, rx=8)
        b += text(x + 10, top + 16, title, "ts") + text(x + 10, top + 31, sub, "tx")

    def part(x, y, w, ramp, name, chapters, h=26):
        return (rect(x, y, w, h, ramp, rx=4) + text(x + 8, y + h / 2, name, "tx")
                + text(x + w - 6, y + h / 2, chapters, "tx", "end"))

    gx, gw = game[0] + 8, game[1] - 16
    for k, (name, chapters) in enumerate((("frame loop", "51"), ("Market, 32 NPCs", "49"),
                                         ("Dialogue, X-ray", "50"), ("action language", "47"))):
        b += part(gx, 98 + k * 34, gw, "purple", name, chapters)

    wx, ww = worker[0] + 8, worker[1] - 16
    b += part(wx, 98, ww, "teal", "InferenceService::Run", "36")
    b += rect(wx, 134, ww, 214, "gray", rx=6)
    b += text(wx + 8, 148, "Engine::Step", "ts") + text(wx + ww - 6, 148, "32", "tx", "end")
    for k, (name, chapters) in enumerate((("conversations", "31"), ("Plan, MakeRoom", "33–34"),
                                         ("KV pages, prefix cache", "7–9"), ("tokens out, actions", "35, 45"),
                                         ("draft model guesses", "46"))):
        b += part(wx + 8, 162 + k * 36, ww - 16, "teal", name, chapters)
    b += part(wx, 358, ww, "teal", "tokenizer, chat format", "5–6")

    px, pw = gpu[0] + 8, gpu[1] - 16
    for k, (name, chapters) in enumerate((("Transformer", "24–25"), ("28 layers × 6", "13–18"),
                                         ("attention", "28, 30"), ("sampling", "19, 26"))):
        b += part(px, 98 + k * 34, pw, "coral", name, chapters)
    # GPU memory, to scale: the FP8 weights, then the KV cache gets the rest.
    kv = MEMORY_BUDGET - FP8_WEIGHT_BYTES
    my, mh, bar = 262, 128, 22
    wh = mh * FP8_WEIGHT_BYTES / MEMORY_BUDGET
    b += text(px, my - 10, f"memory, {MEMORY_BUDGET / 1e9:.2f} GB", "tx")
    b += rect(px, my, bar, wh, "coral", rx=2) + rect(px, my + wh, bar, mh - wh, "amber", rx=2)
    for y, lines in ((my + wh / 2, (f"weights {FP8_WEIGHT_BYTES / 1e9:.2f} GB", "FP8 (22, 37)")),
                     (my + wh + (mh - wh) / 2, (f"KV cache {kv / 1e9:.2f} GB", f"{kv / KV_PAGE_BYTES:,.0f} pages (23)"))):
        b += text(px + bar + 8, y - 7, lines[0], "tx") + text(px + bar + 8, y + 7, lines[1], "tx")

    # What crosses between the columns every step, in order; labels sit in the gaps.
    g = ACCENT["gray"]
    left_gap = (game[0] + game[1] + worker[0]) / 2
    right_gap = (worker[0] + worker[1] + gpu[0]) / 2

    def flow(x1, y1, x2, y2):
        mid = (x1 + x2) / 2
        return path(f"M{x1} {y1} C {mid} {y1}, {mid} {y2}, {x2} {y2}", color=g, width=1.2)

    b += flow(gx + gw, 145, wx, 111) + text(left_gap, 84, "1 Ask(text)", "tx", "middle")
    b += flow(wx + ww, 190, px, 111) + text(right_gap, 84, "2 tokens,", "tx", "middle")
    b += text(right_gap, 98, "page tables", "tx", "middle")
    b += flow(px, 213, wx + ww, 250) + text(right_gap, 268, "3 token ids,", "tx", "middle")
    b += text(right_gap, 282, "4 B each", "tx", "middle")
    b += flow(wx, 280, gx + gw, 179) + text(left_gap, 250, "4 text,", "tx", "middle")
    b += text(left_gap, 264, "actions", "tx", "middle")
    b += text(left_gap, 372, "a queue each", "tx", "middle") + text(left_gap, 386, "way, mutex_ (36)", "tx", "middle")
    b += text(right_gap, 372, "pinned", "tx", "middle") + text(right_gap, 386, "buffer (24)", "tx", "middle")

    # Startup: the model files, read once.
    dy = 420
    b += rect(worker[0], dy, gpu[0] + gpu[1] - worker[0], 26, "gray", rx=4)
    b += text(worker[0] + 8, dy + 13, "disk: config.json, tokenizer.json, model.safetensors", "tx mono")
    b += text(gpu[0] + gpu[1] - 6, dy + 13, "1–3, 22", "tx", "end")
    for x in (wx + ww / 2, px + pw / 2):
        b += line(x, dy, x, bottom + 2, g, 1.0, dash=True, arrow_end=True)
    b += text(24, dy + 13, "read once, at startup", "tx")
    b += text(24, dy + 50, "Each part sits in the column where its code runs; the numbers are the tour's chapters. Every step,", "ts")
    b += text(24, dy + 68, "1 the game queues messages, 2 the worker sends the GPU one batch, 3 gets one token id per", "ts")
    b += text(24, dy + 86, "conversation back, and 4 hands the game text and actions. Chapter 52 follows one message through it.", "ts")
    write("tour_system.svg", dy + 104, "The whole system", b)


def tour_reading_order():
    """The source files in the tour's order, bottom to top: each row a part,
    each block a file as wide as its lines of code. Coloured blocks are the
    core chapters' files; following them upward is the short path."""
    rows = [(numeral, name, [(stem, core, source_lines(stem)) for stem, core in files])
            for numeral, name, files in READING_ORDER]
    X0, X1, H, GAP, PAD = 112, 652, 22, 6, 2
    widest = max(sum(n for _, _, n in files) + PAD * len(files) for _, _, files in rows)
    scale = (X1 - X0) / widest
    top = 40
    b = text(40, 22, "Read from the bottom up: each part uses only the parts below it", "th")
    for i, (numeral, name, files) in enumerate(reversed(rows)):
        y = top + i * (H + GAP)
        b += text(X0 - 10, y + H / 2 + 4, f"{numeral} {name}", "tx", "end")
        x = X0
        for stem, core, n in files:
            w = n * scale
            ramp = "teal" if core else "gray"
            b += rect(x, y, w, H, ramp, rx=3)
            label = SHORT.get(stem, stem.split("/")[-1])
            if char_width(label, 10) + 6 < w:
                b += label_in(x, y, w, H, ramp, label, cls="tx", mono=True)
            x += w + PAD
    bottom = top + len(rows) * (H + GAP) - GAP
    b += arrow(26, bottom, 26, top, ACCENT["gray"], 1.4)
    legend = bottom + 26
    b += rect(X0, legend - 10, 14, 12, "teal", rx=2) + text(X0 + 20, legend, "a core (★) chapter's file", "ts")
    b += rect(X0 + 190, legend - 10, 14, 12, "gray", rx=2) + text(X0 + 210, legend, "depth", "ts")
    b += text(X1, legend, "width: lines of code", "ts", "end")
    write("tour_reading_order.svg", legend + 18, "Source files in reading order", b)


def tour_load_path():
    """Startup: where each file's contents end up, by location."""
    b = text(40, 22, "Startup: three files on disk become an engine on the GPU", "th")
    lanes = [("disk", 50), ("CPU memory", 130), ("GPU memory", 210)]
    for name, y in lanes:
        b += line(130, y + 30, 650, y + 30, None, 0.5, dash=True)
        b += text(120, y + 14, name, "ts", "end")
    files = [("config.json", 150), ("tokenizer.json", 300), ("model.safetensors", 470)]
    for name, x in files:
        b += chip(x, 50, 130, 26, "gray", name, mono=True)
    b += chip(150, 130, 130, 26, "amber", "ModelConfig", mono=True)
    b += chip(300, 130, 130, 26, "amber", "Tokenizer", mono=True)
    b += chip(470, 130, 130, 26, "amber", "memory map", mono=True)
    for x in (215, 365, 535):
        b += arrow(x, 78, x, 128, ACCENT["gray"])
    b += text(222, 104, "FromFile", "tx") + text(372, 104, "FromFile", "tx") + text(542, 104, "MappedFile", "tx")
    b += chip(390, 210, 100, 26, "teal", "weights 3.09 GB", cls="tx")
    b += chip(500, 210, 70, 26, "coral", "KV cache", cls="tx")
    b += chip(580, 210, 66, 26, "purple", "buffers", cls="tx")
    b += arrow(500, 158, 450, 208, ACCENT["teal"])
    b += text(470, 186, "Upload (stacked, BF16)", "tx", "end")
    b += text(390, 260, "Transformer allocates the cache and every buffer once", "tx")
    b += text(40, 290, "The weights are copied straight out of the mapped file onto the GPU (no buffer of our own in between), and", "ts")
    b += text(40, 308, "the map is closed once they're there. The config and tokenizer stay on the CPU; only numbers go to the GPU.", "ts")
    write("tour_load_path.svg", 326, "Startup load path", b)


def tour_request_path():
    """One message across the two threads and the GPU, in time order."""
    b = text(40, 22, "One message, end to end: who runs what, in time order", "th")
    lanes = [("game thread", 56, "purple"), ("worker thread", 136, "teal"), ("GPU", 216, "coral")]
    for name, y, _ in lanes:
        b += line(120, y + 13, 660, y + 13, None, 0.5, dash=True)
        b += text(110, y + 13, name, "ts", "end")

    def step(x, lane, label, w=None):
        y = next(yy for name, yy, _ in lanes if name == lane)
        ramp = next(r for name, _, r in lanes if name == lane)
        w = w or char_width(label, 11) + 14
        return chip(x, y, w, 26, ramp, label, cls="tx", mono=True), x + w

    s, _ = step(124, "game thread", "Say")
    b += s
    b += arrow(150, 84, 160, 134, ACCENT["gray"], 1.0)
    s, x = step(160, "worker thread", "Encode")
    b += s
    s, x2 = step(216, "GPU", "prefill", 64)
    b += s
    b += arrow(200, 164, 222, 214, ACCENT["gray"], 1.0)
    # The decode loop: GPU step, then the engine picks and decodes the token.
    x = 290
    for k in range(5):
        s, xe = step(x, "GPU", "step", 34)
        b += s
        b += arrow(x + 17, 214, x + 30, 164, ACCENT["gray"], 0.8)
        s, xt = step(x + 22, "worker thread", "pick", 34)
        b += s
        b += arrow(x + 40, 134, x + 60, 84, ACCENT["gray"], 0.8)
        s, _ = step(x + 46, "game thread", "draw", 36)
        b += s
        x += 72
    b += text(124, 268, "Say → Ask: queue the message.  Encode: chat template + tokenizer.  prefill: the prompt in a step.", "tx")
    b += text(124, 284, "Then per token: one engine step for every conversation at once, GPU sampling, StreamingDecoder,", "tx")
    b += text(124, 300, "and the next frame drains the text event and grows the speech bubble.", "tx")
    write("tour_request_path.svg", 318, "One message end to end", b)


def tour_action_path():
    """One reply as a strip of what gets written, by whom: the model's words
    (no line breaks allowed), the engine's lead-in, then the action, the
    model's picks and the grammar's bytes alternating; then the game."""
    marked = load("actions.json")["constrained"]["examples"]["quest"]
    reply = load("action_logits.json")["reply"]
    words = reply if len(reply) <= 34 else reply[:31] + "..."
    b = text(40, 22, "One reply, by who writes it", "th")
    y = 70
    x = 40
    parts = [(words, "model"), ("↵ACTION:", "engine")]
    parts += [(visible(p), "model" if chosen else "grammar") for p, chosen in segments(marked)]
    colors = {"model": ACCENT["purple"], "engine": ACCENT["blue"], "grammar": GRAMMAR_GRAY}
    spans = []
    for s, who in parts:
        w = len(s) * MONO_W + 8
        b += fill_rect(x, y - 11, w - 2, 22, colors[who], 0.3 if who != "grammar" else 0.55, rx=3)
        b += mono(x + 4, y, s, "ts")
        spans.append((x, x + w - 2))
        x += w
    b += text((spans[0][0] + spans[0][1]) / 2, y - 26, "speaking", "tx", "middle")
    b += text((spans[2][0] + spans[-1][1]) / 2, y - 26, "acting: under the grammar", "tx", "middle")
    b += line(spans[1][1] + 1, y - 34, spans[1][1] + 1, y + 16, ACCENT["gray"], 1, dash=True)
    y += 44
    b += arrow(x - 20, y - 26, x - 20, y - 4, ACCENT["gray"]) if x < WIDTH - 20 else ""
    b += text(40, y + 6, "kAction event → Market::Act: the item goes into your pockets, with a delivery to make", "ts")
    y += 30
    for i, (who, label) in enumerate((("model", "the model samples"), ("engine", "the engine writes"),
                                      ("grammar", "the grammar fixes it (jump-forward)"))):
        b += fill_rect(40 + i * 180, y - 6, 12, 12, colors[who], 0.4 if who != "grammar" else 0.55, rx=2)
        b += text(58 + i * 180, y, label, "tx")
    write("tour_action_path.svg", y + 20, "One reply: words, then an action under a grammar", b)


if __name__ == "__main__":
    for make in (tour_system, tour_reading_order, tour_load_path, tour_request_path, tour_action_path):
        make()
    print("code tour diagrams written")
