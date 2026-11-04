"""Diagrams for docs/serving_performance.md: batching, the continuous
batching trace, the player's time to first token, prefix sharing and the
memory policy. Real data comes from data/batch_curve.json (`murmur-lab
batch` runs), data/schedule.json (`murmur-lab schedule --json`),
data/market_prompts.json (the market's 32 system prompts) and
data/market_runs.json (`murmur-lab market` runs).

Run from anywhere: python docs/diagrams/make_serving_diagrams.py
"""
import json
import os

from svg_kit import ACCENT, circle, fill_rect, line, path, svg, text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def write(name, height, title, body):
    svg(HERE, name, height, title, body)


def batch_curve():
    """Measured: total tokens/s (bars) rises almost in step with the batch
    size while each conversation's own speed (line) falls only slowly."""
    data = load("batch_curve.json")
    batches, tps, ms = data["batch"], data["tokens_per_second"], data["ms_per_step"]
    b = text(40, 22, "Batching: one read of the weights serves every conversation in the step", "th")
    X0, X1, Y0, Y1 = 90, 600, 50, 250
    n = len(batches)
    bw = (X1 - X0) / n
    top = 7000
    for v in (0, 2000, 4000, 6000):
        yy = Y1 - v / top * (Y1 - Y0)
        b += line(X0, yy, X1, yy, None, 0.4, dash=True) + text(X0 - 6, yy, f"{v:,}", "tx", "end")
    per = [1000 / m for m in ms]
    pts = []
    for i, (batch, t) in enumerate(zip(batches, tps)):
        x = X0 + i * bw + 8
        h = t / top * (Y1 - Y0)
        b += fill_rect(x, Y1 - h, bw - 16, h, ACCENT["teal"], 0.85, rx=2)
        b += text(x + (bw - 16) / 2, Y1 - h - 9, f"{t:,}", "tx", "middle")
        b += text(x + (bw - 16) / 2, Y1 + 14, str(batch), "tx", "middle")
        pts.append((x + (bw - 16) / 2, Y1 - per[i] / 250 * (Y1 - Y0)))
    b += path("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts), color=ACCENT["amber"], width=2, arrow_end=False)
    for x, y in pts:
        b += circle(x, y, 3, ACCENT["amber"])
    b += text(pts[0][0] + 8, pts[0][1] - 10, f"{per[0]:.0f} tokens/s each", "tx")
    b += text(pts[-1][0] - 4, pts[-1][1] - 12, f"{per[-1]:.0f} each", "tx", "end")
    b += text(40, (Y0 + Y1) / 2, "tokens/s", "tx", "middle", f' transform="rotate(-90 40 {(Y0 + Y1) / 2})"')
    b += text((X0 + X1) / 2, Y1 + 32, "conversations generating together", "ts", "middle")
    b += fill_rect(X0, Y1 + 50, 12, 12, ACCENT["teal"], 0.85, rx=2) + text(X0 + 18, Y1 + 56, "all conversations together", "tx")
    b += circle(X0 + 216, Y1 + 56, 4, ACCENT["amber"]) + text(X0 + 226, Y1 + 56, "each conversation (same scale ÷ 28)", "tx")
    b += text(40, Y1 + 84, "Measured on the RTX 4080 SUPER, 512 tokens of context each: 64 conversations get 32× the total", "ts")
    b += text(40, Y1 + 102, "throughput of one, and each still generates 99 tokens a second.", "ts")
    write("batch_curve.svg", Y1 + 120, "Batching throughput and latency", b)


def schedule_grid():
    """A real engine trace: conversations down, steps across. Prompts run in
    chunks (wide cells, with their token counts) within a budget per step;
    decoding conversations add one token each (dots); everyone joins and
    leaves at any step."""
    data = load("schedule.json")
    names, steps, priority = data["names"], data["steps"], data["priority"]
    b = text(40, 22, "Continuous batching: who runs in each step (a real trace, 24 prompt tokens per step)", "th")
    X0, Y0, cw, rh = 110, 50, 30, 30
    for s in range(len(steps)):
        b += text(X0 + s * cw + cw / 2, Y0 - 8, str(s), "tx", "middle")
    for r, name in enumerate(names):
        y = Y0 + r * rh
        color = ACCENT["amber"] if priority[r] > 0 else ACCENT["teal"]
        b += text(X0 - 10, y + rh / 2, name + (" (priority)" if priority[r] > 0 else ""), "ts", "end")
        b += line(X0, y + rh / 2, X0 + len(steps) * cw, y + rh / 2, None, 0.3, dash=True)
        for s, step in enumerate(steps):
            tokens = step["tokens"][r]
            x = X0 + s * cw
            if tokens > 1:
                h = 6 + 16 * tokens / 24
                b += fill_rect(x + 2, y + (rh - h) / 2, cw - 4, h, color, 0.85, rx=2)
                b += text(x + cw / 2, y + rh / 2, str(tokens), "tx", "middle", ' style="fill:#fff"')
            elif tokens == 1:
                b += circle(x + cw / 2, y + rh / 2, 4, color)
    # The prompt budget used in each step.
    by = Y0 + len(names) * rh + 16
    b += text(X0 - 10, by + 10, "prompt budget", "tx", "end")
    for s, step in enumerate(steps):
        used = sum(t for t in step["tokens"] if t > 1)
        h = 20 * used / 24
        b += fill_rect(X0 + s * cw + 4, by + 20 - h, cw - 8, h, ACCENT["gray"], 0.7, rx=1)
    b += text(40, by + 44, "Wide cells: a chunk of a prompt (its token count); dots: one generated token. Mei's prompt takes two steps;", "ts")
    b += text(40, by + 62, "Kenji starts in step 1 with the leftover budget; in step 6 your prompt goes first and Rafael's waits.", "ts")
    write("schedule_trace.svg", by + 80, "Continuous batching trace", b)


def player_first_token():
    """Why the player waits at most about two steps: a message arrives during
    a step, joins the next one, and its first token comes out at that step's
    end. Lanes on a time axis."""
    data = load("market_runs.json")["ten_minutes"]
    b = text(40, 22, "Time to the player's first word: at most the rest of one step plus the next", "th")
    X0, scale = 110, 8.0   # px per ms
    step_ms = data["step_p50_ms"]
    y = 60
    b += text(X0 - 10, y + 12, "engine steps", "ts", "end")
    for k in range(4):
        x = X0 + k * step_ms * scale
        ramp = "amber" if k == 2 else "teal"
        b += fill_rect(x + 1, y, step_ms * scale - 2, 24, ACCENT[ramp], 0.85 if k == 2 else 0.45, rx=3)
        b += text(x + step_ms * scale / 2, y + 12, f"step {k}" + (" + your prompt" if k == 2 else ""), "tx", "middle",
                  ' style="fill:#fff"')
    arrive = X0 + 1.45 * step_ms * scale
    first = X0 + 3 * step_ms * scale
    b += path(f"M{arrive} {y + 50} L{arrive} {y + 28}", color=ACCENT["amber"], width=1.4)
    b += text(arrive, y + 60, "you press Enter", "tx", "middle")
    b += path(f"M{first} {y + 50} L{first} {y + 28}", color=ACCENT["amber"], width=1.4)
    b += text(first, y + 60, "first word", "tx", "middle")
    b += line(arrive, y + 76, first, y + 76, ACCENT["amber"], 1.2)
    b += text((arrive + first) / 2, y + 88, f"measured: p50 {data['player_ttft_p50_ms']:.0f} ms, p99 {data['player_ttft_p99_ms']:.0f} ms", "tx", "middle")
    b += text(40, y + 120, "Steps can't be interrupted, so a message waits for the current one to end; it then runs at the front of the", "ts")
    b += text(40, y + 138, "next step because it outranks the NPCs' chatter. With 30 NPCs talking, a step takes about 17 ms.", "ts")
    write("player_first_token.svg", y + 156, "Player time to first token", b)


def prefix_sharing_tree():
    """The city lore as one trunk of pages, computed once, from which all 32
    NPCs' prompts branch off; and the memory that saves, to scale."""
    data = load("market_prompts.json")
    shared, private = data["shared_pages"], data["private_tokens"]
    npcs = data["npcs"]
    b = text(40, 22, f"Prefix sharing: the city lore is computed once for all {npcs} NPCs", "th")
    X0, cw, Y = 40, 13, 60
    for p in range(shared):
        b += fill_rect(X0 + p * cw, Y, cw - 2, 14, ACCENT["gray"], 0.75, rx=1)
        b += f'<rect x="{X0 + p * cw - 0.5:.1f}" y="{Y - 0.5:.1f}" width="{cw - 1}" height="15" fill="none" stroke="#fff" stroke-width="0.8"/>'
    b += text(X0, Y - 12, f"{shared} pages of lore ({data['lore_tokens']} tokens), shared", "tx")
    trunk_end = X0 + shared * cw
    # 32 branches fanning out, each 2-3 pages of the NPC's own prompt.
    for i, tokens in enumerate(private):
        pages = (tokens + 15) // 16
        yy = 44 + i * 5.2
        color = f"hsl({(i * 137.508) % 360:.0f},55%,62%)"
        b += path(f"M{trunk_end} {Y + 7} C {trunk_end + 30} {Y + 7}, {trunk_end + 30} {yy}, {trunk_end + 60} {yy}",
                  color=color, width=0.8, arrow_end=False)
        for p in range(pages):
            b += f'<rect x="{trunk_end + 62 + p * 7:.1f}" y="{yy - 2:.1f}" width="6" height="4" fill="{color}"/>'
    b += text(trunk_end + 90, Y + 4, "each NPC's own 2–3 pages", "tx")
    # The saving, to scale.
    full = sum(shared + (t + 15) // 16 for t in private)
    with_sharing = shared + sum((t + 15) // 16 for t in private)
    BY, scale = 250, 470 / full
    b += text(X0, BY - 14, "pages for every NPC's system prompt, to scale", "ts")
    b += fill_rect(X0, BY, full * scale, 14, ACCENT["coral"], 0.8, rx=2)
    b += text(X0 + full * scale + 6, BY + 7, f"{full} without sharing", "tx")
    b += fill_rect(X0, BY + 22, with_sharing * scale, 14, ACCENT["teal"], 0.9, rx=2)
    b += text(X0 + with_sharing * scale + 6, BY + 29, f"{with_sharing} with sharing", "tx")
    write("prefix_sharing.svg", BY + 56, "Prefix sharing", b)


def memory_policy_bars():
    """The same 30-NPC load before and after bounding each NPC's memory and
    stopping waiting conversations from evicting running ones."""
    data = load("market_runs.json")
    before, after = data["before"], data["after"]
    b = text(40, 22, "When memory runs out: the same 30 talking NPCs, before and after the memory policy", "th")
    metrics = [("tokens per second", "tokens_per_second", "{:,}"), ("conversations per step", "median_batch", "{}"),
               ("evictions in 30 s", "evictions", "{:,}")]
    X0 = 60
    for k, (label, key, fmt) in enumerate(metrics):
        x = X0 + k * 205
        top = max(before[key], after[key]) * 1.15
        b += text(x, 52, label, "ts")
        for j, (run, ramp, name) in enumerate(((before, "gray", "before"), (after, "teal", "after"))):
            h = run[key] / top * 120
            bx = x + j * 70
            b += fill_rect(bx, 190 - h, 52, max(h, 1.5), ACCENT[ramp], 0.85, rx=2)
            b += text(bx + 26, 190 - h - 9, fmt.format(run[key]), "tx", "middle")
            b += text(bx + 26, 204, name, "tx", "middle")
    b += text(40, 232, "Before: conversations grew without limit, and a conversation waiting to start could evict one in the middle of", "ts")
    b += text(40, 250, "a reply, so they kept evicting each other and recomputing. After: each NPC remembers at most 960 tokens", "ts")
    b += text(40, 268, "(the lore, its persona, its latest turns), and only idle or lower-priority conversations are evicted.", "ts")
    write("memory_policy.svg", 286, "Memory policy before and after", b)


if __name__ == "__main__":
    for make in (batch_curve, schedule_grid, player_first_token, prefix_sharing_tree, memory_policy_bars):
        make()
    print("serving diagrams written")
