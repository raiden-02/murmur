"""Code pictures for the engine's scheduling (src/engine/engine.cpp): one
conversation's tokens and pages, one Engine::Step, Engine::Plan and
Engine::MakeRoom, each drawn as it runs on real numbers, with the code's
own variables and the lines of src/ that do each part.

Data:
- data/code_conversation.json: Mei's market conversation. The system
  prompt's length comes from data/market_prompts.json; the two player turns
  are the market's first two cues (src/game/market.cpp) and the reply is
  Mei's greedy answer to the first (`murmur-lab generate "<cue>"`). Their
  token counts are refreshed with the model's tokenizer when
  models/qwen2.5-1.5b-instruct is present.
- data/code_plan.json: a real `murmur-lab schedule --json` trace (five
  conversations, max_prefill_tokens 24). Plan and Step's bookkeeping are
  replayed over it and checked against every step; rerun that command and
  copy its steps over to refresh it.
- MakeRoom has no recorded trace: it runs on a small scenario with the
  market's sizes (every NPC shares the 20 lore pages).

Run from anywhere: python docs/diagrams/make_code_engine.py
"""
import json
import os
import re

from code_lines import lines
from svg_kit import ACCENT, chip, fill_rect, label_in, line, rect, svg, text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL = os.path.join(HERE, "..", "..", "models", "qwen2.5-1.5b-instruct", "tokenizer.json")
ENGINE = "engine/engine.cpp"
PAGE = 16   # TransformerOptions::page_size: tokens per KV-cache page


def write(name, height, title, body):
    svg(HERE, name, height, title, body)


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def colored(color):
    """A text style attribute: an accent fill that reads in light and dark."""
    return f' style="fill:{color}"'


def notes_block(y, notes, x_text=118):
    """'lines a–b  what they do' rows, as in the other code pictures."""
    b = ""
    for k, (where, what) in enumerate(notes):
        b += text(24, y + k * 17, where, "tx mono") + text(x_text, y + k * 17, what, "tx")
    return b


# --- Token counts ----------------------------------------------------------

def byte_chars():
    """GPT-2's byte-to-character table, as tokenizer.json spells tokens."""
    kept = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = kept[:]
    shifted = 0
    for b in range(256):
        if b not in kept:
            kept.append(b)
            chars.append(256 + shifted)
            shifted += 1
    return {b: chr(c) for b, c in zip(kept, chars)}


# Qwen2's pre-tokenizer split, with \p{L} and \p{N} written for Python's re:
# exact for ASCII text, which is all the texts counted here.
PIECES = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|(?:[^\w\r\n]|_)?[^\W\d_]+|\d| ?(?:[^\w\s]|_)+[\r\n]*"
                    r"|\s*[\r\n]+|\s+(?!\S)|\s+")


class TokenCounter:
    """How many tokens the model's tokenizer makes of a text, and of the chat
    template's pieces (model/chat_template.cpp)."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            merges = json.load(f)["model"]["merges"]
        self.ranks = {}
        for rank, merge in enumerate(merges):
            self.ranks.setdefault(tuple(merge.split(" ") if isinstance(merge, str) else merge), rank)
        self.chars = byte_chars()

    def piece(self, piece):
        symbols = list(piece)
        while len(symbols) > 1:
            best = min(range(len(symbols) - 1), key=lambda i: self.ranks.get((symbols[i], symbols[i + 1]), 1 << 30))
            if (symbols[best], symbols[best + 1]) not in self.ranks:
                break
            symbols[best:best + 2] = [symbols[best] + symbols[best + 1]]
        return len(symbols)

    def count(self, text):
        return sum(self.piece("".join(self.chars[b] for b in p.encode())) for p in PIECES.findall(text))

    def message(self, role, content):
        """EncodeMessage: <|im_start|>, role + \\n + content, <|im_end|>, \\n."""
        return 1 + self.count(role + "\n" + content) + 1 + self.count("\n")

    def generation_prompt(self):
        """EncodeGenerationPrompt: <|im_start|>, assistant + \\n."""
        return 1 + self.count("assistant\n")


# --- Conversation ----------------------------------------------------------

def conversation_data():
    """Mei's conversation: system prompt length, the two turns' and the
    reply's token counts (recounted when the tokenizer is present)."""
    path = os.path.join(DATA, "code_conversation.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["system_tokens"] = load("market_prompts.json")["system_tokens"][0]   # Mei is NPC 0
    if os.path.exists(MODEL):
        counter = TokenCounter(MODEL)
        data["turn_tokens"] = [counter.message("user", cue) + counter.generation_prompt() for cue in data["cues"]]
        data["reply_tokens"] = counter.count(data["reply"])
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({k: v for k, v in data.items() if k != "system_tokens"}, f, ensure_ascii=False, indent=1)
    return data


def conversation_moments(data):
    """The conversation's fields after Send(turn 2) and after the step that
    runs it. Send appends the turn after the 2 tokens Finish wrote (still
    uncached); the step caches every uncached token (a priority-0 prompt this
    short fits the 256-token prefill budget) and Accept appends the first
    reply token."""
    system = data["system_tokens"]
    turn1, turn2 = data["turn_tokens"]
    reply = data["reply_tokens"]
    segments = [("system", 0, system), ("turn", system, system + turn1),
                ("reply", system + turn1, system + turn1 + reply)]
    finish = system + turn1 + reply             # kv.length when the reply ended
    second = finish + 2                          # turn_starts[1]: after <|im_end|> and \n
    segments += [("finish", finish, second), ("turn", second, second + turn2)]
    after_send = {"segments": segments, "size": second + turn2, "kv": finish,
                  "turn_starts": [system, second], "pages": -(-finish // PAGE)}
    ran = after_send["size"]
    after_step = {"segments": segments + [("reply", ran, ran + 1)], "size": ran + 1, "kv": ran,
                  "turn_starts": [system, second], "pages": -(-ran // PAGE)}
    return after_send, after_step


SEGMENT_RAMP = {"system": "gray", "turn": "purple", "reply": "teal", "finish": "coral"}


def code_conversation():
    """Engine::Conversation for Mei: the whole `tokens` array with its pages,
    then the end of it, one cell per token, after Send and after the next
    Step. Solid cells are cached (the first kv.length); amber is what
    changed."""
    data = conversation_data()
    after_send, after_step = conversation_moments(data)
    system = data["system_tokens"]
    b = text(24, 22, "Engine::Conversation: Mei's tokens, what is cached (kv.length) and kv.pages", "th")

    # Overview: every token and every page at one scale.
    end_pages = after_step["pages"]
    ox, ow = 24, 632
    k = ow / (end_pages * PAGE)
    y = 56
    b += text(ox, y - 12, f"tokens: all {after_step['size']}, after the step", "ts")
    for kind, start, stop in after_step["segments"]:
        b += fill_rect(ox + start * k, y, (stop - start) * k, 12, ACCENT[SEGMENT_RAMP[kind]], 0.85)
    b += label_in(ox, y, system * k, 12, "gray", f"system prompt: system_tokens = {system}", cls="tx")
    py = y + 16
    for p in range(end_pages):
        full = (p + 1) * PAGE <= after_step["kv"]
        b += rect(ox + p * PAGE * k + 0.5, py, PAGE * k - 1, 9, "green" if full else "gray", rx=1)
    lore = load("market_prompts.json")["shared_pages"]
    lx = ox + lore * PAGE * k
    b += line(ox, py + 14, lx - 1, py + 14, ACCENT["gray"], 1.0)
    b += text(ox, py + 24, f"kv.pages 0–{lore - 1}: the lore every NPC's prompt starts with, shared through the prefix cache",
              "tx")
    zoom_page = (system - 1) // PAGE          # the page where the system prompt ends
    zx0 = ox + zoom_page * PAGE * k
    b += rect(zx0, y - 3, ox + ow - zx0, 31, "gray", rx=2, extra=' fill-opacity="0" stroke-dasharray="3 2"')
    b += text(ox + ow, py + 24, "enlarged below", "tx", "end")

    # The end of the array, one cell per token, at two moments.
    first = zoom_page * PAGE
    cell = ow / ((end_pages - zoom_page) * PAGE)
    labels = {"system": "system", "turn": "turn", "reply": "reply", "finish": ""}

    def x_of(i):
        return ox + (i - first) * cell

    def moment(y, state, before, title):
        out = text(24, y, title, "ts")
        strip = y + 40
        cached, size = state["kv"], state["size"]
        for i, t in enumerate(state["turn_starts"]):
            out += line(x_of(t), strip - 18, x_of(t), strip - 1, ACCENT["purple"], 1.2)
            out += text(x_of(t) + 3, strip - 22, f"turn_starts[{i}] = {t}", "tx mono")
        for kind, start, stop in state["segments"]:
            for i in range(max(start, first), stop):
                new = i >= before["size"] if before else start == state["turn_starts"][1]
                ramp = "amber" if new else SEGMENT_RAMP[kind]
                if i < cached:
                    out += fill_rect(x_of(i) + 0.5, strip, cell - 1, 16, ACCENT[ramp], 0.85, rx=1)
                else:
                    out += rect(x_of(i) + 0.5, strip, cell - 1, 16, ramp, rx=1, extra=' fill-opacity="0.15"')
            s = max(start, first)
            if stop - s >= 5 and labels[kind]:
                out += text(x_of((s + stop) / 2), strip - 8, labels[kind], "tx", "middle")
        for p in range(zoom_page, state["pages"]):
            published = (p + 1) * PAGE <= cached
            new = before is not None and (p >= before["pages"] or published and (p + 1) * PAGE > before["kv"])
            ramp = "amber" if new else "green" if published else "gray"
            out += rect(x_of(p * PAGE) + 0.5, strip + 21, PAGE * cell - 1, 14, ramp, rx=2)
            out += label_in(x_of(p * PAGE), strip + 21, PAGE * cell, 14, ramp, f"page {p}", cls="tx")
        kx, ex = x_of(cached), x_of(size)
        out += line(kx, strip - 4, kx, strip + 52, ACCENT["blue"], 1.6)
        out += text(kx - 4, strip + 48, f"kv.length = {cached}", "tx mono", "end", colored(ACCENT["blue"]))
        uncached = f"Uncached() = {size - cached}"
        if ex - kx > 120:
            mid, half = (kx + ex) / 2, len(uncached) * 3.2
            out += line(kx + 2, strip + 48, mid - half - 4, strip + 48, ACCENT["amber"], 1.0)
            out += line(mid + half + 4, strip + 48, ex, strip + 48, ACCENT["amber"], 1.0)
            out += text(mid, strip + 48, uncached, "tx mono", "middle", colored(ACCENT["amber"]))
        else:
            out += text(ex, strip + 62, uncached, "tx mono", "end", colored(ACCENT["amber"]))
        return out

    y = py + 50
    b += moment(y, after_send, None, "after Send(): the new turn (amber) is appended, not cached (outlined)")
    y += 112
    b += moment(y, after_step, after_send, "after the next Step(): the turn is cached; Accept appended the first reply token")
    y += 124
    b += fill_rect(24, y - 5, 10, 10, ACCENT["coral"], 0.85, rx=2)
    b += text(38, y, "<|im_end|> \\n that Finish wrote after the reply", "tx")
    b += fill_rect(300, y - 5, 10, 10, ACCENT["green"], 0.85, rx=2)
    b += text(314, y, "full page: published to the prefix cache", "tx")
    hpp = "engine/engine.hpp"
    notes = [
        (lines(ENGINE, "c.turn_starts.push_back(c.tokens.size());", "c.prompt_tokens = c.Uncached();"),
         "Send: record where the turn starts, append it; the reply's prompt is everything uncached"),
        (lines(ENGINE, "c.kv.length += plan[i].count;"), "SampleNext: the step's tokens are cached now"),
        (lines(ENGINE, "const int full_pages = c.kv.length / Cache().PageSize();", "c.published_pages = full_pages;"),
         "Publish: offer the pages that just filled up to the prefix cache"),
        (lines(ENGINE, "Append(c, std::span(&end_of_turn, 1));", "Append(c, template_.EncodeAfterReply());"),
         "Finish: append <|im_end|> and \\n, cached with the next turn"),
        (lines(hpp, "int Uncached() const"), "Uncached() = tokens.size() − kv.length (engine.hpp)"),
    ]
    y += 22
    b += notes_block(y, notes)
    write("code_conversation.svg", y + len(notes) * 17 + 4, "One conversation's tokens and pages", b)


# --- Plan and Step, replayed over the schedule trace -----------------------

def prompt_lengths(trace):
    """Each conversation's prompt: the tokens it ran from its Send until its
    first decode token."""
    lengths = {}
    for step, who in trace["sends"]:
        total = 0
        for row in trace["steps"][step:]:
            if row["tokens"][who] == 1:
                break
            total += row["tokens"][who]
        lengths[who] = total
    return lengths


def plan(convs, trace):
    """Engine::Plan on `convs` (dicts with size, kv, priority, last_active,
    sent): the runnable ones sorted as the code sorts them, then the budget
    loop. Returns one row per conversation the loop looked at, and why it
    ended. MakeRoom always succeeds here: the 1 GB cache never fills."""
    runnable = [c for c in convs if c["replying"] and c["size"] - c["kv"] > 0]
    runnable.sort(key=lambda c: (-c["priority"], 0, c["last_active"]) if c["size"] - c["kv"] == 1
                  else (-c["priority"], 1, c["sent"]))
    rows_left, prefill_left = trace["max_chunk"], trace["max_prefill_tokens"]
    rows, planned = [], 0
    end = "every runnable conversation was looked at"
    for c in runnable:
        if planned == trace["max_batch"] or rows_left == 0:
            end = "plan.size() == max_batch" if planned == trace["max_batch"] else "rows_left == 0"
            break
        uncached = c["size"] - c["kv"]
        budget = rows_left if c["priority"] > 0 else min(prefill_left, rows_left)
        count = 1 if uncached == 1 else min(uncached, budget)
        row = {"c": c, "uncached": uncached, "budget": budget, "count": count,
               "prefill_before": prefill_left, "rows_before": rows_left}
        if count > 0:
            planned += 1
            rows_left -= count
            if uncached > 1:
                prefill_left = max(0, prefill_left - count)
        row.update(prefill_left=prefill_left, rows_left=rows_left)
        rows.append(row)
    return rows, end


def replay_schedule(trace):
    """Plan and the token bookkeeping of Step over the whole trace, checked
    against the counts the engine recorded. Returns per step Plan's rows
    (each with its conversation's state before the step) and why it ended."""
    names = trace["names"]
    prompts = prompt_lengths(trace)
    convs = [{"name": n, "id": i, "priority": trace["priority"][i], "size": 0, "kv": 0, "pages": 0,
              "replying": False, "last_active": 0, "sent": 0} for i, n in enumerate(names)]
    steps = []
    for s, row in enumerate(trace["steps"]):
        for order, (step, who) in enumerate(trace["sends"]):
            if step == s:
                convs[who].update(size=prompts[who], kv=0, pages=0, replying=True, sent=order, sent_step=s)
        before = [dict(c) for c in convs]
        rows, end = plan(convs, trace)
        counts = [0] * len(names)
        for r in rows:
            counts[r["c"]["id"]] = r["count"]
        if counts != row["tokens"]:
            raise ValueError(f"step {s}: Plan replay gives {counts}, the engine ran {row['tokens']}")
        steps.append({"rows": [dict(r, c=before[r["c"]["id"]]) for r in rows], "end": end})
        for r in rows:
            c = convs[r["c"]["id"]]
            if r["count"] == 0:
                continue
            c["pages"] = max(c["pages"], -(-(c["kv"] + r["count"]) // PAGE))   # MakeRoom → Reserve
            c["kv"] += r["count"]                                              # SampleNext
            c["last_active"] = s
            if r["count"] == r["uncached"]:                                    # a logits row: Accept
                c["size"] += 1
                c["replying"] = c["id"] not in row["finished"]
    return steps


# --- Step ------------------------------------------------------------------

STEP_SHOWN = 3   # a decode, a prompt's last chunk and a prompt's first chunk in one batch


def code_engine_step():
    """Engine::Step on a real step: its calls in order (rows) and what each
    leaves in the three planned conversations (columns): their tokens
    (solid: the first kv.length, cached), their pages, and the numbers."""
    trace = load("code_plan.json")
    rows = replay_schedule(trace)[STEP_SHOWN]["rows"]
    b = text(24, 22, f"Engine::Step on step {STEP_SHOWN} of murmur-lab schedule: what each call changes", "th")
    col_x, col_w = [176, 338, 500], 156
    span = max(-(-(r["c"]["kv"] + r["count"] + 1) // PAGE) for r in rows) * PAGE
    unit = col_w / span
    b += text(24, 50, "calls, in order", "tx")
    for x, r in zip(col_x, rows):
        logits = r["count"] == r["uncached"]
        kind = "decoding" if r["uncached"] == 1 else "a prompt's last chunk" if logits else "a prompt's first chunk"
        b += text(x, 50, r["c"]["name"], "ts") + text(x + len(r["c"]["name"]) * 7 + 8, 50, kind, "tx")

    def cell(x, y, size, kv, pages, published, entry=None, written=None, new_cached=None, new_token=False,
             new_pages=()):
        out = ""
        for i in range(size):
            if new_token and i == size - 1 or new_cached and new_cached[0] <= i < new_cached[1]:
                ramp = "amber"
            elif entry and entry[0] <= i < entry[1]:
                ramp = "purple"
            else:
                ramp = "teal" if i < kv else "gray"
            out += fill_rect(x + i * unit, y, unit - 0.6, 14, ACCENT[ramp], 0.3 if ramp == "gray" else 0.9)
        for p in range(pages):
            ramp = "amber" if p in new_pages else "green" if p < published else "gray"
            out += rect(x + p * PAGE * unit + 0.5, y + 17, PAGE * unit - 1, 9, ramp, rx=1)
        if written:
            out += fill_rect(x + written[0] * unit, y + 19.5, (written[1] - written[0]) * unit, 4, ACCENT["purple"])
        return out

    calls = [("Plan", "MakeRoom reserves pages"), ("Speculate", ""), ("Entries", "the forward pass's input"),
             ("transformer_->Step", "one forward pass"), ("SampleNext", "after the pass"), ("TakeTokens", "")]
    y = 72
    for name, what in calls:
        b += text(24, y + 7, name, "ts mono") + (text(24, y + 23, what, "tx") if what else "")
        if name == "Speculate":
            b += text(col_x[0], y + 7, "returns at once: this engine has no Speculator", "tx")
            y += 32
            continue
        for x, r in zip(col_x, rows):
            c, count = r["c"], r["count"]
            kv, size = c["kv"], c["size"]
            logits = count == r["uncached"]
            pages = -(-(kv + count) // PAGE)
            entry = (kv, kv + count)
            if name == "Plan":
                drawn = cell(x, y, size, kv, pages, kv // PAGE, new_pages=range(c["pages"], pages))
                caption = f"count {count}" + (f", {pages - c['pages']} new pages" if pages > c["pages"] else "")
            elif name == "Entries":
                drawn = cell(x, y, size, kv, pages, kv // PAGE, entry=entry)
                caption = f"tokens [{kv}, {kv + count}), logits {int(logits)}"
            elif name == "transformer_->Step":
                drawn = cell(x, y, size, kv, pages, kv // PAGE, entry=entry, written=entry)
                caption = f"kv.length {kv}: not moved"
            elif name == "SampleNext":
                drawn = cell(x, y, size, kv + count, pages, (kv + count) // PAGE, new_cached=entry,
                             new_pages=range(kv // PAGE, (kv + count) // PAGE))
                caption = f"kv.length {kv + count}, " + ("sampled" if logits else "no logits row")
            else:
                drawn = cell(x, y, size + logits, kv + count, pages, (kv + count) // PAGE, new_token=logits)
                caption = f"tokens.size() {size + logits}" + ("" if logits else ": no token")
            b += drawn + text(x, y + 38, caption, "tx mono")
        y += 54

    y += 4
    b += text(24, y, "The forward pass writes the keys and values into the pages (purple bar), but kv.length moves only", "ts")
    b += text(24, y + 17, "in SampleNext; there the pages that are now full are published (amber page, then green).", "ts")
    y += 40
    legend = [("teal", "cached"), ("gray", "not cached"), ("purple", "this step's entry"),
              ("amber", "changed by this call"), ("green", "published page")]
    x = 24
    for ramp, what in legend:
        b += fill_rect(x, y - 5, 10, 10, ACCENT[ramp], 0.9, rx=2) + text(x + 14, y, what, "tx")
        x += 28 + len(what) * 5.6
    notes = [
        (lines(ENGINE, "std::vector<Planned> plan = Plan(events);", "++stats_.steps;"), "Step: the calls in order"),
        (lines(ENGINE, "if (count <= 0 || !MakeRoom(*c, c->kv.length + count, plan)) {"),
         "Plan: MakeRoom reserves pages for kv.length + count"),
        (lines(ENGINE, ".tokens = std::span(c.tokens).subspan(c.kv.length, p.count),",
               ".logits = p.count == c.Uncached() ? 1 : 0});"),
         "Entries: the next count tokens; a logits row only if they reach the end"),
        (lines(ENGINE, "c.kv.length += plan[i].count;", "requests.push_back(SamplingFor(c, TakeDraw(c), c.distinct));"),
         "SampleNext: advance kv.length, Publish, then one sampling request per logits row"),
        (lines(ENGINE, "Accept(c, next[row++], events);"), "TakeTokens: Accept appends the sampled token (not cached)"),
    ]
    y += 22
    b += notes_block(y, notes)
    write("code_engine_step.svg", y + len(notes) * 17 + 4, "One engine step, call by call", b)


# --- Plan ------------------------------------------------------------------

PLAN_STEPS = [3, 6]   # a prompt cut to the budget; the player's prompt taking the whole step


def code_plan():
    """Engine::Plan on two real steps: the runnable conversations in sorted
    order (rows), the tokens each gets, and prefill_left and rows_left after
    each one."""
    trace = load("code_plan.json")
    steps = replay_schedule(trace)
    b = text(24, 22, "Engine::Plan on two real steps: the sorted runnable list and the tokens each one gets", "th")
    b += text(24, 42, f"murmur-lab schedule: max_prefill_tokens {trace['max_prefill_tokens']}, "
                      f"max_chunk {trace['max_chunk']}, max_batch {trace['max_batch']}; 'you' has Priority() 1", "ts")
    cols = {"name": 24, "prio": 96, "key": 162, "bar": 290, "budget": 470, "prefill": 516, "rows": 662}
    tok = 2.6           # px per token in the count bars
    pre = 4.0           # px per token in the prefill_left bars
    y = 70
    for s in PLAN_STEPS:
        step = steps[s]
        b += text(24, y, f"step {s}", "th")
        y += 20
        heads = [("name", "runnable", "start"), ("prio", "Priority()", "start"), ("key", "then", "start"),
                 ("bar", "count of Uncached()", "start"), ("budget", "budget", "start"),
                 ("prefill", "prefill_left", "start"), ("rows", "rows_left", "end")]
        for col, head, anchor in heads:
            b += text(cols[col], y, head, "tx", anchor)
        y += 18
        start = step["rows"][0]
        b += text(cols["name"], y + 8, "start", "tx")
        b += fill_rect(cols["prefill"], y + 3, start["prefill_before"] * pre, 10, ACCENT["teal"], 0.85, rx=1)
        b += text(cols["rows"], y + 8, str(start["rows_before"]), "tx mono", "end")
        y += 22
        for r in step["rows"]:
            c = r["c"]
            decodes = r["uncached"] == 1
            skipped = r["count"] == 0
            b += chip(cols["name"], y, 64, 16, "coral" if skipped else "gray", c["name"], cls="tx")
            b += text(cols["prio"] + 26, y + 8, str(c["priority"]), "tx mono", "middle")
            key = f"decode, last_active {c['last_active']}" if decodes else f"prompt, sent in step {c['sent_step']}"
            b += text(cols["key"], y + 8, key, "tx")
            bx = cols["bar"]
            b += rect(bx, y + 2, r["uncached"] * tok, 12, "gray", rx=1, extra=' fill-opacity="0.1"')
            if r["count"]:
                cut = r["count"] < r["uncached"]
                b += fill_rect(bx, y + 2, r["count"] * tok, 12, ACCENT["amber" if cut else "teal"], 0.85, rx=1)
            label = "skipped: count 0" if skipped else f"{r['count']} of {r['uncached']}" if r["count"] < r["uncached"] \
                else str(r["count"])
            b += text(bx + r["uncached"] * tok + 5, y + 8, label, "tx mono",
                      extra=colored(ACCENT["coral"]) if skipped else "")
            b += text(cols["budget"], y + 8, "—" if decodes else str(r["budget"]), "tx mono")
            px = cols["prefill"]
            b += fill_rect(px, y + 3, r["prefill_left"] * pre, 10, ACCENT["teal"], 0.85, rx=1)
            used = r["prefill_before"] - r["prefill_left"]
            if used:
                b += fill_rect(px + r["prefill_left"] * pre, y + 3, used * pre, 10, ACCENT["amber"], 0.85, rx=1)
            b += text(px + r["prefill_before"] * pre + 4, y + 8, str(r["prefill_left"]), "tx mono")
            changed = r["rows_left"] != r["rows_before"]
            b += text(cols["rows"], y + 8, str(r["rows_left"]), "tx mono", "end",
                      colored(ACCENT["amber"]) if changed else "")
            y += 22
        planned = sum(1 for r in step["rows"] if r["count"])
        b += text(24, y + 4, f"Loop ends: {step['end']} (plan.size() {planned} < max_batch "
                             f"{trace['max_batch']}, rows_left {step['rows'][-1]['rows_left']} > 0).", "tx")
        y += 30
    b += text(24, y, "Amber: a prompt cut to its budget, and the tokens each row takes from prefill_left and rows_left.", "ts")
    b += text(24, y + 17, "A decode always gets 1 token; a Priority() 0 prompt gets at most prefill_left; a higher one, rows_left.", "ts")
    notes = [
        (lines(ENGINE, "std::ranges::sort(runnable,", "return a_decodes ? a->last_active < b->last_active : a->sent_at < b->sent_at;"),
         "sort: higher Priority() first, then decodes, then least recently run / asked first"),
        (lines(ENGINE, "if (static_cast<int>(plan.size()) == limits.max_batch || rows_left == 0) {"),
         "stop when the batch is full or the step has no rows left"),
        (lines(ENGINE, "const int budget = c->Priority() > 0 ? rows_left : std::min(prefill_left, rows_left);",
               "const int count = uncached == 1 ? 1 : std::min(uncached, budget);"),
         "budget, and count: 1 for a decode, else the uncached tokens cut to the budget"),
        (lines(ENGINE, "if (count <= 0 || !MakeRoom(*c, c->kv.length + count, plan)) {"),
         "count 0 (no budget left) or no pages: skip it and look at the next"),
        (lines(ENGINE, "plan.push_back({c, count});", "prefill_left = std::max(0, prefill_left - count);"),
         "take its tokens from rows_left, and from prefill_left if it is a prompt"),
    ]
    y += 40
    b += notes_block(y, notes)
    write("code_plan.svg", y + len(notes) * 17 + 4, "Plan on two real steps", b)


# --- MakeRoom --------------------------------------------------------------

# A market moment: Kenji is answering the player and his next chunk needs
# two more pages; none are free. Every NPC holds the 20 lore pages, shared.
MAKE_ROOM = {
    "c": {"name": "Kenji", "replying": True, "priority": 1, "send_order": 41, "pages": 24, "length": 411},
    "others": [   # in conversations_ order (by id)
        {"name": "Mei", "replying": True, "priority": 0, "send_order": 38, "pages": 31},
        {"name": "Aiko", "replying": False, "priority": 0, "last_active": 1207, "pages": 23},
        {"name": "Rafael", "replying": False, "priority": 0, "last_active": 1193, "pages": 21},
        {"name": "Priya", "replying": True, "priority": 1, "send_order": 39, "pages": 27},
        {"name": "Tomas", "replying": True, "priority": 0, "send_order": 40, "pages": 30, "in_step": True},
        {"name": "Hana", "replying": False, "priority": 0, "last_active": 1150, "pages": 0},
        {"name": "Dmitri", "replying": True, "priority": 0, "send_order": 43, "pages": 29},
        {"name": "Zara", "replying": True, "priority": 1, "send_order": 45, "pages": 24},
    ],
    "shared_pages": 20,
}


def make_room(c, others, shared):
    """Engine::MakeRoom's victim filter, sort and eviction loop. Evicting a
    conversation frees the pages nobody else holds (the shared lore pages
    stay); Reserve succeeds once the free pages cover what `c` lacks."""
    verdicts = []
    for o in others:
        if not o["pages"]:
            verdicts.append("holds no pages")
        elif o.get("in_step"):
            verdicts.append("in this step's plan")
        elif o["priority"] > c["priority"]:
            verdicts.append("higher Priority()")
        elif not o["replying"] or o["priority"] < c["priority"] or o["send_order"] > c["send_order"]:
            verdicts.append(None)
        else:
            verdicts.append("same Priority(), asked earlier")
    victims = [o for o, v in zip(others, verdicts) if v is None]
    victims.sort(key=lambda o: (o["priority"], o["replying"], -o["send_order"] if o["replying"] else o["last_active"]))
    need = -(-c["length"] // PAGE) - c["pages"]
    free, evicted = 0, []
    for v in victims:
        free += v["pages"] - shared
        evicted.append(free)
        if free >= need:
            break
    return verdicts, victims, evicted, need


def code_make_room():
    """Engine::MakeRoom in a market moment: every other conversation (rows,
    in conversations_ order) and whether it may be evicted, then the victims
    in the order the code evicts them, with the free pages adding up until
    Reserve succeeds."""
    s = MAKE_ROOM
    c, others, shared = s["c"], s["others"], s["shared_pages"]
    verdicts, victims, evicted, need = make_room(c, others, shared)
    b = text(24, 22, f"Engine::MakeRoom(Kenji, length {c['length']}): who may be evicted, and in what order", "th")
    b += text(24, 42, f"Kenji answers the player: replying, Priority() {c['priority']}, send_order {c['send_order']}. "
                      f"He holds {c['pages']} pages and needs {c['pages'] + need};", "ts")
    b += text(24, 59, "no free or unused cached pages are left. A scenario at the market's sizes: every NPC holds the "
                      f"{shared} shared lore pages.", "ts")
    cols = {"name": 24, "state": 96, "prio": 152, "key": 216, "bar": 352, "verdict": 486}
    y = 86
    for col, head in [("name", "other"), ("state", "state"), ("prio", "Priority()"), ("key", "send_order or last_active"),
                      ("bar", "kv.pages (gray: shared)"), ("verdict", "victim?")]:
        b += text(cols[col], y, head, "tx")
    y += 14
    unit = 3.0   # px per page
    for o, verdict in zip(others, verdicts):
        victim = verdict is None
        b += chip(cols["name"], y, 64, 16, "teal" if victim else "gray", o["name"], cls="tx")
        b += text(cols["state"], y + 8, "replying" if o["replying"] else "idle", "tx")
        b += text(cols["prio"] + 26, y + 8, str(o["priority"]), "tx mono", "middle")
        key = f"send_order {o['send_order']}" if o["replying"] else f"last_active {o['last_active']}"
        b += text(cols["key"], y + 8, key, "tx mono")
        if o["pages"]:
            b += fill_rect(cols["bar"], y + 3, shared * unit, 10, ACCENT["gray"], 0.6, rx=1)
            b += fill_rect(cols["bar"] + shared * unit, y + 3, (o["pages"] - shared) * unit, 10,
                           ACCENT["teal"], 0.85, rx=1)
            b += text(cols["bar"] + o["pages"] * unit + 4, y + 8, str(o["pages"]), "tx mono")
        b += text(cols["verdict"], y + 8, "victim" if victim else f"no: {verdict}", "tx",
                  extra=colored(ACCENT["teal"]) if victim else "")
        y += 21

    y += 14
    b += text(24, y, "The victims, sorted: lower Priority() first; idle before replying; idle ones least recently active",
              "ts")
    b += text(24, y + 17, "first, replying ones newest first. Bars: free pages after each Evict.", "ts")
    col_w = 116
    k = 14   # px per free page
    base = y + 40 + max(evicted) * k
    right = 24 + col_w * len(victims)
    b += line(24, base, right, base, None, 0.8)
    b += line(24, base - need * k, right, base - need * k, ACCENT["coral"], 1.2, dash=True)
    b += text(right + 6, base - need * k, f"needs {need}", "tx", extra=colored(ACCENT["coral"]))
    for i, v in enumerate(victims):
        x = 24 + i * col_w
        key = f"send_order {v['send_order']}" if v["replying"] else f"last_active {v['last_active']}"
        b += text(x + 4, base + 12, f"{i + 1}. {v['name']}", "ts")
        b += text(x + 4, base + 27, f"{'replying' if v['replying'] else 'idle'}, Priority() {v['priority']}", "tx")
        b += text(x + 4, base + 41, key, "tx mono")
        if i < len(evicted):
            freed = v["pages"] - shared
            before = evicted[i] - freed
            if before:
                b += fill_rect(x + 4, base - before * k, 40, before * k, ACCENT["gray"], 0.5, rx=1)
            b += fill_rect(x + 4, base - evicted[i] * k, 40, freed * k, ACCENT["amber"], 0.9, rx=1)
            fits = evicted[i] >= need
            b += text(x + 50, base - evicted[i] * k + 6, f"frees {freed}", "tx")
            b += text(x + 4, base + 55, "Reserve succeeds" if fits else "Reserve fails", "tx",
                      extra=colored(ACCENT["green"] if fits else ACCENT["coral"]))
        else:
            b += text(x + 4, base + 55, "not evicted", "tx")
    y = base + 80
    b += text(24, y, "Evicting frees the pages only the victim held; it keeps its tokens and recomputes them when it next runs.", "ts")
    notes = [
        (lines(ENGINE, "if (&other == &c || other.kv.pages.empty() || in_step || other.Priority() > c.Priority()) {",
               "victims.push_back(&other);"),
         "victim: holds pages, not in this step, not higher Priority(); idle, lower, or asked later"),
        (lines(ENGINE, "std::ranges::sort(victims,",
               "return a->replying ? a->send_order > b->send_order : a->last_active < b->last_active;"),
         "eviction order"),
        (lines(ENGINE, "for (Conversation* victim : victims) {", "return false;\n}\n\nvoid Engine::Evict"),
         "evict one at a time until Reserve succeeds; false if none is left"),
    ]
    y += 24
    b += notes_block(y, notes)
    write("code_make_room.svg", y + len(notes) * 17 + 4, "MakeRoom's victims", b)


DIAGRAMS = [code_conversation, code_engine_step, code_plan, code_make_room]

if __name__ == "__main__":
    for make in DIAGRAMS:
        make()
    print(f"{len(DIAGRAMS)} code pictures written")
