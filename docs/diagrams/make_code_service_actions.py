"""Code pictures for the worker thread, actions under a grammar and
speculative decoding: each function drawn as it runs on a small real
input, with its own variables and the lines of src/ that do each part.

- code_service_run.svg: InferenceService::Run and the game's calls on one
  time axis. Frame and step lengths are real (data/market_runs.json, the
  ten-minute `murmur-lab market` run); the order of the calls is an example.
- code_action_phases.svg: the end of a real reply and its action, token by
  token, through Engine::Accept, StartAction, AcceptAction, WriteForced and
  SamplingFor. data/code_action_phases.json holds a greedy replay of the
  engine's logic with the model (Mei's system prompt and grammar from
  data/actions.json); it is refreshed when models/qwen2.5-1.5b-instruct and
  the reference environment (torch, transformers, regex) are present:
    reference/.venv/Scripts/python.exe docs/diagrams/make_code_service_actions.py
- code_verify.svg: Engine::Verify on a real step of data/speculative_fp8.json
  (`murmur-lab speculative`, greedy, 4 guesses per step).

Run from anywhere: python docs/diagrams/make_code_service_actions.py
"""
import json
import math
import os
import re

from code_lines import lines
from svg_kit import ACCENT, arrow, circle, fill_rect, label_in, line, rect, svg, text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL = os.path.join(HERE, "..", "..", "models", "qwen2.5-1.5b-instruct")
CHAR = 6.1   # width of one monospace character at 11px


def write(name, height, title, body):
    svg(HERE, name, height, title, body)


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def shown(token):
    """A token's text with its spaces and line breaks made visible."""
    return token.replace(" ", "␣").replace("\n", "\\n")


def code_notes(notes, y):
    """One line per (line range, what it does) pair, from y down."""
    b = ""
    for k, (where, what) in enumerate(notes):
        b += text(24, y + k * 17, where, "tx mono") + text(130, y + k * 17, what, "tx")
    return b


# --- InferenceService::Run ---------------------------------------------------

FRAME_MS = 1000 / 60   # the game's target frame rate, kTargetFps
LOCK_MS = 0.45         # a lock hold, drawn wider than its few microseconds


def service_run():
    """The game thread and the worker on one time axis: the game's calls
    queue Commands under mutex_ and notify work_ready_; the worker sleeps in
    the wait until there is work, swaps the commands out under the lock,
    applies them and runs engine->Step() without it, then publishes the
    step's events under the lock for the game's next TakeEvents."""
    step_ms = load("market_runs.json")["ten_minutes"]["step_p50_ms"]
    end_ms = 75.0
    x0, x1 = 130, 656
    game_y, worker_y, lane_h = 84, 186, 26

    def tx(t):
        return x0 + t / end_ms * (x1 - x0)

    def hold(t, y):
        return fill_rect(tx(t), y - 3, tx(LOCK_MS) - tx(0), lane_h + 6, ACCENT["amber"], rx=1)

    b = text(24, 22, "InferenceService::Run: the worker thread and the game's calls on one time axis", "th")
    b += text(24, game_y + lane_h / 2, "game thread", "ts") + text(24, worker_y + lane_h / 2, "worker: Run()", "ts")

    # The game: a frame every 16.7 ms, each starting with TakeEvents() and Stats().
    frames = []
    t = 0.0
    while t < end_ms:
        frames.append(t)
        b += rect(tx(t) + 1, game_y, tx(min(t + FRAME_MS, end_ms)) - tx(t) - 2, lane_h, "gray")
        b += hold(t + 0.3, game_y)
        t += FRAME_MS
    b += text(tx(frames[3]) + 6, game_y - 14, "TakeEvents, Stats", "tx mono")
    calls = [(15.5, "Open(Mei)"), (28.0, "Ask(Mei)"), (38.0, "Ask(Kenji)")]
    for t, name in calls:
        b += hold(t, game_y) + text(tx(t) + 6, game_y - 14, name, "tx mono")

    # The worker, replayed: asleep in the wait until a call notifies it;
    # then each pass swaps the queued commands out and applies them, and
    # while a reply is running, runs a step and publishes its events.
    def wait(start, stop, label):
        return (line(tx(start), worker_y + lane_h - 6, tx(stop), worker_y + lane_h - 6, ACCENT["gray"], 1.2, dash=True)
                + text(tx(start) + 4, worker_y + 8, label, "tx mono"))

    def swap(t, taken):
        b = hold(t, worker_y) + text(tx(t), worker_y + lane_h + 14, taken, "tx mono")
        if taken != "[ ]":   # Apply each command
            b += fill_rect(tx(t + LOCK_MS), worker_y + 4, tx(LOCK_MS) - tx(0), lane_h - 8, ACCENT["purple"])
            t += LOCK_MS
        return b, t + LOCK_MS

    part, t = swap(calls[0][0], "[kOpen]")   # then the wait takes the lock to check, and sleeps
    b += wait(0, calls[0][0], "work_ready_.wait") + part + hold(t, worker_y)
    b += wait(t + LOCK_MS, calls[1][0], "wait again")
    t = calls[1][0]
    publishes = []
    for taken in ("[kAsk]", "[kAsk]", "[ ]", "[ ]"):
        part, t = swap(t, taken)
        stop = min(t + step_ms, end_ms)
        b += part + rect(tx(t), worker_y, tx(stop) - tx(t), lane_h, "teal")
        if stop - t > 12:
            b += label_in(tx(t), worker_y, tx(stop) - tx(t), lane_h, "teal", "engine->Step()", cls="tx", mono=True)
        if stop == end_ms:
            break
        b += hold(stop, worker_y)
        publishes.append(stop)
        t = stop + LOCK_MS

    # Who wakes whom, and where commands and events wait for the other thread.
    for t, _ in calls[:2]:
        b += arrow(tx(t) + 2, game_y + lane_h + 3, tx(t) + 2, worker_y - 4, ACCENT["coral"], 1.2)
    busy_ask, taken_at = calls[2][0], publishes[0] + LOCK_MS
    queue_y = worker_y - 24
    b += line(tx(busy_ask) + 2, game_y + lane_h + 3, tx(busy_ask) + 2, queue_y, ACCENT["coral"], 1.2, dash=True)
    b += line(tx(busy_ask) + 2, queue_y, tx(taken_at), queue_y, ACCENT["gray"], 1.0, dash=True)
    b += arrow(tx(taken_at), queue_y, tx(taken_at), worker_y - 4, ACCENT["gray"], 1.0)
    b += text(tx(busy_ask) - 4, queue_y, "commands_", "tx mono", "end")
    for t in publishes:
        frame = next((f for f in frames if f > t), None)
        if frame is not None:
            b += arrow(tx(t) + 1, worker_y - 4, tx(frame) + 3, game_y + lane_h + 5, ACCENT["teal"], 1.0)
    b += text(tx(publishes[0]) + 26, (game_y + lane_h + worker_y) / 2, "events_", "tx mono")

    axis = worker_y + lane_h + 30
    b += line(x0, axis, x1, axis, None, 0.8)
    for v in range(0, int(end_ms) + 1, 10):
        b += line(tx(v), axis, tx(v), axis + 4, None, 0.8) + text(tx(v), axis + 14, f"{v} ms", "tx", "middle")

    y = axis + 42
    legend = [(ACCENT["amber"], "holds mutex_ (drawn wider: each hold lasts microseconds)"),
              (ACCENT["purple"], "Apply(command)"), (ACCENT["gray"], "frame")]
    x = 24
    for color, what in legend:
        b += fill_rect(x, y - 5, 10, 10, color, 0.9 if color != ACCENT["gray"] else 0.4, rx=2) + text(x + 16, y, what, "tx")
        x += 34 + len(what) * 5.2
    b += text(24, y + 22, f"Frames 16.7 ms (60 fps); steps {step_ms} ms, the median step of murmur-lab market "
                          "(30 NPCs). The calls are an example.", "ts")
    b += text(24, y + 40, "The worker holds mutex_ only to swap commands_ out and to publish; Step() runs without it.", "ts")
    f = "engine/inference_service.cpp"
    notes = [
        (lines(f, "commands_.push_back(std::move(command));", "work_ready_.notify_one();"),
         "Open, Ask, ... push a Command under the lock, then notify the worker"),
        (lines(f, "const bool busy = engine->HasWork();", "commands.swap(commands_);"),
         "sleep until a reply is running or a command is queued; swap all queued commands out"),
        (lines(f, "for (Command& command : commands) {", "if (!engine->HasWork()) {"),
         "apply them without the lock; with no reply running, go back to the wait"),
        (lines(f, "std::vector<EngineEvent> events = engine->Step();", "stats_ = std::move(stats);\n        } catch"),
         "run one step without the lock, then lock to append its events and copy the stats"),
        (lines(f, "return std::exchange(events_, {});"), "the game's TakeEvents takes every event under the lock"),
    ]
    b += code_notes(notes, y + 66)
    write("code_service_run.svg", y + 66 + len(notes) * 17 + 6, "InferenceService::Run on a time axis", b)


# --- The speaking and acting phases --------------------------------------------

def action_phases_data():
    """The token rows of code_action_phases.svg, replayed with the model
    when it and the reference environment are present."""
    path = os.path.join(DATA, "code_action_phases.json")
    try:
        import regex
        import torch
        import transformers
        refresh = os.path.exists(MODEL)
    except ImportError:
        refresh = False
    if refresh:
        data = replay_action(regex, torch, transformers)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def byte_decoder():
    """The byte-level tokenizer's character for each byte, reversed."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    shifted = [b for b in range(256) if b not in printable]
    chars = {b: chr(b) for b in printable} | {b: chr(256 + k) for k, b in enumerate(shifted)}
    return {c: b for b, c in chars.items()}


def grammar_regex(regex, grammar):
    """The grammar of src/engine/grammar.hpp as one regular expression:
    literals escaped, sets and repeats as they are, rules inlined."""
    rules = dict(re.match(r"\s*(\w+)\s*::=\s*(.*)", line).groups()
                 for line in grammar.splitlines() if line.strip() and not line.lstrip().startswith("#"))
    item = re.compile(r'\s*(?:"((?:[^"\\]|\\.)*)"|(\[[^\]]*\]|\{\d+(?:,\d+)?\}|[?*+|()])|(\w+))')

    def expand(name):
        out, body, pos = "", rules[name], 0
        while pos < len(body):
            m = item.match(body, pos)
            pos = m.end()
            literal, operator, rule = m.groups()
            if literal is not None:
                out += regex.escape(re.sub(r"\\(.)", lambda e: "\n" if e[1] == "n" else e[1], literal))
            else:
                out += operator or f"(?:{expand(rule)})"
        return out

    return regex.compile(expand(next(iter(rules))))


class ActionGrammar:
    """TokenGrammar's tables for one state at a time. A state is named by the
    action text written so far (the automaton's state after those bytes);
    a token's text is its bytes read as Latin-1, one character per byte."""

    def __init__(self, pattern, token_text, end_token, encode):
        self.pattern, self.token_text, self.end, self.encode = pattern, token_text, end_token, encode
        self.by_first = {}
        for token, t in enumerate(token_text):
            if t:
                self.by_first.setdefault(t[0], []).append(token)

    def alive(self, text):
        return self.pattern.fullmatch(text, partial=True) is not None

    def accepting(self, text):
        return self.pattern.fullmatch(text) is not None

    def allowed(self, state):
        """Every token whose bytes keep the automaton alive; the end token in accepting states."""
        out = [t for first, tokens in self.by_first.items() if self.alive(state + first)
               for t in tokens if self.alive(state + self.token_text[t])]
        return sorted(out + ([self.end] if self.accepting(state) else []))

    def forced(self, state):
        """ForcedBytes as tokens, without a last token the model could merge
        with its next choice; or the one allowed token when that is all."""
        text = ""
        while not self.accepting(state + text):
            only = [chr(b) for b in range(256) if self.alive(state + text + chr(b))]
            if len(only) != 1:
                break
            text += only[0]
        if not text:
            allowed = self.allowed(state)
            return allowed if len(allowed) == 1 else []
        tokens = self.encode(text)
        before_last = state + "".join(self.token_text[t] for t in tokens[:-1])
        last = self.token_text[tokens[-1]]
        if any(t != self.end and len(self.token_text[t]) > len(last) and self.token_text[t].startswith(last)
               for t in self.allowed(before_last)):
            tokens.pop()
        return tokens


def replay_action(regex, torch, transformers):
    """Mei answers the first question of data/action_logits.json, greedy,
    the way the engine runs a conversation that acts: the reply under the
    speech mask until a stop token, then StartAction (the lead-in), then the
    action under the grammar's masks with jump-forward. Keeps the reply's
    last two tokens and everything after."""
    actions, question = load("actions.json"), load("action_logits.json")["question"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    special = set(tokenizer.get_added_vocab().values())
    decode = byte_decoder()
    token_text = [None] * len(tokenizer)
    for piece, token in tokenizer.get_vocab().items():
        if token not in special:
            token_text[token] = bytes(decode[c] for c in piece).decode("latin-1")
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stops = {end, tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    speech = [t for t, s in enumerate(token_text) if t in stops or (s is not None and "\n" not in s and "(" not in s)]
    grammar = ActionGrammar(grammar_regex(regex, actions["grammar"]), token_text, end,
                            lambda s: tokenizer.encode(s, add_special_tokens=False))

    messages = [{"role": "system", "content": actions["system_prompt_mei"]}, {"role": "user", "content": question}]
    pending, cache = tokenizer.apply_chat_template(messages, add_generation_prompt=True), None

    def sample(allowed):
        """One engine step: run the pending tokens, take the top allowed token."""
        nonlocal pending, cache
        with torch.no_grad():
            out = model(torch.tensor([pending]), past_key_values=cache, use_cache=True)
        pending, cache = [], out.past_key_values
        logits = out.logits[0, -1, : len(token_text)]
        return allowed[int(torch.argmax(logits[allowed]))]

    def text_of(token):
        if token in special:
            return tokenizer.decode([token])
        return token_text[token].encode("latin-1").decode("utf-8", errors="replace")

    rows, state = [], ""   # state: the action text so far, which names grammar_state

    def write_forced():
        """WriteForced: returns True once the end token finished the action."""
        nonlocal state
        while forced := grammar.forced(state):
            for token in forced:
                rows.append({"text": text_of(token), "by": "forced"})
                if token == end:
                    return True
                state += token_text[token]
                pending.append(token)
        return False

    reply = []
    while (token := sample(speech)) not in stops:
        reply.append(token)
        pending.append(token)
    rows += [{"text": text_of(t), "by": "speech", "allowed": len(speech)} for t in reply[-2:]]
    rows.append({"text": text_of(token), "by": "stop", "allowed": len(speech)})
    for token in tokenizer.encode("\nACTION:", add_special_tokens=False):
        rows.append({"text": text_of(token), "by": "lead_in"})
        pending.append(token)
    finished = write_forced()
    while not finished:
        allowed = grammar.allowed(state)
        token = sample(allowed)
        rows.append({"text": text_of(token), "by": "grammar", "allowed": len(allowed)})
        if token == end:
            break
        state += token_text[token]
        pending.append(token)
        finished = write_forced()
    return {"question": question, "reply": tokenizer.decode(reply), "rows": rows}


# Who wrote a token: the ramp of its cell, and the function that appended
# it (none for the stop token, which is sampled but never appended).
WRITERS = {"speech": ("teal", "Accept"), "stop": ("teal", None), "lead_in": ("gray", "StartAction"),
           "grammar": ("purple", "AcceptAction"), "forced": ("gray", "WriteForced")}


def action_phases():
    """The end of a real reply and its action, one cell per token, grouped
    by the engine step that wrote it. Above each token the model chose: how
    many tokens SamplingFor's mask allowed (the speech mask, then the
    grammar's DeviceMask(grammar_state)). Below: the function that appended
    it, the grammar_state chain (one dot per state; each token moves it to
    Next(grammar_state, token)) and the phase."""
    data = action_phases_data()
    rows = data["rows"]
    widths = [max(16, len(shown(r["text"])) * CHAR + 6) for r in rows]
    gap, x0 = 1.5, 112
    xs = [x0 + sum(widths[:i]) + gap * i for i in range(len(rows))]
    band_top, bar_base, tok_y, tok_h = 62, 154, 158, 24

    def bar_height(allowed):
        return 11 * math.log10(allowed)

    reply = data["reply"]
    reply = "…" + reply[-60:].split(" ", 1)[1] if len(reply) > 60 else reply
    b = text(24, 22, "A reply's last tokens and its action: who writes each token, and under which mask", "th")
    b += text(24, 44, f"You: “{data['question']}”   Mei (greedy): “{reply}”", "tx")

    # One engine step per token the model chose; the tokens written after it come in the same step.
    steps = [i for i, r in enumerate(rows) if "allowed" in r] + [len(rows)]
    for k, (first, stop) in enumerate(zip(steps, steps[1:])):
        left, right = xs[first] - 0.75, xs[stop - 1] + widths[stop - 1] + 0.75
        if k % 2 == 0:
            b += fill_rect(left, band_top, right - left, tok_y + tok_h + 3 - band_top, ACCENT["gray"], 0.12, rx=2)
        b += text((left + right) / 2, band_top + 8, str(k + 1), "tx", "middle")
    b += text(24, band_top + 8, "Engine::Step", "tx mono")

    # The mask SamplingFor gave the sampler, as the count of tokens it allows (log scale).
    b += text(24, bar_base - 30, "tokens the", "tx") + text(24, bar_base - 17, "mask allows", "tx")
    for i, r in enumerate(rows):
        if "allowed" in r:
            h = bar_height(r["allowed"])
            color = ACCENT["purple"] if r["by"] == "grammar" else ACCENT["teal"]
            b += fill_rect(xs[i] + widths[i] / 2 - 4, bar_base - h, 8, h, color, 0.9, rx=1)
            if r["by"] == "grammar":
                b += text(xs[i] + widths[i] / 2, bar_base - h - 8, str(r["allowed"]), "tx", "middle")
    speech = [i for i, r in enumerate(rows) if r["by"] in ("speech", "stop")]
    middle = (xs[speech[0]] + xs[speech[-1]] + widths[speech[-1]]) / 2
    allowed = rows[speech[0]]["allowed"]
    b += text(middle, bar_base - bar_height(allowed) - 8, f"{allowed:,}", "tx", "middle")

    b += text(24, tok_y + tok_h / 2, "token", "tx")
    for i, r in enumerate(rows):
        ramp = WRITERS[r["by"]][0]
        b += rect(xs[i], tok_y, widths[i], tok_h, ramp) + label_in(xs[i], tok_y, widths[i], tok_h, ramp,
                                                                   shown(r["text"]), cls="tx", mono=True)
        if r["by"] == "stop":
            b += line(xs[i] + 3, tok_y + tok_h / 2, xs[i] + widths[i] - 3, tok_y + tok_h / 2, ACCENT["red"], 1.2)

    # The function that appended each run of tokens.
    y = tok_y + tok_h + 18
    b += text(24, y, "appended by", "tx")
    first, named = 0, 0
    while first < len(rows):
        name = WRITERS[rows[first]["by"]][1]
        last = first
        while last + 1 < len(rows) and WRITERS[rows[last + 1]["by"]][1] == name:
            last += 1
        if name:   # neighbouring names alternate between two lines so they never overlap
            left, right = xs[first] + 1, xs[last] + widths[last] - 1
            b += line(left, y - 10, right, y - 10, ACCENT["gray"], 1.0)
            b += text((left + right) / 2, y + 12 * (named % 2), name, "tx mono", "middle")
            named += 1
        first = last + 1
    y += 12

    # grammar_state: Start() before the action's first token, then Next(grammar_state, token)
    # after each one, until the end token's Next is kFinished.
    y += 28
    b += text(24, y, "grammar_state", "tx mono")
    acting = [i for i, r in enumerate(rows) if r["by"] in ("grammar", "forced")]
    dots = [xs[i] for i in acting] + [xs[acting[-1]] + widths[acting[-1]]]
    b += line(dots[0], y, dots[-1], y, ACCENT["purple"], 1.0)
    for x in dots[:-1]:
        b += circle(x, y, 3, ACCENT["purple"])
    b += circle(dots[-1], y, 3, ACCENT["gray"])
    b += text(dots[0] - 6, y, "Start()", "tx mono", "end")
    b += text((dots[-2] + dots[-1]) / 2, y - 10, "kFinished", "tx mono", "middle")

    # phase: kSpeaking until the stop token, kActing through the action, kActed at its end.
    y += 26
    b += text(24, y, "phase", "tx mono")
    lead = next(i for i, r in enumerate(rows) if r["by"] == "lead_in")
    spans = [("kSpeaking", 0, lead - 1, "teal"), ("kActing", lead, len(rows) - 2, "purple"),
             ("kActed", len(rows) - 1, len(rows) - 1, "gray")]
    for name, first, last, ramp in spans:
        left, right = xs[first], xs[last] + widths[last]
        b += rect(left, y - 9, right - left, 18, ramp) + label_in(left, y - 9, right - left, 18, ramp, name, "tx", True)

    y += 34
    legend = [("teal", "chosen by the model under the speech mask"), ("purple", "chosen under the grammar's mask"),
              ("gray", "written without a model step")]
    x = 24
    for ramp, what in legend:
        b += rect(x, y - 6, 12, 12, ramp) + text(x + 18, y, what, "tx")
        x += 34 + len(what) * 5.2
    b += text(24, y + 22, "The stop token is sampled but not appended: Accept calls StartAction, which appends the "
                          "lead-in instead.", "ts")
    b += text(24, y + 40, "Each dot is a grammar_state; each action token moves it on. Tokens written without a model "
                          "step go", "ts")
    b += text(24, y + 58, "through the model in the next step, with the token it chooses then.", "ts")
    f = "engine/engine.cpp"
    notes = [
        (lines(f, "SamplingRequest request{.params = c.options.sampling",
               "request.allowed = actions_.grammar->DeviceMask(c.grammar_state);"),
         "SamplingFor: the speech mask while kSpeaking, DeviceMask(grammar_state) while kActing"),
        (lines(f, "if (std::ranges::find(stop_tokens_, token) != stop_tokens_.end()) {", "Finish(c, false, events);"),
         "Accept: a stop token starts the action instead of being appended"),
        (lines(f, "c.action_start = c.tokens.size();", "c.grammar_state = actions_.grammar->Start();"),
         "StartAction: append the lead-in, phase = kActing, grammar_state = Start()"),
        (lines(f, "void Engine::AcceptAction(", "if (AppendActionToken(c, token, next, false, events)) {"),
         "AcceptAction: grammar_state = Next(grammar_state, token); the end token finishes"),
        (lines(f, "for (std::span<const TokenId> forced = actions_.grammar->Forced(c.grammar_state);",
               "if (!AppendActionToken(c, token, next, true, events)) {"),
         "WriteForced: append each token Forced(grammar_state) holds, without a model step"),
    ]
    b += code_notes(notes, y + 84)
    write("code_action_phases.svg", y + 84 + len(notes) * 17 + 6, "Speaking and acting, token by token", b)


# --- Engine::Verify -----------------------------------------------------------

DRAFT_TOKENS = 4
STAND_IN = " most"   # the rejected guess: the lab records only which guesses were kept


def speculative_steps(pieces):
    """A reply's [text, guessed] pieces as engine steps: each step emits the
    guesses it kept, then one token of the model's own."""
    steps, kept = [], []
    for piece, guessed in pieces:
        if guessed:
            kept.append(piece)
        else:
            steps.append((kept, piece))
            kept = []
    return steps


def verify():
    """Engine::Verify on a real step that kept all guesses but the last:
    step_tokens (the newest token and the draft's guesses), checked (the
    model's token after each), the accept loop's comparisons, what Accept
    appends to c.tokens, and the cache before and after Truncate."""
    spec = load("speculative_fp8.json")
    run = next(r for r in spec["runs"] if r["draft_tokens"] == DRAFT_TOKENS)
    steps = speculative_steps(run["first_reply"])
    s = next(s for s in range(1, len(steps)) if len(steps[s][0]) == DRAFT_TOKENS - 1)
    kept, bonus = steps[s]
    newest = steps[s - 1][1]
    step_tokens = [newest, *kept, STAND_IN]
    checked = [*kept, bonus, None]   # the row after the stand-in is sampled but never read
    accepted = len(kept)
    so_far = "".join(piece for k, b in steps[:s] for piece in (*k, b))

    x0, cw, w, h = 166, 81, 75, 24
    cols = [x0 + c * cw for c in range(DRAFT_TOKENS + 2)]
    dashed = ' stroke-dasharray="3 2"'
    b = text(24, 22, f"Engine::Verify on a real step: {DRAFT_TOKENS} guesses, {accepted} kept, "
                     "then one token of the model's own", "th")
    b += text(24, 44, f"Mei, greedy, draft_tokens = {DRAFT_TOKENS}, reply so far: “…{so_far[-48:]}”", "tx")

    y = 66
    b += text(24, y, "position in c.kv", "tx")
    for c in range(DRAFT_TOKENS + 1):
        b += text(cols[c] + w / 2, y, f"before + {c}", "tx mono", "middle")

    y += 12
    b += text(24, y + h / 2, "c.step_tokens", "tx mono")
    for c, token in enumerate(step_tokens):
        ramp = "gray" if c == 0 else "teal"
        b += rect(cols[c], y, w, h, ramp, extra=dashed if token == STAND_IN else "")
        b += label_in(cols[c], y, w, h, ramp, shown(token), "tx", True)

    # The accept loop compares each guess with the model's token for the same position.
    y += h
    for a in range(DRAFT_TOKENS):
        same = checked[a] == step_tokens[1 + a]
        b += text(cols[1 + a] + w / 2, y + 11, "=" if same else "≠", "ts", "middle",
                  f' style="fill:{ACCENT["green"] if same else ACCENT["red"]};font-weight:600"')
    b += text(24, y + 11, "checked[a] == step_tokens[1 + a]", "tx mono")

    y += 22
    b += text(24, y + h / 2, "checked", "tx mono")
    for r, token in enumerate(checked):
        ramp = "purple" if token is not None else "gray"
        b += rect(cols[r + 1], y, w, h, ramp) + label_in(cols[r + 1], y, w, h, ramp,
                                                         shown(token) if token else "not read", "tx", token is not None)
        b += text(cols[r + 1] + w / 2, y + h + 9, f"[{r}]", "tx mono", "middle")
    b += text(cols[DRAFT_TOKENS + 1] + w / 2, y - 11, f"accepted = {accepted}", "tx mono", "middle")

    # Accept(checked[j]) for j = 0..accepted appends to c.tokens; the last is the model's own.
    top = y + h
    y += h + 44
    b += text(24, y + h / 2, "c.tokens", "tx mono")
    b += rect(cols[0], y, w, h, "gray") + label_in(cols[0], y, w, h, "gray", shown(newest), "tx", True)
    for j in range(accepted + 1):
        x = cols[j + 1] + w / 2
        b += arrow(x, top + 14, x, y - 3, ACCENT["amber"], 1.1)
        b += rect(cols[j + 1], y, w, h, "amber") + label_in(cols[j + 1], y, w, h, "amber", shown(checked[j]), "tx", True)
    b += text(cols[1] - 8, (top + 14 + y) / 2, "Accept(checked[j])", "tx mono", "end")

    # The pass cached the newest token and every guess; Truncate keeps the ones c.tokens still has.
    rows = [("c.kv after the pass", DRAFT_TOKENS + 1), ("after Truncate", accepted + 1)]
    y += h + 20
    for label, length in rows:
        b += text(24, y + 6, label, "tx mono") + text(24, y + 18, f"length = before + {length}", "tx mono")
        for c, token in enumerate(step_tokens):
            if c < length:
                b += rect(cols[c], y, w, h, "teal") + label_in(cols[c], y, w, h, "teal", shown(token), "tx", True)
            elif length < DRAFT_TOKENS + 1:
                b += rect(cols[c], y, w, h, "gray", extra=dashed)
                b += line(cols[c] + 4, y + h / 2, cols[c] + w - 4, y + h / 2, ACCENT["red"], 1.2)
        y += h + 10

    y += 14
    b += text(24, y, f"“{shown(STAND_IN)}” (dashed) stands in for the 4th guess: the lab records only that it was not "
                     f"“{shown(bonus)}”.", "ts")
    b += text(24, y + 18, f"Teal: the draft's guesses, and the cache. Amber: what Accept appended. In this run "
                          f"{run['kept_percent']:.0f}% of guesses were kept.", "ts")
    f = "engine/engine.cpp"
    notes = [
        (lines(f, "c.step_tokens.assign(1, c.tokens.back());", "p.drafted = drafted;"),
         "Speculate: step_tokens = the newest token, then the guesses"),
        (lines(f, ".tokens = c.step_tokens, .logits = p.count, .decode_each_token = true"),
         "Entries: each step token is its own decode row with its own logits"),
        (lines(f, "c.kv.length = before + 1 + drafted;", "++accepted;"),
         "Verify: count the guesses that equal the model's token at their position"),
        (lines(f, "for (int j = 0; j <= accepted && c.replying", "Accept(c, checked[j], events, j < accepted);"),
         "Accept each kept guess, then checked[accepted], the model's own"),
        (lines(f, "int valid = 0;", "Cache().Truncate(c.kv, c.kv.length);"),
         "keep the cached positions c.tokens still holds; drop the rest"),
    ]
    b += code_notes(notes, y + 44)
    write("code_verify.svg", y + 44 + len(notes) * 17 + 6, "Engine::Verify step by step", b)


DIAGRAMS = [service_run, action_phases, verify]

if __name__ == "__main__":
    for make in DIAGRAMS:
        make()
    print(f"{len(DIAGRAMS)} code pictures written")
