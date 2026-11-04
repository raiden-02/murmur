"""Code pictures for the tokenizer and the KV pages (code tour Parts I–II):
each function drawn as it runs on a small real input, with its own
variables after every step and the lines of src/ that do each part.

  code_encode_piece.svg      Tokenizer::EncodePiece, one row per pop
  code_page_pool.svg         PagePool: ref_counts_ and free_ after each call
  code_prefix_cache.svg      PrefixCache: PageKey's chain, Insert, Match, EvictOne
  code_kv_cache_manager.svg  KvCacheManager: AttachPrefix, and Reserve growing a page table

The pictures replay the C++ logic in Python, step for step, on real data:
- data/code_encode_piece.json holds the model's merge ranks for every pair
  that can occur inside the piece;
- data/code_market_tokens.json holds two market NPCs' system messages (the
  city lore, then Mei's or Kenji's persona from src/game/personas.hpp) as
  the chat template encodes them, so the KV pages hold real tokens.
Both are refreshed from the model's tokenizer.json when
models/qwen2.5-1.5b-instruct is present. The page bookkeeping is
replayed on small pools (8, 32 and 26 pages; the game's has about 1,450)
with the engine's real page size, 16 tokens.

Run from anywhere: python docs/diagrams/make_code_tokenizer_pages.py
"""
import copy
import heapq
import json
import os
import re

from code_lines import lines
from svg_kit import ACCENT, arrow, fill_rect, label_in, line, rect, svg, text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL = os.path.join(HERE, "..", "..", "models", "qwen2.5-1.5b-instruct", "tokenizer.json")
PERSONAS = os.path.join(HERE, "..", "..", "src", "game", "personas.hpp")
NONE = -1


def write(name, height, title, body):
    svg(HERE, name, height, title, body)


def shown(token):
    """A byte-level token as text, its leading space and newline made visible."""
    return token.replace("Ġ", "␣").replace("Ċ", "↵")


# --- EncodePiece -----------------------------------------------------------

PIECE = "Ġtonight"   # " tonight" in tokenizer.json's byte-level spelling


def encode_piece_data():
    """The merge ranks EncodePiece can look up inside PIECE, refreshed from
    the model's tokenizer when it is present."""
    path = os.path.join(DATA, "code_encode_piece.json")
    if os.path.exists(MODEL):
        with open(MODEL, encoding="utf-8") as f:
            tok = json.load(f)
        ranks = {}
        for rank, merge in enumerate(tok["model"]["merges"]):
            left, right = merge.split(" ") if isinstance(merge, str) else merge
            if left + right in PIECE:
                ranks.setdefault((left, right), rank)
        data = {"piece": PIECE, "id": tok["model"]["vocab"][PIECE],
                "merges": [[left, right, rank] for (left, right), rank in sorted(ranks.items(), key=lambda m: m[1])]}
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def replay_encode_piece(piece, ranks):
    """Tokenizer::EncodePiece step for step. Symbols hold their text in
    place of an id (ids only grow into longer tokens, so equal text is an
    equal id). Returns the state after setup and after every pop."""
    symbols = [[c, i - 1, i + 1 if i + 1 < len(piece) else NONE] for i, c in enumerate(piece)]
    heap = []

    def push_pair(left):
        if left == NONE or symbols[left][2] == NONE:
            return
        pair = (symbols[left][0], symbols[symbols[left][2]][0])
        if pair in ranks:
            heapq.heappush(heap, (ranks[pair], left, *pair))

    for i in range(len(piece) - 1):
        push_pair(i)
    states = [{"pop": None, "symbols": [s[0] for s in symbols], "heap": sorted(heap)}]
    while heap:
        rank, left, a, b = heapq.heappop(heap)
        s = symbols[left]
        stale = s[0] != a or s[2] == NONE or symbols[s[2]][0] != b
        if not stale:
            right = symbols[s[2]]
            s[0], s[2] = a + b, right[2]
            if right[2] != NONE:
                symbols[right[2]][1] = left
            right[0] = None
            push_pair(s[1])
            push_pair(left)
        states.append({"pop": (rank, a, b, stale), "symbols": [s[0] for s in symbols], "heap": sorted(heap)})
    return states


def symbol_spans(symbols):
    """(start, end, text) for each live symbol: it covers the bytes from its
    own index up to the next live one."""
    live = [i for i, s in enumerate(symbols) if s is not None]
    return list(zip(live, live[1:] + [len(symbols)], (symbols[i] for i in live)))


def encode_piece():
    """EncodePiece on " tonight": the symbols array and the heap after setup
    and after every pop of the while loop. Amber marks what each step
    changed; a stale pop changes nothing."""
    data = encode_piece_data()
    states = replay_encode_piece(data["piece"], {(a, b): r for a, b, r in data["merges"]})
    row, top = 25, 76
    sx, cell = 190, 28      # symbols: x of cell 0, width of one cell
    hx, chip = 420, 30      # heap: x of the top entry, width of one entry

    b = text(24, 22, 'EncodePiece(" tonight"): the symbols array and the heap after every pop', "th")
    b += text(24, 52, "popped", "ts") + text(sx, 52, "symbols[0..7]", "ts")
    b += text(hx, 52, "heap, next to pop first", "ts")
    for i in range(len(data["piece"])):
        b += text(sx + i * cell + cell / 2, top - 8, str(i), "tx", "middle")
    before = None
    for r, state in enumerate(states):
        y = top + r * row
        if state["pop"] is None:
            b += text(24, y + 10, "setup", "ts mono")
        else:
            rank, left, right, stale = state["pop"]
            pair = f"{shown(left)}+{shown(right)}"
            b += text(24, y + 10, f"{rank} {pair} {'stale' if stale else 'merge'}", ("tx" if stale else "ts") + " mono")
        for start, end, token in symbol_spans(state["symbols"]):
            ramp = "amber" if before and before["symbols"][start] != token else "teal"
            w = (end - start) * cell - 2
            b += rect(sx + start * cell + 1, y, w, row - 6, ramp) + label_in(
                sx + start * cell + 1, y, w, row - 6, ramp, shown(token), cls="tx", mono=True)
        earlier = list(before["heap"]) if before else None
        for k, entry in enumerate(state["heap"]):
            kept = earlier is None or entry in earlier
            if earlier and entry in earlier:
                earlier.remove(entry)
            ramp = "gray" if kept else "amber"
            b += rect(hx + k * (chip + 1), y, chip, row - 6, ramp) + label_in(
                hx + k * (chip + 1), y, chip, row - 6, ramp, str(entry[0]), cls="tx", mono=True)
        before = state

    y = top + len(states) * row + 10
    b += text(24, y, f"Output: one token, {shown(data['piece'])}, id {data['id']}. "
                     "Amber: what that step changed; a stale pop changes nothing.", "ts")
    f = "model/tokenizer.cpp"
    notes = [
        (lines(f, "std::vector<Symbol> symbols(count);", "for (int i = 0; i + 1 < count; ++i) {"),
         "one symbol per byte, each linked to its neighbours; every neighbouring pair in the merge list is pushed"),
        (lines(f, "while (!heap.empty()) {", "continue;   // stale"),
         "pop the lowest rank; skip it if either symbol has merged with another since it was pushed"),
        (lines(f, "Symbol& right = symbols[left.next];", "push_pair(candidate.left);"),
         "merge the right symbol into the left one, then push the two new neighbouring pairs"),
        (lines(f, "for (int i = count > 0 ? 0 : kNone; i != kNone; i = symbols[i].next) {",
               "out.push_back(symbols[i].id);"), "walk the list from symbol 0 and output each id"),
    ]
    for k, (where, what) in enumerate(notes):
        b += text(24, y + 24 + k * 17, where, "tx mono") + text(130, y + 24 + k * 17, what, "tx")
    write("code_encode_piece.svg", y + 24 + len(notes) * 17 + 6, "EncodePiece step by step", b)


# --- The market's prompts as tokens ------------------------------------------

PAGE_SIZE = 16   # TransformerOptions::page_size
LITERAL = r'"((?:[^"\\]|\\.)*)"'   # the body of one C++ string literal

# The model's pre-tokenizer split (tokenizer.json), restricted to ASCII text:
# \p{L} becomes A-Za-z and \p{N} becomes 0-9.
SPLIT = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+[\r\n]*"
                   r"|\s*[\r\n]+|\s+(?!\S)|\s+", re.ASCII)


def read_personas():
    """kCityLore and kMarketCast's (name, role, manner) from personas.hpp."""
    with open(PERSONAS, encoding="utf-8") as f:
        source = f.read()

    def unquote(body):
        return json.loads(f'"{body}"')

    start = source.index("kCityLore =")
    lore = "".join(unquote(s) for s in re.findall(LITERAL, source[start:source.index('";\n', start) + 1]))
    persona = r"Persona\{" + r",\s*".join([LITERAL] * 3) + r"\}"
    cast = [tuple(map(unquote, fields)) for fields in re.findall(persona, source[source.index("kMarketCast"):])]
    return lore, cast


def byte_symbols():
    """tokenizer.json's byte-level alphabet: the printable character that
    stands for each byte (a space is Ġ, a newline Ċ)."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    others = [b for b in range(256) if b not in printable]
    return {**{b: chr(b) for b in printable}, **{b: chr(256 + n) for n, b in enumerate(others)}}


def encode_text(text_, ranks):
    """The byte-level tokens of ASCII text: split like the model's
    pre-tokenizer, then each piece merged by replay_encode_piece."""
    if not text_.isascii():
        raise ValueError("encode_text handles ASCII text only")
    symbol = byte_symbols()
    tokens = []
    for piece in SPLIT.findall(text_):
        merged = replay_encode_piece("".join(symbol[b] for b in piece.encode()), ranks)[-1]["symbols"]
        tokens += [s for s in merged if s is not None]
    return tokens


def market_tokens():
    """Mei's and Kenji's system messages as ChatTemplate::EncodeMessage
    encodes them, <|im_start|>system\\n{prompt}<|im_end|>\\n, where the prompt
    is MarketSystemPrompt: the city lore, then the persona. Refreshed from
    the model's tokenizer when it is present."""
    path = os.path.join(DATA, "code_market_tokens.json")
    if os.path.exists(MODEL):
        with open(MODEL, encoding="utf-8") as f:
            tok = json.load(f)
        ranks = {}
        for rank, merge in enumerate(tok["model"]["merges"]):
            ranks.setdefault(tuple(merge.split(" ") if isinstance(merge, str) else merge), rank)
        ids = {**tok["model"]["vocab"], **{t["content"]: t["id"] for t in tok["added_tokens"]}}
        lore, cast = read_personas()
        prompts = []
        for name, role, manner in cast[:2]:
            body = encode_text(f"system\n{lore}You are {name}, and you {role}. {manner}", ranks)
            pieces = ["<|im_start|>", *body, "<|im_end|>", *encode_text("\n", ranks)]
            prompts.append({"name": name, "ids": [ids[p] for p in pieces], "pieces": pieces})
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"page_size": PAGE_SIZE, "prompts": prompts}, f, ensure_ascii=False)
    with open(path, encoding="utf-8") as f:
        return {p["name"]: p for p in json.load(f)["prompts"]}


# --- The page bookkeeping, replayed ------------------------------------------

ROOT_KEY = 0x9E3779B97F4A7C15   # kRootKey
MASK = (1 << 64) - 1


class PagePool:
    """PagePool: an owner count per page and the free_ stack (its end is the top)."""

    def __init__(self, page_count):
        self.ref_counts = [0] * page_count
        self.free = list(range(page_count - 1, -1, -1))

    def allocate(self):
        if not self.free:
            return None
        page = self.free.pop()
        self.ref_counts[page] = 1
        return page

    def retain(self, page):
        assert self.ref_counts[page] > 0, "Retain of a free page"
        self.ref_counts[page] += 1

    def release(self, page):
        assert self.ref_counts[page] > 0, "Release of a free page"
        self.ref_counts[page] -= 1
        if self.ref_counts[page] == 0:
            self.free.append(page)


def mix(x):
    """prefix_cache.cpp's Mix: splitmix64's finalizer."""
    x ^= x >> 30
    x = x * 0xBF58476D1CE4E5B9 & MASK
    x ^= x >> 27
    x = x * 0x94D049BB133111EB & MASK
    return x ^ x >> 31


def page_key(parent, page_tokens):
    """PrefixCache::PageKey: the parent's key, then every token, mixed in."""
    key = mix(parent)
    for token in page_tokens:
        key = mix(key ^ token & 0xFFFFFFFF)
    return key


class PrefixCache:
    """PrefixCache: entries_ maps a key to (parent, tokens, page); recency_
    lists the keys, least recently used first."""

    def __init__(self, pool, page_size):
        self.pool, self.page_size = pool, page_size
        self.entries = {}
        self.recency = []

    def pages(self, tokens):
        """The full pages of `tokens`, each as a list of 16 ids."""
        size = self.page_size
        return [tokens[i * size:(i + 1) * size] for i in range(len(tokens) // size)]

    def find(self, key, parent, page_tokens):
        entry = self.entries.get(key)
        return entry if entry and entry[0] == parent and entry[1] == page_tokens else None

    def touch(self, key):
        self.recency.remove(key)
        self.recency.append(key)

    def match(self, tokens, max_pages):
        pages, parent = [], ROOT_KEY
        for page_tokens in self.pages(tokens):
            if len(pages) >= max_pages:
                break
            key = page_key(parent, page_tokens)
            entry = self.find(key, parent, page_tokens)
            if entry is None:
                break
            self.pool.retain(entry[2])
            self.touch(key)
            pages.append(entry[2])
            parent = key
        return pages

    def insert(self, tokens, pages):
        parent = ROOT_KEY
        for page_tokens, page in zip(self.pages(tokens), pages):
            key = page_key(parent, page_tokens)
            if self.find(key, parent, page_tokens):
                self.touch(key)
            elif key in self.entries:
                return
            else:
                self.pool.retain(page)
                self.recency.append(key)
                self.entries[key] = (parent, page_tokens, page)
            parent = key

    def evict_one(self):
        for key in self.recency:
            page = self.entries[key][2]
            if self.pool.ref_counts[page] == 1:
                self.recency.remove(key)
                del self.entries[key]
                self.pool.release(page)
                return True
        return False

    def recency_pages(self):
        return [self.entries[key][2] for key in self.recency]


class KvSequence:
    def __init__(self):
        self.pages, self.length = [], 0


def pages_for(length):
    return (length + PAGE_SIZE - 1) // PAGE_SIZE


class KvCacheManager:
    """KvCacheManager's Reserve, Release, AttachPrefix and Publish."""

    def __init__(self, page_count):
        self.pool = PagePool(page_count)
        self.prefix_cache = PrefixCache(self.pool, PAGE_SIZE)

    def reserve(self, sequence, length):
        had = len(sequence.pages)
        while len(sequence.pages) < pages_for(length):
            page = self.pool.allocate()
            if page is None and self.prefix_cache.evict_one():
                page = self.pool.allocate()
            if page is None:
                while len(sequence.pages) > had:
                    self.pool.release(sequence.pages.pop())
                return False
            sequence.pages.append(page)
        return True

    def release(self, sequence):
        for page in sequence.pages:
            self.pool.release(page)
        sequence.pages, sequence.length = [], 0

    def attach_prefix(self, sequence, tokens):
        if not tokens:
            return 0
        max_pages = (len(tokens) - 1) // PAGE_SIZE
        sequence.pages = self.prefix_cache.match(tokens, max_pages)
        sequence.length = len(sequence.pages) * PAGE_SIZE
        return sequence.length

    def publish(self, sequence, tokens):
        self.prefix_cache.insert(tokens, sequence.pages)


def amber_text(x, y, s, changed, cls="tx", anchor="middle"):
    """Text drawn in the amber accent when `changed`."""
    return text(x, y, s, cls, anchor, f' style="fill:{ACCENT["amber"]}"' if changed else "")


def notes_block(y, notes, column=130):
    """The line ranges of src/ and what each does, one per line."""
    return "".join(text(24, y + k * 17, where, "tx mono") + text(column, y + k * 17, what, "tx")
                   for k, (where, what) in enumerate(notes))


# --- PagePool ----------------------------------------------------------------

OWNER_RAMP = {"A": "teal", "B": "purple", "C": "coral", "D": "green"}
POOL_CALLS = [("A", "Allocate"), ("A", "Allocate"), ("A", "Allocate"), ("B", "Allocate"), ("B", "Allocate"),
              ("C", "Retain", 0), ("C", "Retain", 1), ("C", "Allocate"),
              ("A", "Release", 0), ("A", "Release", 1), ("A", "Release", 2),
              ("B", "Release", 3), ("B", "Release", 4), ("D", "Allocate")]


def page_pool():
    """PagePool(8) through a run of calls by four conversations: A takes
    three pages, B two; C shares A's first two through Retain; A and B
    release theirs; D allocates and gets the page freed last. Each row shows
    ref_counts_ and the free_ stack after the call; amber marks what it
    changed."""
    pool = PagePool(8)
    owners = [[] for _ in pool.ref_counts]   # drawn only: PagePool stores the count, not who
    row, top = 26, 88
    rx, cell = 196, 30      # ref_counts_: x of page 0, width of one page
    fx, slot = 456, 26      # free_: x of its first element (the bottom)

    b = text(24, 22, "PagePool(8): ref_counts_ and the free_ stack after each call", "th")
    b += text(24, 44, "8 pages here; the game's pool has about 1,450, of 16 tokens each.", "ts")
    b += text(24, 64, "caller, call", "ts") + text(rx, 64, "ref_counts_[page]", "ts")
    b += text(fx, 64, "free_ (top on the right)", "ts")
    for page in range(len(owners)):
        b += text(rx + page * cell + cell / 2, top - 8, str(page), "tx", "middle")

    def draw(y, who, call, counts, free):
        """One row: the call, then the pool's state after it. `counts` and
        `free` are the state before it, to mark what changed."""
        r = ""
        if who:
            r += rect(24, y, 18, 18, OWNER_RAMP[who]) + label_in(24, y, 18, 18, OWNER_RAMP[who], who, "tx")
        r += text(50, y + 9, call, "ts mono")
        for p, count in enumerate(pool.ref_counts):
            x = rx + p * cell
            ramp = "amber" if count != counts[p] else "gray"
            r += rect(x + 1, y, cell - 2, 18, ramp, extra="" if count or ramp == "amber" else ' fill-opacity="0.3"')
            r += label_in(x + 1, y, cell - 2, 18, ramp, str(count), "tx", mono=True)
            for k, owner in enumerate(owners[p]):   # one colour per owner, under the count
                w = (cell - 2) / len(owners[p])
                r += fill_rect(x + 1 + k * w, y + 19, w, 3, ACCENT[OWNER_RAMP[owner]])
        for k, p in enumerate(pool.free):
            ramp = "amber" if k >= len(free) or free[k] != p else "gray"
            r += rect(fx + k * slot, y, slot - 2, 18, ramp) + label_in(fx + k * slot, y, slot - 2, 18, ramp, str(p), "tx", True)
        for k in range(len(pool.free), len(free)):   # popped by this call: drawn faint where it was
            r += rect(fx + k * slot, y, slot - 2, 18, "amber", extra=' fill-opacity="0.25" stroke-dasharray="3 2"')
        return r

    b += draw(top, "", "PagePool(8)", pool.ref_counts, pool.free)
    for r, (who, op, *args) in enumerate(POOL_CALLS, start=1):
        counts, free = list(pool.ref_counts), list(pool.free)
        if op == "Allocate":
            page = pool.allocate()
            owners[page].append(who)
            call = f"Allocate() → {page}"
        elif op == "Retain":
            page = args[0]
            pool.retain(page)
            owners[page].append(who)
            call = f"Retain({page})"
        else:
            page = args[0]
            pool.release(page)
            owners[page].remove(who)
            call = f"Release({page})"
        b += draw(top + r * row, who, call, counts, free)

    y = top + (len(POOL_CALLS) + 1) * row + 6
    b += text(24, y, "Amber: changed by the call; dashed: the page Allocate popped from free_.", "tx")
    y += 16
    b += text(24, y, "The coloured bar under a count shows which conversations own the page; PagePool stores only the count.", "tx")
    f = "engine/page_pool.cpp"
    notes = [
        (lines(f, "ref_counts_.assign(page_count, 0);", "for (PageId page = page_count - 1; page >= 0; --page) {"),
         "every count 0; free_ filled from the highest page down, so page 0 is on top"),
        (lines(f, "if (free_.empty()) {", "ref_counts_[page] = 1;"),
         "Allocate: pop the top of free_ and give it one owner (nullopt when free_ is empty)"),
        (lines(f, "++ref_counts_[page];"), "Retain: one more owner of a page that already has one"),
        (lines(f, 'RequireOwned(page, "Release");', "        free_.push_back(page);\n    }\n}\n\nint"),
         "Release: one owner fewer; at 0 the page goes on top of free_"),
    ]
    b += notes_block(y + 22, notes)
    write("code_page_pool.svg", y + 22 + len(notes) * 17 + 4, "PagePool calls step by step", b)


# --- PrefixCache -------------------------------------------------------------

def short_key(key):
    return f"{key:016x}"[:6] + "…"


def token_strip(x, y, pieces, ramp):
    """One page's tokens as chips sized to their text."""
    b = ""
    for piece in pieces:
        s = shown(piece)
        w = len(s) * 5.6 + 4
        b += rect(x, y, w, 18, ramp) + text(x + w / 2, y + 9, s, f"tx k-{ramp} mono", "middle", ' style="font-size:10px"')
        x += w + 1
    return b


def key_chip(x, y, key, ramp, w=64):
    """A 64-bit key, its first six hex digits."""
    return rect(x, y, w, 18, ramp) + label_in(x, y, w, 18, ramp, short_key(key), "tx", mono=True)


def cache_row(x0, y, step, pages, refs, changed_pages, changed_refs, ramps=None):
    """A row of page cells (recency_ order, or Match's result) with each
    page's ref count under it."""
    b = ""
    for i, page in enumerate(pages):
        x = x0 + i * step
        ramp = (ramps or {}).get(page) or ("amber" if page in changed_pages else "teal")
        b += rect(x, y, step - 2, 18, ramp) + label_in(x, y, step - 2, 18, ramp, str(page), "tx", True)
        if refs is not None:
            b += amber_text(x + (step - 2) / 2, y + 27, str(refs[page]), page in changed_refs)
    return b


def prefix_cache():
    """PrefixCache on two market NPCs' system messages (16-token pages):
    how PageKey chains each page's key from its parent's, then Insert of
    Mei's pages, Match and Insert of Kenji's (which share the lore), Kenji's
    release, and an EvictOne that skips every page a sequence still uses."""
    prompts = market_tokens()
    mei, kenji = prompts["Mei"], prompts["Kenji"]
    manager = KvCacheManager(32)
    pool, cache = manager.pool, manager.prefix_cache
    mei_kv, kenji_kv = KvSequence(), KvSequence()
    manager.reserve(mei_kv, len(mei["ids"]))

    b = text(24, 22, f"PrefixCache: Mei's and Kenji's system messages ({len(mei['ids'])} and {len(kenji['ids'])} tokens), "
                     f"{PAGE_SIZE} tokens a page", "th")

    # PageKey's chain runs down the left: each key is made from the key above
    # it and the page's tokens beside the arrow.
    y = 50
    b += text(24, y, "PageKey(parent, page_tokens) for Mei's first two pages, and for page 1's tokens with kRootKey as parent", "ts")
    ids, pieces = mei["ids"], mei["pieces"]
    keys = [ROOT_KEY]
    for i in range(2):
        keys.append(page_key(keys[-1], ids[i * PAGE_SIZE:(i + 1) * PAGE_SIZE]))
    alone = page_key(ROOT_KEY, ids[PAGE_SIZE:2 * PAGE_SIZE])

    def link(y_, page, key, ramp, label):
        """An arrow from the chip at y_ down to `key`'s chip, with the page's
        16 tokens joining it from the right."""
        r = arrow(56, y_ + 18, 56, y_ + 43) + line(56, y_ + 31, 100, y_ + 31)
        r += token_strip(100, y_ + 22, pieces[page * PAGE_SIZE:(page + 1) * PAGE_SIZE], "gray")
        return r + key_chip(24, y_ + 44, key, ramp) + text(96, y_ + 53, label, "tx")

    y += 14
    b += key_chip(24, y, ROOT_KEY, "gray") + text(96, y + 9, "kRootKey", "tx mono")
    b += link(y, 0, keys[1], "teal", "page 0's key") + link(y + 44, 1, keys[2], "teal", "page 1's key: its parent is page 0's key")
    y += 120
    b += key_chip(24, y, ROOT_KEY, "gray") + text(96, y + 9, "kRootKey", "tx mono")
    b += link(y, 1, alone, "coral", "the same tokens under another parent: another key, so Match never finds them here")

    # Insert, Match, Insert, Release, EvictOne: recency_ after each call
    x0, step = 92, 23

    def header(y_, call, what):
        return text(24, y_, call, "ts mono") + text(24 + len(call) * 7 + 10, y_, what, "tx")

    def recency_row(y_, pages, changed_pages, refs_before, ramps=None):
        changed = {p for p, n in enumerate(pool.ref_counts) if n != refs_before[p]}
        return text(24, y_ + 9, "recency_", "tx mono") + cache_row(x0, y_, step, pages, pool.ref_counts, changed_pages, changed, ramps)

    y += 80
    refs = list(pool.ref_counts)
    cache.insert(mei["ids"], mei_kv.pages)
    lru = cache.recency_pages()
    b += header(y, "Insert(mei, mei.pages)", f"{len(lru)} full pages, all new: Retain (1 → 2 owners), key appended to recency_")
    b += recency_row(y + 16, lru, set(lru), refs)

    y += 61
    refs = list(pool.ref_counts)
    max_pages = (len(kenji["ids"]) - 1) // PAGE_SIZE
    kenji_kv.pages = cache.match(kenji["ids"], max_pages)
    hits = len(kenji_kv.pages)
    lru = cache.recency_pages()
    b += header(y, f"Match(kenji, {max_pages})",
                f"pages 0–{hits - 1} found: Retain, move to the back; page {hits} differs from Mei's: break")
    b += text(24, y + 25, "pages", "tx mono") + cache_row(x0, y + 16, step, kenji_kv.pages, None, set(), set())
    miss = x0 + hits * step
    b += rect(miss, y + 16, step - 2, 18, "coral") + label_in(miss, y + 16, step - 2, 18, "coral", "✗", "tx")
    b += recency_row(y + 42, lru, set(kenji_kv.pages), refs)

    y += 87
    manager.reserve(kenji_kv, len(kenji["ids"]))
    known, refs = set(lru), list(pool.ref_counts)
    cache.insert(kenji["ids"], kenji_kv.pages)
    lru = cache.recency_pages()
    added = [p for p in lru if p not in known]
    b += header(y, "Insert(kenji, kenji.pages)",
                f"pages 0–{hits - 1} found: moved to the back; Kenji's pages {added[0]} and {added[1]} are new")
    b += recency_row(y + 16, lru, set(added), refs)

    y += 61
    refs = list(pool.ref_counts)
    manager.release(kenji_kv)
    b += header(y, "Release(kenji)", "KvCacheManager::Release: each of Kenji's pages loses one owner")
    b += recency_row(y + 16, lru, set(), refs)

    y += 61
    refs = list(pool.ref_counts)
    cache.evict_one()
    evicted = next(p for p in lru if p not in cache.recency_pages())
    skipped = lru.index(evicted)
    b += header(y, "EvictOne()", f"skips {skipped} pages with 2 owners (a sequence and the cache); page {evicted} has 1: Release")
    b += recency_row(y + 16, lru, set(), refs, {p: "gray" for p in lru[:skipped]} | {evicted: "coral"})
    b += line(x0 + 4, y + 12, x0 + skipped * step + 6, y + 12, ACCENT["coral"], 1.2, arrow_end=True)

    y += 62
    b += text(24, y, "Teal: cached; amber: added or moved to the back by the call; gray: skipped; coral: miss or evicted. "
                     "Under a page: its ref count.", "tx")
    f = "engine/prefix_cache.cpp"
    notes = [
        (lines(f, "Key key = Mix(parent);", "key = Mix(key ^ static_cast<std::uint32_t>(token));"),
         "PageKey: mix the parent's key, then each of the page's 16 tokens"),
        (lines(f, "pool_->Retain(pages[i]);", "pages[i], std::prev(recency_.end())});"),
         "Insert, new page: Retain it, append its key to recency_, store parent, tokens and page in entries_"),
        (lines(f, "for (std::size_t i = 0; i < full_pages && static_cast<int>(pages.size()) < max_pages; ++i) {",
               "pages.push_back(entry->page);"), "Match: key each page from its parent's; stop at the first miss"),
        (lines(f, "for (auto it = recency_.begin(); it != recency_.end(); ++it) {", "pool_->Release(page);"),
         "EvictOne: the first page in recency_ with ref count 1 leaves the cache and is released"),
    ]
    b += notes_block(y + 22, notes)
    write("code_prefix_cache.svg", y + 22 + len(notes) * 17 + 4, "PrefixCache calls step by step", b)


# --- KvCacheManager ----------------------------------------------------------

KV_POOL = 26    # pages: Mei's 22 cached ones, Kenji's 3 new ones and one more, so the pool runs out
KV_STEP = 20    # one page's width: 16 positions at 1.25 px each


def kv_pages(x0, y, pages, length, shared, before=None):
    """A page table, one cell per page, filled as far as `length` (the
    positions whose keys and values are written). The first `shared` pages
    came from the prefix cache (teal), the rest from the pool (purple). Amber
    marks what changed since `before` = (pages, length)."""
    b = ""
    old_pages, old_length = before or (pages, length)
    w = KV_STEP - 1
    for i, page in enumerate(pages):
        x = x0 + i * KV_STEP
        if i >= len(old_pages):
            b += rect(x, y, w, 18, "amber") + label_in(x, y, w, 18, "amber", str(page), "tx", True)
            continue
        ramp = "teal" if i < shared else "purple"
        kept = min(max(old_length - i * PAGE_SIZE, 0), PAGE_SIZE) / PAGE_SIZE * w
        filled = min(max(length - i * PAGE_SIZE, 0), PAGE_SIZE) / PAGE_SIZE * w
        b += rect(x, y, w, 18, ramp, extra=' fill-opacity="0.3"')
        b += fill_rect(x, y, kept, 18, ACCENT[ramp], 0.5) + fill_rect(x + kept, y, filled - kept, 18, ACCENT["amber"], 0.7)
        b += label_in(x, y, w, 18, ramp, str(page), "tx", True)
    return b


def length_mark(x0, y, length, note=""):
    """A pointer under a page table at position `length`."""
    x = x0 + length * KV_STEP / PAGE_SIZE
    tip = f"M{x:.1f} {y:.1f} l-4 6 h8 z"
    return (f'<path d="{tip}" fill="{ACCENT["amber"]}"/>'
            + text(x + 7, y + 4, f"length {length}{note}", "tx"))


def kv_cache_manager():
    """KvCacheManager on Kenji's system message after Mei's conversation was
    released (her 22 full pages stay cached, the cache their only owner).
    AttachPrefix starts Kenji on the cached lore pages, and shows its
    max_pages rule on a prompt that ends on a page boundary. Reserve then
    grows the page table, reclaims a cached page once the pool is empty, and
    rolls back when it still cannot get a page. length changes only between
    the Reserve calls."""
    prompts = market_tokens()
    mei, kenji = prompts["Mei"], prompts["Kenji"]
    manager = KvCacheManager(KV_POOL)
    mei_kv, seq = KvSequence(), KvSequence()
    manager.reserve(mei_kv, len(mei["ids"]))
    mei_kv.length = len(mei["ids"])
    full = mei["ids"][:len(mei["ids"]) // PAGE_SIZE * PAGE_SIZE]   # Mei's full pages, as Engine::Publish passes them
    manager.publish(mei_kv, full)
    manager.release(mei_kv)
    x0, fx = 24, 588

    b = text(24, 22, f"KvCacheManager: Kenji's page table, {PAGE_SIZE} positions a page", "th")
    b += text(24, 44, f"A pool of {KV_POOL} pages here (about 1,450 in the game). Mei's conversation was released; "
                      f"her {len(full) // PAGE_SIZE} full pages stay cached.", "ts")

    # AttachPrefix on Kenji's prompt, and on Mei's full pages (on a copy, so the run below is unchanged).
    y = 72
    b += text(24, y, "AttachPrefix(sequence, tokens): pages = Match(tokens, max_pages), "
                     "max_pages = (tokens.size() − 1) / page_size", "ts")
    probe = KvSequence()
    copy.deepcopy(manager).attach_prefix(probe, full)
    manager.attach_prefix(seq, kenji["ids"])
    cases = [(kenji["ids"], seq, f"Match stops at page {len(seq.pages)}, the first that differs from Mei's"),
             (full, probe, f"every page is cached, but page {(len(full) - 1) // PAGE_SIZE} holds the last token")]
    for k, (tokens, sequence, why) in enumerate(cases):
        ry = y + 18 + k * 50
        n, size = len(sequence.pages), len(tokens)
        who = "Kenji" if sequence is seq else "Mei's first"
        b += text(24, ry, f"{who} {size} tokens: max_pages = {size - 1} / 16 = {(size - 1) // PAGE_SIZE} (coral); {why}", "tx")
        b += kv_pages(x0, ry + 10, sequence.pages, sequence.length, n)
        for i in range(n, pages_for(size)):   # the positions left to compute, dashed
            w = min(size - i * PAGE_SIZE, PAGE_SIZE) / PAGE_SIZE * (KV_STEP - 1)
            b += rect(x0 + i * KV_STEP, ry + 10, w, 18, "gray", extra=' fill-opacity="0.3" stroke-dasharray="3 2"')
        cap = x0 + (size - 1) // PAGE_SIZE * KV_STEP - 0.5
        b += line(cap, ry + 7, cap, ry + 29, ACCENT["coral"], 2)
        b += length_mark(x0, ry + 30, sequence.length, f"; {size - sequence.length} to compute")

    # Reserve: the page table at each call, and after the caller advances length.
    y += 130
    b += text(24, y, "kenji.pages and kenji.length at each call", "ts") + text(fx, y, "free_", "ts")
    shared = len(seq.pages)
    rows = []

    def call(label, note, change):
        """Runs `change` and records its row: the call, the state before and
        after it, the free_ stack before, and the cached pages it evicted."""
        before, free = (list(seq.pages), seq.length), list(manager.pool.free)
        cached = set(manager.prefix_cache.recency_pages())
        result = change()
        evicted = sorted(cached - set(manager.prefix_cache.recency_pages()))
        if result is not None:
            label += f" → {str(result).lower()}"
        rows.append((label, note, before, (list(seq.pages), seq.length), free, list(manager.pool.free), evicted, result))

    def set_length(length):
        seq.length = length

    n = len(kenji["ids"])
    call(f"Reserve(kenji, {n})", "Allocate() one page at a time; length is not changed",
         lambda: manager.reserve(seq, n))
    call(f"kenji.length = {n}", "the caller, after the forward pass has written those positions",
         lambda: set_length(n))
    seq.length = len(seq.pages) * PAGE_SIZE   # decoding, one position per step, fills the last page
    call(f"Reserve(kenji, {seq.length + 1})", "decoding: this position needs a new page, the last free one",
         lambda: manager.reserve(seq, seq.length + 1))
    seq.length = len(seq.pages) * PAGE_SIZE
    call(f"Reserve(kenji, {seq.length + 1})", "free_ is empty: EvictOne() frees cached page {evicted}; Allocate() takes it",
         lambda: manager.reserve(seq, seq.length + 1))
    seq.length += 8
    call(f"Reserve(kenji, {seq.length + 33})", "the next message, 33 positions, needs 2 more pages",
         lambda: manager.reserve(seq, seq.length + 33))

    ry = y + 18
    for label, note, before, (pages, length), free, free_after, evicted, result in rows:
        gone = ", ".join(map(str, evicted))
        b += text(24, ry, label, "ts mono") + text(24 + len(label) * 7 + 12, ry, note.format(evicted=gone), "tx")
        b += kv_pages(x0, ry + 10, pages, length, shared, before)
        b += length_mark(x0, ry + 30, length)
        for i, page in enumerate(free_after):
            ramp = "amber" if i >= len(free) or free[i] != page else "gray"
            b += rect(fx + i * 17, ry + 10, 16, 18, ramp) + label_in(fx + i * 17, ry + 10, 16, 18, ramp, str(page), "tx", True)
        for i in range(len(free_after), len(free)):   # popped by this call: drawn faint where it was
            b += rect(fx + i * 17, ry + 10, 16, 18, "amber", extra=' fill-opacity="0.25" stroke-dasharray="3 2"')
        if result is False:   # the page this call took and gave back, then the one it could not get
            x = x0 + len(pages) * KV_STEP
            b += rect(x, ry + 10, KV_STEP - 1, 18, "coral", extra=' fill-opacity="0.3" stroke-dasharray="3 2"')
            b += label_in(x, ry + 10, KV_STEP - 1, 18, "coral", gone, "tx", True)
            b += rect(x + KV_STEP, ry + 10, KV_STEP - 1, 18, "coral") + label_in(x + KV_STEP, ry + 10, KV_STEP - 1, 18, "coral", "✗", "tx")
            b += text(24, ry + 46, f"EvictOne() frees page {gone} and Allocate() takes it; for the second page no cached page "
                                   f"has 1 owner left, so Release({gone}) and return false.", "tx")
            ry += 16
        ry += 46

    y = ry + 2
    b += text(24, y, "Teal: shared pages from the prefix cache; purple: Kenji's own; filled part: positions below length; "
                     "amber: changed by that row.", "tx")
    f = "engine/kv_cache_manager.cpp"
    notes = [
        (lines(f, "const int max_pages = static_cast<int>(tokens.size() - 1) / page_size_;",
               "sequence.length = static_cast<int>(sequence.pages.size()) * page_size_;"),
         "AttachPrefix: at most (size − 1) / page_size pages from Match (dashed coral line); length = their positions"),
        (lines(f, "while (static_cast<int>(sequence.pages.size()) < PagesFor(length, page_size_)) {",
               "sequence.pages.push_back(*page);"),
         "Reserve: Allocate; if the pool is empty, EvictOne and Allocate once more"),
        (lines(f, "while (sequence.pages.size() > had) {", "return false;"),
         "still no page: release every page this call added, return false"),
        (lines("engine/engine.cpp", "c.kv.length += plan[i].count;"),
         "engine.cpp: the caller advances length after the forward pass"),
    ]
    b += notes_block(y + 22, notes)
    write("code_kv_cache_manager.svg", y + 22 + len(notes) * 17 + 4, "KvCacheManager calls step by step", b)


DIAGRAMS = [encode_piece, page_pool, prefix_cache, kv_cache_manager]

if __name__ == "__main__":
    for make in DIAGRAMS:
        make()
    print(f"{len(DIAGRAMS)} code pictures written")
