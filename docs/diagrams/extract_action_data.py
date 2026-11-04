"""Real data for docs/actions_and_speculation.md (the token-boundary table)
and the code tour's action diagram, written to data/action_logits.json: Mei
answers the player (greedy, with the market's system prompt), then the
model's next-token odds right after "ACTION:" (and after "ACTION: ", the
first framing, whose actions had no leading space), and again once it has
picked its action and the grammar has written "(".
Each top token is marked with whether the grammar allows it there, and the
grammar's best token is found among the whole vocabulary, with its rank.

Needs data/actions.json (`murmur-lab actions --json docs/diagrams/data/actions.json`)
for the exact system prompt and grammar. Run with the reference
environment, from the repository root:
  reference/.venv/Scripts/python.exe docs/diagrams/extract_action_data.py
"""
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODEL = os.path.join(ROOT, "models", "qwen2.5-1.5b-instruct")
DATA = os.path.join(HERE, "data")
QUESTION = "Can I buy something from you?"
TOP = 10


def alternatives(grammar, rule):
    """The quoted alternatives of a `rule ::= "a" | "b"` line."""
    line = next(l for l in grammar.splitlines() if l.startswith(rule + " "))
    return re.findall(r'"([^"]*)"', line.split("::=", 1)[1])


def odds(model, ids, vocab, allowed):
    """After `ids`: the model's top tokens with their probabilities and
    whether the grammar allows each, then the best allowed token of the
    whole vocabulary with its rank."""
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor([ids])).logits[0, -1].float(), dim=-1)
    order = torch.argsort(probs, descending=True).tolist()
    top = [{"token": vocab[i], "p": probs[i].item(), "allowed": allowed(vocab[i])} for i in order[:TOP]]
    rank, best = next((r, i) for r, i in enumerate(order) if i < len(vocab) and allowed(vocab[i]))
    allowed_mass = sum(probs[i].item() for i in range(len(vocab)) if allowed(vocab[i]))
    return {"top": top, "best_allowed": {"token": vocab[best], "p": probs[best].item(), "rank": rank + 1},
            "allowed_mass": allowed_mass}


def main():
    actions = json.load(open(os.path.join(DATA, "actions.json"), encoding="utf-8"))
    grammar = actions["grammar"]
    openers = [" " + v + "(" for v in ("walk", "give", "sell", "hack", "quest", "call")] + [" none"]
    items, places, npcs = alternatives(grammar, "item"), alternatives(grammar, "place"), alternatives(grammar, "npc")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    special = set(tokenizer.all_special_tokens) | set(tokenizer.added_tokens_encoder)
    vocab = [tokenizer.decode([i]) for i in range(len(tokenizer))]
    vocab = [t if t not in special else "\0special" for t in vocab]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    messages = [{"role": "system", "content": actions["system_prompt_mei"]}, {"role": "user", "content": QUESTION}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt]), max_new_tokens=60, do_sample=False, eos_token_id=end)
    reply_ids = out[0, len(prompt):].tolist()
    reply_ids = reply_ids[: reply_ids.index(end)] if end in reply_ids else reply_ids
    reply = tokenizer.decode(reply_ids).split("\n")[0]

    def starts_sentence(t):
        return bool(t) and any(o.startswith(t) for o in openers)

    lead = tokenizer.encode(reply + "\nACTION:", add_special_tokens=False)
    at_start = odds(model, prompt + lead, vocab, starts_sentence)
    # The first framing: a lead-in ending in a space, and actions without one.
    lead_space = tokenizer.encode(reply + "\nACTION: ", add_special_tokens=False)
    at_start_space = odds(model, prompt + lead_space, vocab,
                          lambda t: bool(t) and any(o[1:].startswith(t) for o in openers))
    # The grammar's best opener, then what the grammar writes up to "(".
    verb = next(o for o in openers if o.startswith(at_start["best_allowed"]["token"]))
    fills = {" walk(": places, " hack(": places, " call(": npcs}.get(verb, items)
    after = tokenizer.encode(reply + "\nACTION:" + verb, add_special_tokens=False)
    at_argument = odds(model, prompt + after, vocab, lambda t: bool(t) and any(f.startswith(t) for f in fills))

    # Every token that fits at the start of an action: its bytes begin a sentence of the grammar.
    fits = sorted({t for t in vocab if starts_sentence(t)},
                  key=lambda t: (openers.index(next(o for o in openers if o.startswith(t))), len(t)))
    json.dump({"question": QUESTION, "reply": reply, "verb": verb, "at_start": at_start,
               "at_start_after_space": at_start_space, "at_argument": at_argument,
               "openers": openers, "fits_at_start": fits},
              open(os.path.join(DATA, "action_logits.json"), "w", encoding="utf-8"), indent=1)
    print(reply, "|", verb, "|", len(fits), "tokens fit at the start")
    for name, o in (("start", at_start), ("start after a space", at_start_space), ("argument", at_argument)):
        print(name, "best allowed", o["best_allowed"], "allowed mass", round(o["allowed_mass"], 4))
        for r in o["top"][:6]:
            print(f"  {r['token']!r:>16} {r['p']:.3f} {'ok' if r['allowed'] else '--'}")


if __name__ == "__main__":
    main()
