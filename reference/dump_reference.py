"""Dump reference tensors from the Hugging Face implementation of the model.

For each fixed prompt this writes, into reference/out/:
  <name>.safetensors  every intermediate tensor of the prompt's forward pass (FP32)
  <name>.json         prompt text, token ids, and the greedy continuation
plus tokenizer_cases.json: tricky strings and their exact token ids.
With --draft it dumps the 0.5B draft model's prompts instead, as draft_<name>.

The engine's tests load these files and compare checkpoint by checkpoint.

Usage (from any directory):
  reference/.venv/Scripts/python.exe -I -X utf8 reference/dump_reference.py [--draft]
"""

import argparse
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "qwen2.5-1.5b-instruct"
DRAFT_DIR = REPO_ROOT / "models" / "qwen2.5-0.5b-instruct"   # the speculative-decoding draft
OUT_DIR = REPO_ROOT / "reference" / "out"
GREEDY_TOKENS = 32

PROMPTS = {
    "plain": {"text": "The capital of France is"},
    "noodle_vendor": {
        "chat": [
            {"role": "system", "content": "You are Mei, a noodle vendor in a neon-lit night market. "
                                          "Answer in one or two short sentences."},
            {"role": "user", "content": "What's good tonight?"},
        ]
    },
}

# One case per branch of the pre-tokenizer regex, plus edge cases around them.
TOKENIZER_CASES = [
    "Hello, world!",
    "  leading spaces and   runs of spaces",
    "trailing spaces   ",
    "numbers 12345 and 3.14159",
    "line one\nline two\n\nline four",
    "windows\r\nline endings\r\n",
    "tabs\tand\t\tmore  \n  indented",
    "contractions: I'm, you're, they'll, we've, she'd, it's, don't",
    "SHOUTING: I'M, YOU'RE, THEY'LL, 'S",
    "an 'sun' and 'Tis",
    "punctuation!!! ...?? (brackets) [x] {y} <z> --- ***",
    "unicode: café, naïve, 東京, Ελληνικά, Привет",
    "emoji 🍜🌃 and symbols ¥€$ ½ ²",
    "<|im_start|>user\nhi<|im_end|>",
    "",
]

TOKENIZER_CORPUS_FILES = ["README.md", "LICENSE"]

Store = dict[str, torch.Tensor]


def _as_row_major_fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Drop the batch dimension and return an owned, contiguous FP32 copy."""
    return tensor.detach().float().squeeze(0).contiguous().clone()


def _output_hook(store: Store, name: str) -> Callable:
    def hook(_module, _inputs, output):
        store[name] = _as_row_major_fp32(output[0] if isinstance(output, tuple) else output)
    return hook


def _input_hook(store: Store, name: str) -> Callable:
    def hook(_module, inputs):
        store[name] = _as_row_major_fp32(inputs[0])
    return hook


@contextmanager
def captured_checkpoints(model: torch.nn.Module) -> Iterator[Store]:
    """Record every intermediate the engine reimplements, keyed by a stable name.

    Hooks are removed on exit, even if the forward pass raises.
    """
    store: Store = {}
    m = model.model
    handles = [m.embed_tokens.register_forward_hook(_output_hook(store, "embed"))]
    for i, layer in enumerate(m.layers):
        p = f"layers.{i}."
        attn, mlp = layer.self_attn, layer.mlp
        handles += [
            layer.input_layernorm.register_forward_hook(_output_hook(store, p + "attn_norm")),
            attn.q_proj.register_forward_hook(_output_hook(store, p + "q")),
            attn.k_proj.register_forward_hook(_output_hook(store, p + "k")),
            attn.v_proj.register_forward_hook(_output_hook(store, p + "v")),
            # Attention's weighted mix of values, before the output projection.
            attn.o_proj.register_forward_pre_hook(_input_hook(store, p + "attn_context")),
            attn.register_forward_hook(_output_hook(store, p + "attn_out")),
            layer.post_attention_layernorm.register_forward_hook(_output_hook(store, p + "mlp_norm")),
            # SiLU(gate) * up, before the down projection.
            mlp.down_proj.register_forward_pre_hook(_input_hook(store, p + "mlp_act")),
            mlp.register_forward_hook(_output_hook(store, p + "mlp_out")),
            layer.register_forward_hook(_output_hook(store, p + "out")),
        ]
    handles.append(m.norm.register_forward_hook(_output_hook(store, "final_norm")))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


def encode_prompt(tokenizer: PreTrainedTokenizerBase, prompt: dict) -> tuple[str, BatchEncoding]:
    """Apply the chat template when the prompt is a conversation, then tokenize."""
    if "chat" in prompt:
        text = tokenizer.apply_chat_template(prompt["chat"], tokenize=False, add_generation_prompt=True)
    else:
        text = prompt["text"]
    return text, tokenizer(text, return_tensors="pt")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def dump_prompt(model, tokenizer: PreTrainedTokenizerBase, name: str, prompt: dict) -> None:
    text, encoded = encode_prompt(tokenizer, prompt)
    ids = encoded.input_ids

    with captured_checkpoints(model) as store:
        logits = model(**encoded).logits
    store["logits"] = _as_row_major_fp32(logits)
    store["input_ids"] = ids.squeeze(0).to(torch.int32).contiguous()
    save_file(store, OUT_DIR / f"{name}.safetensors")

    # Pure greedy: the model's generation_config would otherwise add a repetition penalty.
    generated = model.generate(**encoded, max_new_tokens=GREEDY_TOKENS, do_sample=False,
                               temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
    new_ids = generated[0, ids.shape[1]:].tolist()
    reply = tokenizer.decode(new_ids)
    write_json(OUT_DIR / f"{name}.json", {
        "prompt_text": text,
        "input_ids": ids[0].tolist(),
        "greedy_ids": new_ids,
        "greedy_text": reply,
    })
    print(f"{name}: {ids.shape[1]} prompt tokens, {len(store)} tensors -> {reply!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump reference tensors from Hugging Face.")
    parser.add_argument("--draft", action="store_true", help="the 0.5B draft model, written as draft_<name>")
    draft = parser.parse_args().draft
    model_dir = DRAFT_DIR if draft else MODEL_DIR

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)   # the draft shares it
    # FP32 on the CPU: the reference must be the most precise version, not the fastest.
    # Eager attention keeps the math in textbook order (no fused kernels).
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()

    for name, prompt in PROMPTS.items():
        dump_prompt(model, tokenizer, ("draft_" if draft else "") + name, prompt)
    if draft:
        return   # the draft shares the main model's tokenizer

    corpus = [(MODEL_DIR / name).read_text(encoding="utf-8") for name in TOKENIZER_CORPUS_FILES]
    cases = [{"text": text, "ids": tokenizer(text).input_ids} for text in TOKENIZER_CASES + corpus]
    write_json(OUT_DIR / "tokenizer_cases.json", cases)
    print(f"tokenizer cases: {len(cases)}")


if __name__ == "__main__":
    main()
