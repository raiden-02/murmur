"""Writes Qwen2.5-1.5B-Instruct (the safetensors in models/) as a GGUF file
for llama.cpp, in F16 or Q8_0, so llama.cpp runs the very weights Murmur and
vLLM run. Follows llama.cpp's own converter for Qwen2: the same tensor names,
the same metadata, the tokenizer as a GPT-2-style byte-level BPE with the
"qwen2" pre-tokenizer, vectors (norms, biases) kept in F32, and the
vocabulary padded to the model's 151,936 rows with unused tokens.

Needs numpy, torch, safetensors and the `gguf` package (llama.cpp's Python
library). Run from the repository root:
  python bench/compare/to_gguf.py --type f16 --out out/ollama/qwen2.5-1.5b-instruct-f16.gguf
  python bench/compare/to_gguf.py --type q8_0 --out out/ollama/qwen2.5-1.5b-instruct-q8_0.gguf
"""
import argparse
import json
import os

import gguf
import numpy as np
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODEL = os.path.join(ROOT, "models", "qwen2.5-1.5b-instruct")

# Hugging Face name (after "model.layers.N.") -> llama.cpp name (after "blk.N.")
LAYER_NAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.weight": "attn_k.weight", "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.o_proj.weight": "attn_output.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


def gguf_name(name):
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"
    parts = name.split(".")   # model.layers.N.rest
    return f"blk.{parts[2]}.{LAYER_NAMES['.'.join(parts[3:])]}"


def add_tokenizer(writer, vocab_size):
    tokenizer = json.load(open(os.path.join(MODEL, "tokenizer.json"), encoding="utf-8"))
    config = json.load(open(os.path.join(MODEL, "tokenizer_config.json"), encoding="utf-8"))
    by_id = {i: t for t, i in tokenizer["model"]["vocab"].items()}
    added = {t["id"]: t["content"] for t in tokenizer["added_tokens"]}
    tokens, types = [], []
    for i in range(vocab_size):
        if i in added:
            tokens.append(added[i])
            types.append(gguf.TokenType.CONTROL)
        elif i in by_id:
            tokens.append(by_id[i])
            types.append(gguf.TokenType.NORMAL)
        else:
            tokens.append(f"[PAD{i}]")
            types.append(gguf.TokenType.UNUSED)
    merges = [m if isinstance(m, str) else " ".join(m) for m in tokenizer["model"]["merges"]]
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    ids = {t: i for i, t in added.items()}
    writer.add_eos_token_id(ids[config["eos_token"]])
    writer.add_pad_token_id(ids[config["pad_token"]])
    writer.add_bos_token_id(ids["<|endoftext|>"])
    writer.add_add_bos_token(False)
    writer.add_chat_template(config["chat_template"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["f16", "q8_0"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.load(open(os.path.join(MODEL, "config.json"), encoding="utf-8"))

    writer = gguf.GGUFWriter(args.out, "qwen2")
    writer.add_name("Qwen2.5-1.5B-Instruct")
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(config["hidden_size"])
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_feed_forward_length(config["intermediate_size"])
    writer.add_head_count(config["num_attention_heads"])
    writer.add_head_count_kv(config["num_key_value_heads"])
    writer.add_rope_freq_base(config["rope_theta"])
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if args.type == "f16" else gguf.LlamaFileType.MOSTLY_Q8_0)
    add_tokenizer(writer, config["vocab_size"])

    weights = load_file(os.path.join(MODEL, "model.safetensors"))
    for name, tensor in weights.items():
        data = tensor.float().numpy()
        if data.ndim == 1:
            writer.add_tensor(gguf_name(name), data.astype(np.float32))
        elif args.type == "f16":
            writer.add_tensor(gguf_name(name), data.astype(np.float16))
        else:
            q = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(gguf_name(name), q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out}: {os.path.getsize(args.out) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
