"""The shared workload (workload.json) through vLLM's offline engine, measured
the way `murmur-lab compare` measures Murmur:
  - time to first token: one request at a time, max_tokens 1 (median of 5);
  - decode: one request at a time, 256 tokens; (time - time to first token)
    / (tokens - 1) (median of 5);
  - throughput: 1, 8, 32 and 64 requests at once, generated tokens over the
    wall time.
Prefix caching is off, so no run reuses another's prompt (every request
also starts with its own tag).

Run inside a Python environment that has vLLM (Linux or WSL), from the
repository root. Under WSL2, vLLM also needs VLLM_WSL2_ENABLE_PIN_MEMORY=1
(and on some setups VLLM_USE_FLASHINFER_SAMPLER=0 and --eager):
  python bench/compare/run_vllm.py --format bf16 --out bench/compare/results/vllm_bf16.json
  python bench/compare/run_vllm.py --format fp8  --out bench/compare/results/vllm_fp8.json
"""
import argparse
import json
import os
import statistics
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODEL = os.path.join(ROOT, "models", "qwen2.5-1.5b-instruct")
REPEATS = 5
BATCH_SIZES = [1, 8, 32, 64]


def conversation(request):
    return [{"role": "system", "content": request["system"]}, {"role": "user", "content": request["user"]}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--out", required=True)
    parser.add_argument("--eager", action="store_true", help="no CUDA graphs (needed on some WSL setups)")
    parser.add_argument("--workload", default=os.path.join(HERE, "workload.json"))
    args = parser.parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    workload = json.load(open(args.workload, encoding="utf-8"))
    requests = workload["requests"]
    max_tokens = workload["max_tokens"]
    llm = LLM(model=MODEL, dtype="bfloat16", quantization="fp8" if args.format == "fp8" else None,
              max_model_len=1024, gpu_memory_utilization=0.6, enable_prefix_caching=False, seed=0,
              enforce_eager=args.eager)

    def run(first, count, tokens):
        convs = [conversation(requests[(first + i) % len(requests)]) for i in range(count)]
        params = SamplingParams(temperature=0.0, max_tokens=tokens)
        start = time.perf_counter()
        outputs = llm.chat(convs, params, use_tqdm=False)
        seconds = time.perf_counter() - start
        return seconds, outputs

    run(0, 8, 32)   # warm-up
    ttft, decode, prompt_tokens = [], [], 0
    for r in range(REPEATS):
        seconds, outputs = run(r, 1, 1)
        ttft.append(seconds)
        prompt_tokens = len(outputs[0].prompt_token_ids)
    first_token = statistics.median(ttft)
    first_reply = ""
    for r in range(REPEATS):
        seconds, outputs = run(REPEATS + r, 1, max_tokens)
        first_reply = first_reply or outputs[0].outputs[0].text
        n = len(outputs[0].outputs[0].token_ids)
        decode.append((n - 1) / (seconds - first_token))
    print(f"vLLM {vllm.__version__}, {args.format}: time to first token {first_token * 1e3:.1f} ms "
          f"({prompt_tokens} prompt tokens), decode {statistics.median(decode):.0f} tokens/s")

    throughput = []
    for n in BATCH_SIZES:
        seconds, outputs = run(0, n, max_tokens)
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"{n:>9} {tokens / seconds:>12.0f} {seconds:>9.2f} {tokens:>14}")
        throughput.append({"requests": n, "tokens_per_second": tokens / seconds, "seconds": seconds,
                           "tokens": tokens, "prompt_tokens": sum(len(o.prompt_token_ids) for o in outputs)})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"engine": f"vLLM {vllm.__version__}", "format": args.format, "cuda_graphs": not args.eager, "ttft_ms": first_token * 1e3,
               "prompt_tokens": prompt_tokens, "decode_tokens_per_second": statistics.median(decode),
               "first_reply": first_reply, "throughput": throughput}, open(args.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
