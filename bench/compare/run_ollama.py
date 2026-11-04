"""The shared workload (workload.json) through llama.cpp, as served by
Ollama, measured the way `murmur-lab compare` measures Murmur:
  - time to first token: one request at a time, 1 token (median of 5);
  - decode: one request at a time, 256 tokens; (time - time to first token)
    / (tokens - 1) (median of 5);
  - throughput: 1, 8, 32 and 64 requests at once, generated tokens over the
    wall time.
Times are taken on the client, around each HTTP call (loopback, no
streaming). Every request starts with its own tag, so no prompt is reused.

Needs a server started with enough parallel slots, for example:
  OLLAMA_HOST=127.0.0.1:11500 OLLAMA_NUM_PARALLEL=64 OLLAMA_FLASH_ATTENTION=1 ollama serve
and a model made from to_gguf.py's file (Modelfile: FROM ./qwen2.5-1.5b-instruct-f16.gguf). Then:
  python bench/compare/run_ollama.py --model murmur-qwen-f16 --format f16 --out bench/compare/results/llamacpp_f16.json
"""
import argparse
import concurrent.futures
import json
import os
import statistics
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPEATS = 5
BATCH_SIZES = [1, 8, 32, 64]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:11500")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    workload = json.load(open(os.path.join(HERE, "workload.json"), encoding="utf-8"))
    requests = workload["requests"]
    max_tokens = workload["max_tokens"]

    def call(request, tokens):
        body = {"model": args.model, "stream": False, "keep_alive": "60m",
                "messages": [{"role": "system", "content": request["system"]},
                             {"role": "user", "content": request["user"]}],
                "options": {"temperature": 0, "seed": 0, "num_predict": tokens, "num_ctx": 1024}}
        data = json.dumps(body).encode("utf-8")
        http = urllib.request.Request(args.url + "/api/chat", data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(http, timeout=600) as response:
            return json.load(response)

    def run(first, count, tokens):
        batch = [requests[(first + i) % len(requests)] for i in range(count)]
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            replies = list(pool.map(lambda r: call(r, tokens), batch))
        return time.perf_counter() - start, replies

    version = json.load(urllib.request.urlopen(args.url + "/api/version"))["version"]
    run(0, 8, 32)   # warm-up: loads the model
    ttft, decode, prompt_tokens = [], [], 0
    for r in range(REPEATS):
        seconds, replies = run(r, 1, 1)
        ttft.append(seconds)
        prompt_tokens = replies[0]["prompt_eval_count"]
    first_token = statistics.median(ttft)
    first_reply = ""
    for r in range(REPEATS):
        seconds, replies = run(REPEATS + r, 1, max_tokens)
        decode.append((replies[0]["eval_count"] - 1) / (seconds - first_token))
        first_reply = first_reply or replies[0]["message"]["content"]
    print(f"llama.cpp (Ollama {version}), {args.format}: time to first token {first_token * 1e3:.1f} ms "
          f"({prompt_tokens} prompt tokens), decode {statistics.median(decode):.0f} tokens/s")

    throughput = []
    for n in BATCH_SIZES:
        seconds, replies = run(0, n, max_tokens)
        tokens = sum(r["eval_count"] for r in replies)
        print(f"{n:>9} {tokens / seconds:>12.0f} {seconds:>9.2f} {tokens:>14}")
        throughput.append({"requests": n, "tokens_per_second": tokens / seconds, "seconds": seconds, "tokens": tokens,
                           "prompt_tokens": sum(r["prompt_eval_count"] for r in replies)})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"engine": f"llama.cpp (Ollama {version})", "format": args.format, "ttft_ms": first_token * 1e3,
               "prompt_tokens": prompt_tokens, "decode_tokens_per_second": statistics.median(decode),
               "first_reply": first_reply, "throughput": throughput},
              open(args.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
