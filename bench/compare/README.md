# Murmur against vLLM and llama.cpp

The same workload through three engines, on the same model (Qwen2.5-1.5B-Instruct, the weights in `models/`) and the same GPU, measured the same way.

**Workload** ([workload.json](workload.json), written by `murmur-lab compare`): 64 requests, each a system prompt (its own tag, then the market's city lore; 377 tokens with the chat template) and a request for a long story about one of the market's NPCs. Greedy, up to 256 tokens. The tag at the start of every request keeps any engine from reusing another request's cached prompt.

**Measures**, all from the client's side of each engine's API:
- time to first token: one request at a time, 1 token (median of 5);
- decode: one request at a time, 256 tokens, (time − time to first token) / (tokens − 1) (median of 5);
- throughput: 1, 8, 32 and 64 requests at once, generated tokens over the wall time from the first request to the last token.

| Engine | How it runs | Runner |
| --- | --- | --- |
| Murmur | `murmur-lab compare --weights bf16\|fp8\|int4 --json ...` (Windows) | [compare_benchmark.cpp](../../src/lab/compare_benchmark.cpp) |
| vLLM | offline `LLM.chat`, prefix caching off, CUDA graphs on (WSL2; `VLLM_WSL2_ENABLE_PIN_MEMORY=1`) | [run_vllm.py](run_vllm.py) |
| llama.cpp | through Ollama's `/api/chat` with 64 parallel slots and flash attention (Windows) | [run_ollama.py](run_ollama.py) |

llama.cpp reads GGUF files; [to_gguf.py](to_gguf.py) writes them from the same safetensors (F16 and Q8_0) with llama.cpp's own `gguf` library, following its Qwen2 converter. The six 16-bit and 8-bit runs agree word for word up to the point where the reference model itself is torn between two tokens ([results/near_tie.json](results/near_tie.json): " memory" 19.6442 against " augment" 19.6375); Murmur's INT4 departs earlier.

Results are in [results/](results); the write-up is [docs/comparison.md](../../docs/comparison.md).
