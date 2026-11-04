# NPCs that act: constrained, jump-forward and speculative decoding

Every NPC reply ends with an action the game carries out. This report measures how the engine holds the model to the action grammar, what jump-forward saves, and whether speculative decoding with a draft model pays off on this hardware. The code is walked through in [Part IX of the code tour](CODE_TOUR.md#part-ix--npcs-that-act).

RTX 4080 SUPER, Release build, Qwen2.5-1.5B-Instruct with FP8 weights and a BF16 KV cache unless noted. Other programs shared the GPU during some runs, so speeds carry a few percent of noise. Reproduce with `murmur-lab actions` and `murmur-lab speculative` (both take `--json`).

## The action grammar

The grammar is built from the game's own lists ([action_language.cpp](../src/game/action_language.cpp)): seven verbs, the ten storefronts, 22 items, the 32 NPCs, prices 1–999 and sign messages of up to 18 capitals.

| Stage | Size | Time |
| --- | --- | --- |
| Grammar text | 1,100 bytes, 12 rules | |
| Byte automaton (NFA → DFA) | 913 states | 1.9 ms |
| Token masks (per state: allowed tokens, next states, a GPU bitmask) | 101 allowed tokens per state on average; 29 of 151,665 at the start | 30 ms |
| States whose next bytes are fixed | 835 of 913 | |

## Valid actions

All 32 market NPCs answer the same 8 requests ("Can I buy something from you?", "Do you have a job for me?", "Go and see what's happening at the docks", ...): 256 actions, sampled with the NPCs' settings (temperature 0.7, top-k 20, top-p 0.8).

| | Parse | Model steps per action | Written by the grammar |
| --- | --- | --- | --- |
| Grammar + jump-forward | **256 / 256** | **2.5** | 2.5 tokens |
| Grammar, no jump-forward | 256 / 256 | 4.6 | 0 |
| Free text after the same `ACTION:` (spacing forgiven) | 215 / 256 (84%) | 5.0 | 0 |

Free text that doesn't parse: `sell(ITEM, PRICE)`, `sell(KNIFE, 500)`, `sell(SKINNY BAG, 500)`, `walk(POWERS)`, `sell(data shard, 100 credits)`, `walk(Ramen)`, `give(tea, 1)`.

Action mix: under the grammar walk 56, call 28, hack 27, give 8, sell 7, quest 1, none 129; as free text (of those that parse) walk 55, call 25, give 4, none 131. The two mixes agree, so the mask doesn't change which action the model picks.

## Token boundaries

The first version wrote the lead-in `ACTION: ` (ending in a space) and actions without a leading space. The model writes a word after a colon as one token with its space, so every token it preferred was masked:

| Framing | Allowed share of the model's odds at the action's start | Grammar's best token |
| --- | --- | --- |
| `ACTION: ` + `give(…)` | 0.02% | `none`, the model's 65th choice |
| `ACTION:` + ` give(…)` | 91% | ` give`, its first |

With the first framing the action mix was walk 22, quest 55 against free text's walk 55, so the mask was choosing the action for the model. Jump-forward needs the same care: it must not write the action's leading space as a lone token, so a fixed last token that an allowed token extends is left to the model.

Measured with Hugging Face Transformers on the same prompt ([extract_action_data.py](diagrams/extract_action_data.py)).

## Speculative decoding

Draft model Qwen2.5-0.5B-Instruct (24 layers, 896 wide, same tokenizer), checked against its own reference (every checkpoint within 1e-3). Eight NPC replies, one conversation at a time, greedy unless noted.

| Guesses per step | FP8 model, INT4 draft | Kept | Identical replies | BF16 model, FP8 draft | Kept | Identical | FP8, sampled NPCs | Kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 271 tokens/s | | | 152 tokens/s | | | 245 tokens/s | |
| 1 | 0.88× | 48% | 6 / 8 | 0.96× | 49% | 8 / 8 | 0.85× | 42% |
| 2 | 0.85× | 40% | 6 / 8 | **1.01×** | 40% | 8 / 8 | 0.75× | 29% |
| 4 | 0.67× | 26% | 6 / 8 | 0.90× | 26% | 8 / 8 | 0.57× | 17% |
| 8 | 0.54× | 16% | 6 / 8 | 0.66× | 15% | 8 / 8 | 0.40× | 9% |

- **Why it doesn't pay:** one draft step costs 1.97 ms against 3.58 ms for a plain step of the big model (FP8): 55%, where the parameter ratio would suggest 33%. Both models are small enough to be bound by launches and synchronization rather than bytes. With 48% of guesses kept, one guess yields 1.48 tokens per step for 1.97 + 4.13 ms: 4.1 ms per token against 3.6.
- **Exactness:** both models use the same random draw per position, so a guess is kept exactly when the big model would have chosen it. With BF16 weights every reply is identical to plain decoding. With FP8, 6 of 8 (sampled: 4 of 8): checking several rows at once goes through the tensor-core product, which rounds activations to FP16, while one row goes through the matrix-vector kernel, and near-ties can flip.
- **Policy:** the engine speculates only while at most 4 conversations decode, and the game starts with it off (G turns it on). In the game, with sampled NPCs and four guesses, 19% of guesses were kept.

## Scheduling under load

The 32-NPC actions test found two bugs:

1. `Engine` had no destructor, so the KV pages of conversations still open when an engine went away stayed referenced. The cache belongs to the `Transformer` and outlives the engine; every later engine had less memory, until none could make progress.
2. The eviction rule (a reply may evict an equally important reply only once it is itself generating) deadlocks when every reply is still waiting for its first token and the cache is full. Replies are now ordered by when they were asked, and a reply may evict only newer replies of its priority, newest first. `ConversationsThatFillTheCacheTogetherAllFinish` reproduces the deadlock under the old rule.

The 30-NPC market benchmark (60 s) after the change: no evictions, the player's first token at 24 ms p50 and 39 ms p99, every page free afterwards.

## In the game

- 32 NPCs act on their own (walk to a storefront, wave someone over, hack a sign) and towards the player (give, sell, hand over a delivery). An NPC rests 20 s between errands of its own and stands still while talking to the player.
- The player buys offers with B, carries deliveries for credits, and each NPC hears what the player did as a stage direction in front of the next message.
- Words are masked too: no line breaks and no `(` while an NPC speaks, so actions can't leak into speech.
- The X-ray counts every action, shows the share of action tokens the grammar wrote (about half), and how many of the draft model's guesses were kept.
