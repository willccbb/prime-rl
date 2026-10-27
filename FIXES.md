# gpt-oss fixes

`swarm-rl/dev` is prime-rl `main` with each fix below merged in from its own topic branch, for running gpt-oss-20b (native MXFP4) with LoRA on one trainer GPU and one vLLM GPU, on SM120 (RTX Pro 6000 Blackwell) as well as datacenter GPUs. Its submodules pin the `swarm-rl/dev` branches of verifiers and renderers.

To catch up with a newer `main`, rebase or re-cherry-pick each topic branch onto it, rebuild `swarm-rl/dev` from `main` and the topic branches, re-lock with the latest uv, and run the tests named under each fix. Retire a topic branch once `main`, or the upstream project it works around, carries an equivalent fix.

## Fixes

### FA4 98c1cee and its SM80-family backward
Branch: `fa4-98c1cee-sm80-backward`

- **Broke:** FA4 rev `0251105` read out of bounds in its SM80/SM120 varlen kernels (compute-sanitizer memcheck), and gpt-oss LoRA RL on RTX Pro 6000 showed occasional 1e3–1e4x gradient-norm spikes. FA4 `98c1cee` fixes the reads, but its SM80-family backward offsets the dK/dV accumulator by `offset_k + batch_idx * n_block_size` where its postprocess reads `padded_offset_k`, so dK/dV are wrong for GQA with packed sequences whose boundaries aren't tile-aligned.
- **Fix:** bump FA4 to `98c1cee` and install its own SM80 epilogue with `padded_offset_k`. The patch raises if upstream's epilogue changes, and its import sits outside FA4's ImportError guard, so a moved FA4 module fails at import.
- **Evidence:** memcheck on `0251105`; `test_flash_attn_4_varlen_op_matches_eager_reference` fails without the epilogue fix.
- **Tests:** `tests/unit/train/models/test_flash_attn_4.py` (GPU).
- **On main:** no. Not reported upstream to FA4.

### gpt-oss attention sinks on SM12x
Branch: `gpt-oss-sm12x-sinks`

- **Broke:** FA4's SM12x kernels have no learnable-sink support, so gpt-oss attention failed on RTX Pro 6000.
- **Fix:** on SM12x, run FA4 without the sink and fold it in as `out * sigmoid(lse - sink)`, which is exact. Ulysses context parallelism still passes the sink to FA4 and fails its assertion on SM12x, so context parallelism isn't supported there.
- **Tests:** `test_gpt_oss_attention_matches_eager_reference` in `tests/unit/train/models/test_gpt_oss.py` (GPU). It exercises the varlen GQA backward that `fa4-98c1cee-sm80-backward` fixes, so on SM12x it needs that branch too. The context-parallel test there needs several GPUs.
- **On main:** no.

### Native MXFP4 experts, frozen under LoRA
Branch: `gpt-oss-mxfp4-lora`

- **Broke:** prime-rl could not load or train an MXFP4 checkpoint (`openai/gpt-oss-20b`).
- **Fix:** keep routed experts as packed uint8 blocks and e8m0 scales, dequantized per projection in forward and again in backward for the input gradient. Outputs and gradients are bitwise equal to the bf16 checkpoint, and experts take 10 GB instead of 76 GB of fp32 masters. The experts are frozen: training an MXFP4 checkpoint without LoRA raises, as does expert parallelism. Active-parameter counts include them.
- **Tests:** `tests/unit/train/models/test_mxfp4.py` (GPU) and the MXFP4 cases in `test_gpt_oss.py` (CPU). `test_moe.py` and `tests/unit/train/test_perf.py` cover the MoE, LoRA-expert and active-parameter code it touches.
- **On main:** no.

### vLLM loads MXFP4 checkpoints itself
Branch: `gpt-oss-mxfp4-vllm-loader`

- **Broke:** prime-rl's gpt-oss weight-loading patch checked expert shapes strictly and rejected MARLIN's padded expert biases, so vLLM failed to load `openai/gpt-oss-20b`.
- **Fix:** the patch defers to vLLM's loader when the quantization is `gpt_oss_mxfp4`.
- **Tests:** none; vLLM serving `openai/gpt-oss-20b` checks it.
- **On main:** no.

### FlashInfer plan() staging under async scheduling
Branch: `flashinfer-plan-staging`

- **Broke:** since vLLM 0.31 (#57214) there is no host/device barrier between FlashInfer `plan()` calls for models without speculative decoding. Under async scheduling, step N+1's plan overwrites the pinned staging buffer before step N's copy runs, and serving crashes with a CUDA illegal memory access when the GPU lags the host.
- **Fix:** stage `plan()` through a pageable buffer. Raise when vLLM first builds a FlashInfer wrapper (in practice at startup) if FlashInfer no longer has `_pin_memory_int_workspace_buffer`.
- **Evidence:** the crash reproduced under sustained gpt-oss serving with async scheduling and stopped with the patch. Upstream issue: vllm-project/vllm#61239 (open).
- **Tests:** none; serving under async scheduling checks it.
- **On main:** no. Drop it once vLLM fixes #61239.

### One harmony build
Branch: `oss-harmony-only`

- **Broke:** vLLM 0.32 requires `oss-harmony` and renderers required `openai-harmony`. Both install the `openai_harmony` module, and `openai-harmony` 0.0.8's native library overwrote `oss-harmony` 0.0.11's. vLLM's chat-completions parser then read recipient-first tool calls as names like `ipython<|channel|>commentary`.
- **Fix:** point `deps/renderers` at renderers `5dc6315` (topic branch `harmony-dedupe`), which depends on `oss-harmony` and renders the same tokens, so the lock holds only `oss-harmony`. `swarm-rl/dev` pins renderers `swarm-rl/dev`, which carries it.
- **Evidence:** the installed `.so` hash matched `openai-harmony` 0.0.8's RECORD; 26% of eval tool calls in a Prime Agent run had malformed names. On 0.0.11 both header orders parse to `functions.ipython`, and a training prompt re-renders to the same token ids.
- **Tests:** `uv lock --check`; only `oss_harmony` is installed.
- **On main:** no. `5dc6315` lives on renderers branch `harmony-dedupe`, whose PR (PrimeIntellect-ai/renderers#204) was closed unmerged; keep that branch until renderers depends on `oss-harmony`.

### In-training evals through the renderer
Branch: `eval-through-renderer`

- **Broke:** the orchestrator evaluated through vLLM's chat-completions endpoint while training sampled through the renderer, so evals and training parsed and templated the model's output on different stacks, and eval traces carried no token ids.
- **Fix:** the policy's eval client is a renderer client, like its train client.
- **Consequences:** eval sources take the same constraints as training: harnesses must speak chat completions, reasoning effort comes from `[orchestrator.renderer]`, not eval sampling, and `chat_template_kwargs` must be ones the renderer accepts. Eval requests go through `/inference/v1/generate`, so with router replay on, eval traces carry routed-expert data too. The standalone eval runner keeps chat completions, so its scores and in-training eval scores come from different stacks.
- **Tests:** none; a run's step-0 eval checks it (token ids on eval traces, no malformed tool names).
- **On main:** no. PrimeIntellect-ai/prime-rl#4161 (open) takes another route, putting the renderer behind the chat endpoint for online evals.

### LoRA on the gpt-oss unembedding
Branch: `gpt-oss-lm-head-lora` (cut from a newer `main` than this branch's base; cherry-picked here)

- **Broke:** vLLM applies LoRA to `lm_head` only for models that list it in `embedding_modules`, and `GptOssForCausalLM` lists none, so Tinker gpt-oss adapters, which also train the unembedding, failed to load ("expected target modules in {...} but received ['lm_head']").
- **Fix:** declare `lm_head` as gpt-oss's output embedding, as Llama and Qwen3 do. Base requests and adapters without an `lm_head` part get a zero delta.
- **Evidence:** a converted Tinker rank-32 adapter loads, and served logprobs match Tinker's sampler at mismatch KL about .0014 in both directions, against .12 for the base model, so the adapter's attention and expert parts are reproduced. On that checkpoint the unembedding part's effect is below the noise, so its correctness is not separately shown.
- **Limit:** vLLM's `prompt_logprobs` path maps the logits LoRA per request, not per token, so teacher-forced scoring leaves an adapter's unembedding part out, and by a reading of the code (not seen live) can add it to the first positions of another request's prompt logprobs in the same batch, base-model requests included. Sampling applies it correctly. Every gpt-oss LoRA server now runs logits through the LoRA path, about 2% more logits compute.
- **Tests:** none; loading a Tinker adapter on the eval server checks the load.
- **On main:** no.

## Carried elsewhere

- **verifiers `swarm-rl/dev`:** `prime-agent-local-release` (`env.agent.harness.release` installs a locally built Prime Agent release) and `prime-agent-save-session` (Prime Agent writes its session log again; prime-agent-basics scores from it), among others.
- **renderers `swarm-rl/dev`:** `harmony-dedupe`.
- **Prime Agent `swarm-rl/dev`:** releases built from Prime Agent `main` plus its own topic branches.
- **Recipe:** gpt-oss rollouts train as one sequence per context with `[orchestrator.renderer] auto_drop_analysis = false`.
