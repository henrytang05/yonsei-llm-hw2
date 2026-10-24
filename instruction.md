# Cell-by-Cell Instruction Guide — `assn2_stu_v2.ipynb`

Each code cell is described with: what it does, what it depends on, what it produces, and what (if anything) you are allowed or expected to edit.

Cells marked **🔴 EDIT HERE** are graded. Cells marked **🟢 DO NOT EDIT** must be left unchanged.

---

## Cell 00 — Assignment Overview (Markdown)

**What it is:** The assignment briefing — goal, grading rubric, pipeline overview, and dataset table.

**Edit:** 🟢 Do not edit.

---

## Cell 01 — Section Header: Environment Setup (Markdown)

Section label only. No code.

---

## Cell 02 — Imports and Path Definitions

**What it does:** Imports Python stdlib modules and `openrlhf`. Defines all global path variables and the base model ID used across every subsequent cell.

| Variable | Value |
|---|---|
| `PROJECT_ROOT` | Current working directory |
| `DATA_DIR` | `PROJECT_ROOT/data` |
| `CKPT_DIR` | `PROJECT_ROOT/checkpoints` |
| `MODEL_BASE_ID` | `meta-llama/Llama-3.2-1B-Instruct` (or `ASSN2_BASE_MODEL` env var) |
| `JSONL_SFT_TRAIN` | `data/train.jsonl` |
| `JSONL_PREF_DPO` | `data/assn2_pref_dpo.jsonl` |
| `JSONL_EVAL` | `data/assn2_eval.jsonl` |
| `CKPT_SFT_LORA` | `checkpoints/assn2-sft` |
| `CKPT_SFT_MERGED` | `checkpoints/assn2-sft-merged-hf` |
| `CKPT_DPO` | `checkpoints/assn2-dpo` |
| `CKPT_SIMPO` | `checkpoints/assn2-simpo` |

**Input:** None (reads `Path.cwd()` and env vars).

**Output:** All path variables and `MODEL_BASE_ID` available to the entire kernel session.

**Edit:** 🟢 Do not edit the file. To change the base model, set the environment variable **before** running this cell:
```python
import os; os.environ["ASSN2_BASE_MODEL"] = "Qwen/Qwen3-4B"
```

---

## Cell 03 — Directory Creation and Environment Variables

**What it does:** Creates `third_party/`, `data/`, and `checkpoints/` directories if they do not exist. Sets HF cache directories inside the project root (useful on shared compute). Prints Python version, PyTorch version, CUDA availability, and GPU name as a sanity check.

**Input:** Path variables from Cell 02.

**Output:**
- Directories `data/` and `checkpoints/` on disk.
- Env vars `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE` set.
- Printed: Python version, torch version, GPU name.

**Edit:** 🟢 Do not edit.

---

## Cell 04 — Inject Assignment Helpers into OpenRLHF

**What it does:** Creates a sub-package `openrlhf/assn2/` inside the editable OpenRLHF install and writes two files:

- `__init__.py` — package marker
- `reasoning.py` — contains two functions:
  - `build_prompt(question)`: Wraps a math question in a "solve step-by-step, end with `#### <answer>`" system prompt.
  - `extract_final(text)`: Scans model output for the last `#### <number>` pattern and returns the number string, or `None` if not found.

Because OpenRLHF is installed with `pip install -e`, these functions are immediately importable as `from openrlhf.assn2.reasoning import build_prompt, extract_final`.

**Input:** `openrlhf.__file__` (editable install path).

**Output:** `openrlhf/assn2/reasoning.py` written to disk; `build_prompt` and `extract_final` imported.

**Edit:** 🔴 You **may** edit `build_prompt` and `extract_final` to improve preference data quality:
- `build_prompt`: Change the system message, few-shot examples, or output format instruction. A better prompt → more consistent model outputs → cleaner chosen/rejected pairs.
- `extract_final`: Change the regex if you switch the answer format (e.g. if your prompt asks for `\boxed{}` instead of `####`).

---

## Cell 05 — Section Header: Hugging Face Authentication (Markdown)

Section label only. No code.

---

## Cell 06 — Hugging Face Login

**What it does:** Prompts for your HF access token (or reuses `HF_TOKEN` if already set), calls `huggingface_hub.login()`, and stores the token in `os.environ["HF_TOKEN"]`.

**Input:** HF token — entered interactively or via `HF_TOKEN` environment variable.

**Output:** Authenticated HF session for all subsequent model downloads and uploads.

**Edit:** 🟢 Do not edit the code. To skip the interactive prompt, set the token before running:
```python
import os; os.environ["HF_TOKEN"] = "hf_..."
```

---

## Cell 07 — Section Header: Preference Data Generation (Markdown)

Section overview — describes what the section does and flags it as editable.

---

## Cell 08 — Subsection Header: 3.1 vLLM Environment Setup (Markdown)

Subsection label only.

---

## Cell 09 — vLLM Environment Setup

**What it does:** Sets three environment variables required to prevent common vLLM crashes inside Jupyter:

| Variable set | Value | Why |
|---|---|---|
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Avoids fork-related deadlocks in notebooks |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | Disables flashinfer sampler that fails without CUDA library |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | Keeps engine in-process for notebook compatibility |

Also checks whether `flash_attn` is installed but has a mismatched CUDA ABI (`undefined symbol: flash_attn_2_cuda`). If so, it uninstalls `flash-attn` automatically and raises an error asking you to restart the kernel.

**Input:** None.

**Output:** Env vars set in the kernel process.

**Edit:** 🟢 Do not edit. These are required stability patches for vLLM in Jupyter.

---

## Cell 10 — Subsection Header: 3.2 Install vLLM and Imports (Markdown)

Subsection label only.

---

## Cell 11 — Install vLLM and Import Dependencies

**What it does:** Quietly pip-installs `vllm` if not already available. Imports:
- `load_dataset` from HuggingFace `datasets`
- `LLM`, `SamplingParams` from `vllm`
- `build_prompt`, `extract_final` from `openrlhf.assn2.reasoning` (Cell 04)

**Input:** None.

**Output:** vLLM package available; names imported into kernel.

**Edit:** 🟢 Do not edit. If you need a specific vLLM version, change `%pip install -q vllm` to `%pip install -q "vllm==x.y.z"`.

---

## Cell 12 — Subsection Header: 3.3 Configuration (EDIT HERE) (Markdown)

Subsection label. Flags this as an editable section.

---

## Cell 13 — Preference Generation Configuration

**What it does:** Sets all hyperparameters controlling how many responses are generated and how vLLM uses GPU resources.

| Variable | Default | Meaning |
|---|---|---|
| `MAX_SEED` | `1000` | Number of GSM8K questions to sample |
| `NUM_SAMPLES` | `8` | Completions generated per question |
| `GEN_TEMP` | `0.9` | Sampling temperature (higher = more diverse) |
| `MAX_NEW` | `4096` | Max tokens per completion |
| `VLLM_BATCH` | `20` | Questions per vLLM batch |
| `VLLM_GPU_MEM` | `0.75` | Fraction of GPU VRAM allocated to vLLM |
| `VLLM_TP` | `1` | Tensor-parallel GPUs (1 = single GPU) |
| `VLLM_MAX_MODEL_LEN` | `4096` | Context window length |
| `MAX_DPO_PAIRS` | `1000` | Stop generating after this many DPO pairs |

**Input:** Optionally any `ASSN2_*` env vars (allow overriding without editing the cell).

**Output:** All variables set in memory.

**Edit:** 🔴 **This is the primary config cell for preference data.** You may freely change any of these:
- `NUM_SAMPLES`: More samples give more coverage but take longer. Try `16`–`32` for better pair diversity.
- `GEN_TEMP`: `0.9` is a good default. Try `0.7` for more focused responses.
- `MAX_SEED`: How many questions to draw from. Can increase up to the full dataset size.
- `MAX_DPO_PAIRS`: How many training pairs to collect. More is generally better up to a point.

---

## Cell 14 — Subsection Header: 3.4 Helper Functions (Markdown)

Subsection label only.

---

## Cell 15 — Helper Functions for Data Generation

**What it does:** Defines five utility functions:

| Function | Purpose |
|---|---|
| `norm_answer(x)` | Normalizes a raw answer to an integer string (e.g. `"4.0"` → `"4"`, `None` → `""`) |
| `fallback_positive(ans)` | Creates a synthetic "correct" response when no positive sample was generated (rarely triggered) |
| `fallback_negative(ans)` | Creates a synthetic "wrong" response (adds 3 to the correct answer) when no negative sample was generated |
| `_cleanup_vllm_dist()` | Tears down vLLM's distributed process group; call before re-initializing vLLM to avoid port conflicts |
| `load_vllm()` | Initializes vLLM `LLM` engine with the config from Cell 13; handles attention backend selection and ImportError from flash-attn |

**Input:** Config variables from Cell 13 (`MODEL_BASE_ID`, `VLLM_TP`, `VLLM_GPU_MEM`, `VLLM_MAX_MODEL_LEN`).

**Output:** All 5 functions defined in the kernel namespace.

**Edit:** 🔴 You **may** modify:
- `fallback_positive` / `fallback_negative` — change the template text or strategy for fallback responses.
- `load_vllm` — change attention backend via `ASSN2_VLLM_ATTENTION_BACKEND` env var (e.g. `TRITON_ATTN`, `FLASH_ATTN`), or adjust `enforce_eager`.

---

## Cell 16 — Subsection Header: 3.5 Load Dataset (Markdown)

Subsection label only.

---

## Cell 17 — Load GSM8K Training Data

**What it does:** Downloads `talzoomanzoo/gsm8k` train split from HF Hub, shuffles it with `seed=42`, and selects the first `MAX_SEED` rows. Stores the result as `rows` (a Python list of dicts with `"Question"` and `"answer"` keys).

**Input:** `MAX_SEED` (from Cell 13), HF authentication (Cell 06).

**Output:**
- `ds` — HuggingFace Dataset object
- `rows` — Python list of question dicts

**Edit:** 🔴 You **may** change the dataset source entirely (different dataset, additional filtering). If you use different field names, update `ex["Question"]` and `ex["answer"]` in Cell 19 to match. You may also add additional filtering logic here (e.g. only keep questions with numeric answers).

---

## Cell 18 — Subsection Header: 3.6 Generate Responses with vLLM (Markdown)

Subsection label only.

---

## Cell 19 — vLLM Generation Loop

**What it does:** The core generation cell.

1. Loads the vLLM engine with `load_vllm()`.
2. Runs `SamplingParams` with `NUM_SAMPLES` completions per prompt at temperature `GEN_TEMP`.
3. For each question, classifies each completion:
   - **Positive** (`chosen`): `extract_final(completion)` normalized matches the gold answer.
   - **Negative** (`rejected`): `extract_final(completion)` is non-empty but wrong.
   - Ignored: no `####` marker found.
4. Appends to `label_rows` (every classified response with `label: True/False`) and `dpo_rows` (strict pairs — requires ≥1 real positive AND ≥1 real negative; skips questions where only one side exists).
5. Stops once `len(dpo_rows) >= MAX_DPO_PAIRS`.

**Input:** `llm` (created here), `rows`, `NUM_SAMPLES`, `GEN_TEMP`, `MAX_NEW`, `build_prompt`, `extract_final`, `norm_answer`, `fallback_*`.

**Output:**
- `label_rows` — list of `{prompt, response, label: bool, answer}` dicts
- `dpo_rows` — list of `{prompt, chosen, rejected, answer}` dicts

**Edit:** 🔴 **This is the most grade-relevant cell.** You may change:
- **Pair selection strategy**: Instead of `random.choice(positives)` / `random.choice(negatives)`, pick the longest correct response as `chosen`, or the most confidently wrong as `rejected`.
- **Filtering logic**: Add quality filters (e.g. minimum length, discard fallback responses from DPO pairs).
- **Labeling logic**: Change the classification — e.g. treat partial credit responses differently.
- **Prompt**: Edit `build_prompt` in Cell 04 to produce higher-quality outputs.

---

## Cell 20 — Subsection Header: 3.7 Cleanup GPU Memory and Save (Markdown)

Subsection label only.

---

## Cell 21 — Cleanup vLLM and Save Data to Disk

**What it does:**
1. Deletes the `llm` object and runs `gc.collect()` + `torch.cuda.empty_cache()` to free GPU memory.
2. Calls `_cleanup_vllm_dist()` to tear down distributed state.
3. Writes `dpo_rows` as newline-delimited JSON to `data/assn2_pref_dpo.jsonl`.
4. Writes `label_rows` to `data/assn2_pref_labels.jsonl`.

**Input:** `llm`, `dpo_rows`, `label_rows`, `JSONL_PREF_DPO`, `JSONL_PREF_LABELS`.

**Output:**
- `data/assn2_pref_dpo.jsonl` — DPO training pairs used in Section 8.
- `data/assn2_pref_labels.jsonl` — full label file (optional, for analysis).
- GPU memory freed.

**Edit:** 🟢 Do not edit.

---

## Cell 22 — Section Header: Evaluation Dataset Preparation (Markdown)

Section label only.

---

## Cell 23 — Build the Held-Out Evaluation Set

**What it does:** Downloads the `talzoomanzoo/gsm8k` **test** split, selects 100 randomly shuffled examples (seed 42), normalizes the gold answers to integer strings (using `norm_aime_answer`), and writes them to `data/assn2_eval.jsonl` as `{"question": ..., "answer": ...}`.

**Input:** HF Hub (talzoomanzoo/gsm8k test split), `JSONL_EVAL` path.

**Output:** `data/assn2_eval.jsonl` — 100 evaluation questions with normalized gold answers.

**Edit:** 🟢 Do not edit. This fixed set is used to compare all your models (base, SFT, DPO, SimPO) on the same 100 questions.

---

## Cell 24 — Section Header: Evaluation Infrastructure (Markdown)

Section overview. Explains that `eval_model(...)` defined in this section is reused by all evaluation cells.

---

## Cell 25 — Subsection Header: 5.1 Imports and Basic Helpers (Markdown)

Subsection label only.

---

## Cell 26 — Evaluation Imports and Basic Helper Functions

**What it does:** Sets up the evaluation utilities.
- Enables fast HF downloads (`hf_transfer`).
- Defines the shared helpers used by both evaluation backends:

| Function | Purpose |
|---|---|
| `build_eval_prompt(question)` | Wraps question as "Please answer … provide final answer in `\boxed{YOUR_ANSWER}`" |
| `extract_final_eval(text)` | Extracts the last `\boxed{...}` from model output; falls back to `#### <number>` if no boxed found |
| `norm_aime_answer(x)` | Normalizes answer to integer string; handles `float` → `int` rounding |
| `load_eval_examples(path)` | Reads a JSONL file and returns a list of `{question, answer}` dicts |

**Input:** `openrlhf.assn2.reasoning.extract_final` (from Cell 04).

**Output:** 4 functions defined; `_BOXED_RE` regex compiled.

**Edit:** 🟢 Do not edit.

---

## Cell 27 — Subsection Header: 5.2 HF-Backend Evaluation Function (Markdown)

Subsection label only.

---

## Cell 28 — `_eval_model_hf` — Transformers Evaluation Backend

**What it does:** Implements `_eval_model_hf(model_name_or_path, examples, max_new_tokens, mp) → float`.

- If the path contains `adapter_config.json` (LoRA adapter): loads the base model with 4-bit quantization (`BitsAndBytesConfig`) and attaches the LoRA adapter with PEFT.
- Otherwise: loads the model as a full-precision (bf16/fp16) checkpoint.
- For each eval example: generates one response with greedy decoding (`do_sample=False`), extracts the answer with `extract_final_eval`, normalizes, compares to gold.
- Returns accuracy as a float between 0.0 and 1.0.

**Input:** Model path (adapter or merged), `eval_examples`, `max_new_tokens`, `MODEL_BASE_ID`.

**Output:** Returns `float` accuracy. Prints per-question `Q / pred / gold / ok` to stdout.

**Edit:** 🟢 Do not edit. Activate with `os.environ["ASSN2_EVAL_BACKEND"] = "hf"`.

---

## Cell 29 — Subsection Header: 5.3 vLLM-Backend Evaluation Function (Markdown)

Subsection label only.

---

## Cell 30 — `_eval_model_vllm` — vLLM Evaluation Backend

**What it does:** Implements `_eval_model_vllm(model_name_or_path, examples, max_new_tokens, mp) → float`.

- Repeats the same vLLM env guards as Cell 09 (flashinfer, V1 multiprocessing, flash-attn check).
- Detects LoRA adapters — if found, uses `LoRARequest` and sets `enable_lora=True` so vLLM loads the base model + adapter efficiently.
- Generates responses in batches of `ASSN2_EVAL_VLLM_BATCH_SIZE` (default 16) using greedy decoding.
- Returns accuracy as a float.

**Input:** Model path, eval examples, `max_new_tokens`, vLLM config env vars.

**Output:** Returns `float` accuracy. Prints per-question results.

**Edit:** 🟢 Do not edit. This is the default backend.

---

## Cell 31 — Subsection Header: 5.4 Evaluation Dispatcher (Markdown)

Subsection label only.

---

## Cell 32 — `eval_model` Dispatcher and Load Eval Examples

**What it does:** Two things in one cell:

1. Defines `eval_model(model_name_or_path, examples, max_new_tokens=None) → float`:
   - Defaults `max_new_tokens` to `ASSN2_EVAL_MAX_NEW` env var (default `768`).
   - Routes to `_eval_model_hf` if `ASSN2_EVAL_BACKEND=hf`, otherwise uses `_eval_model_vllm`.

2. Loads `eval_examples` from `data/assn2_eval.jsonl` into memory (used by all eval cells).

**Input:** `JSONL_EVAL` path, `_eval_model_hf`, `_eval_model_vllm`.

**Output:** `eval_model` function and `eval_examples` list available to all subsequent eval calls.

**Edit:** 🟢 Do not edit.

---

## Cell 33 — Section Header: Base Model Evaluation (Markdown)

Section label only.

---

## Cell 34 — Evaluate Base Model (Baseline)

**What it does:** Calls `eval_model(MODEL_BASE_ID, eval_examples)` on the raw, untrained base model. Stores the result in `base_acc`. This is the **performance floor** — every fine-tuned model should score higher.

**Input:** `MODEL_BASE_ID`, `eval_examples`, `eval_model`.

**Output:** `base_acc` — float accuracy, printed to stdout.

**Edit:** 🟢 Do not edit.

---

## Cell 35 — Section Header: SFT Training Pipeline (Markdown)

Original section header from the assignment. No code.

---

## Cell 36 — Subsection Header: SFT 1 — Prepare Training Data (Markdown)

Subsection label only.

---

## Cell 37 — Prepare SFT Training Data

**What it does:**
1. Downloads `openai/gsm8k` train split, shuffles with seed 42, selects up to `MAX_SAMPLES` (5000) examples.
2. Converts the native `#### <answer>` format to `\boxed{answer}` using `to_boxed_solution()`.
3. Writes `data/train.jsonl` where each row is `{"problem": <question>, "solution": <boxed_reasoning>}`.

The `problem` / `solution` field names match what OpenRLHF's SFT trainer reads via `--data.input_key problem --data.output_key solution`.

**Input:** HF Hub (openai/gsm8k), `JSONL_SFT_TRAIN`, `CKPT_SFT_LORA`.

**Output:** `data/train.jsonl` on disk (up to 5000 rows).

**Edit:** 🟢 Do not edit.

---

## Cell 38 — Subsection Header: SFT 2 — Training Configuration (Markdown)

Subsection label only.

---

## Cell 39 — SFT Training Configuration

**What it does:** Reads all SFT hardware parameters from environment variables (with safe defaults for 1× RTX 3090):

| Variable | Default | Meaning |
|---|---|---|
| `NUM_GPUS` | `1` | Number of GPUs for DeepSpeed |
| `MASTER_PORT` | Random 20000–40000 | DeepSpeed communication port |
| `MAX_LEN` | `4096` | Max token length for SFT |
| `TRAIN_BATCH` | `4` | Global training batch size |
| `MICRO_BATCH` | `1` | Per-GPU micro batch size |
| `MAX_EPOCHS` | `2` | Training epochs |
| `ZERO_STAGE` | `2` | DeepSpeed ZeRO stage |
| `LOAD_4BIT` | `True` | Use 4-bit QLoRA |
| `LORA_RANK` | `16` (if 4-bit) | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA alpha |
| `PARAM_DTYPE` | `bf16` | Model parameter dtype |

**Input:** Env vars `ASSN2_SFT_*` (all optional).

**Output:** Config variables set in memory.

**Edit:** 🟢 Do not edit. These are memory-safe defaults — changing them may cause OOM errors.

---

## Cell 40 — Subsection Header: SFT 3 — Run Training (EDIT HERE) (Markdown)

Subsection label. Flags `--adam.lr` as the only allowed edit.

---

## Cell 41 — Build and Run SFT Training Command

**What it does:** Assembles the full DeepSpeed SFT command from all config variables and runs it:
```
deepspeed --num_gpus=1 --master_port=<port> \
  --module openrlhf.cli.train_sft \
  --model.model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --data.dataset data/train.jsonl \
  --data.input_key problem --data.output_key solution \
  --data.max_len 4096 \
  --train.batch_size 4 --train.micro_batch_size 1 \
  --train.max_epochs 2 --train.seed 42 \
  --ds.zero_stage 2 --ds.param_dtype bf16 \
  --adam.lr 5e-6 \
  --ckpt.output_dir checkpoints/assn2-sft --ckpt.save_hf \
  --ds.load_in_4bit --ds.lora.rank 16 --ds.lora.alpha 32
```

**Input:** Config from Cell 39, `MODEL_BASE_ID`, `JSONL_SFT_TRAIN`, `CKPT_SFT_LORA`.

**Output:** LoRA adapter checkpoint saved to `checkpoints/assn2-sft/`. Training logs printed to notebook (~25 min on 1× RTX 3090).

**Edit:** 🔴 **Only change `--adam.lr`** (line `os.environ.get("ASSN2_SFT_ADAM_LR", "5e-6")`).
- Default: `5e-6`
- Recommended range: `2e-6` (more conservative) to `1e-5` (faster but risks instability).
- Too high → adapter collapses and performance degrades. Too low → slow convergence.

---

## Cell 42 — Subsection Header: SFT 4 — Merge LoRA Adapter (Markdown)

Subsection label only.

---

## Cell 43 — Merge SFT LoRA into Base Model

**What it does:**
1. Finds the most-recently-modified directory under `checkpoints/assn2-sft/` that contains `adapter_config.json`.
2. Loads the base model in bf16 and attaches the LoRA adapter with PEFT.
3. Calls `peft.merge_and_unload()` to fuse the adapter weights permanently into the base model.
4. Saves the merged full model to `checkpoints/assn2-sft-merged-hf/` (safetensors format).
5. Saves the tokenizer alongside.

**Input:** `CKPT_SFT_LORA` (adapter checkpoint), `MODEL_BASE_ID` (base weights), `CKPT_SFT_MERGED` (output path).

**Output:**
- Merged HF model at `checkpoints/assn2-sft-merged-hf/`.
- Variable `SFT_MERGED_DIR` set.

**Edit:** 🟢 Do not edit.

---

## Cell 44 — Subsection Header: SFT 5 — Evaluate Merged Model (Markdown)

Subsection label only.

---

## Cell 45 — Evaluate Merged SFT Model

**What it does:** Calls `eval_model(SFT_MERGED_DIR, eval_examples)` to score the merged SFT model on the held-out set. Prints `sft_acc` and the delta relative to `base_acc`.

**Input:** `SFT_MERGED_DIR`, `eval_examples`, `base_acc`, `eval_model`.

**Output:** `sft_acc` (float, printed). GPU memory freed after evaluation.

**Edit:** 🟢 Do not edit.

---

## Cell 46 — Subsection Header: SFT 6 — Upload to Hugging Face (Markdown)

Subsection label only.

---

## Cell 47 — Upload Merged SFT Model to HF Hub

**What it does:**
1. Validates that `SFT_MERGED_DIR` contains full model weights (not just an adapter).
2. Creates a model repository on HF Hub (or reuses if it exists).
3. Uploads the entire directory with `api.upload_folder(...)`.

**Input:** `SFT_MERGED_DIR`, HF token (from Cell 06), `HF_SFT_REPO_ID` env var or interactive prompt.

**Output:** Model uploaded to `https://huggingface.co/<repo_id>`.

**Edit:** 🟢 Do not edit the code. To skip the interactive prompt:
```python
import os; os.environ["HF_SFT_REPO_ID"] = "your-username/assn2-sft"
```

---

## Cell 48 — Section Header: DPO Training Pipeline (Markdown)

Original section header from the assignment. No code.

---

## Cell 49 — Subsection Header: DPO 1 — Environment and Path Setup (Markdown)

Subsection label only.

---

## Cell 50 — DPO Environment and Base Config

**What it does:**
- Sets `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation OOM errors.
- Sets `PATH_DPO_INIT` — the model to start DPO from (default: merged SFT model). Override with `ASSN2_DPO_INIT_MODEL`.
- Creates `checkpoints/assn2-dpo/`.
- Installs DeepSpeed if missing.
- Reads base hardware config: `NUM_GPUS` (4), `MAX_LEN` (4096), `ZERO_STAGE` (2), `PARAM_DTYPE` (bf16), `REF_OFFLOAD` (True), `ADAM_OFFLOAD` (True).

**Input:** `CKPT_SFT_MERGED`, env vars `ASSN2_NUM_GPUS`, `ASSN2_MAX_LEN`, etc.

**Output:** Variables `PATH_DPO_INIT`, `NUM_GPUS`, `MAX_LEN`, etc. set in memory.

**Edit:** 🟢 Do not edit. To start DPO from a different checkpoint:
```python
import os; os.environ["ASSN2_DPO_INIT_MODEL"] = "/path/to/model"
```

---

## Cell 51 — Subsection Header: DPO 2 — Configuration (EDIT HERE) (Markdown)

Subsection label. Flags batch sizes as editable.

---

## Cell 52 — DPO Hyperparameter Configuration

**What it does:** Reads DPO-specific parameters and validates the DeepSpeed batch constraint.

| Variable | Default | Meaning |
|---|---|---|
| `TRAIN_BATCH_SIZE` | `16` | Global DPO batch size |
| `MICRO_BATCH_SIZE` | `1` | Per-GPU micro batch |
| `LORA_RANK` | `0` | LoRA rank (0 = no LoRA for DPO by default) |
| `PACKING_SAMPLES` | `False` | Sequence packing (needs flash-attn) |
| `RING_ATTN_SIZE` | `1` | Ring attention group size |
| `GRAD_CKPT` | `True` | Gradient checkpointing |

Also auto-disables gradient checkpointing when running 4-bit (avoids a known PEFT bug).

Validates: `TRAIN_BATCH_SIZE >= NUM_GPUS * MICRO_BATCH_SIZE`. If not, auto-raises `TRAIN_BATCH_SIZE`.

**Input:** Env vars `ASSN2_TRAIN_BATCH_SIZE`, `ASSN2_MICRO_BATCH_SIZE`, etc.; `NUM_GPUS` from Cell 50.

**Output:** `TRAIN_BATCH_SIZE`, `MICRO_BATCH_SIZE`, and all auxiliary flags set in memory.

**Edit:** 🔴 **Only `TRAIN_BATCH_SIZE` and `MICRO_BATCH_SIZE` are allowed edits.**
- A larger *effective batch* (`TRAIN_BATCH_SIZE / MICRO_BATCH_SIZE / NUM_GPUS`) stabilizes DPO because preference signals are noisy.
- Keep `TRAIN_BATCH_SIZE >= NUM_GPUS * MICRO_BATCH_SIZE` (the cell enforces this automatically).
- Example: `TRAIN_BATCH_SIZE=32, MICRO_BATCH_SIZE=2` on 4 GPUs → grad accumulation steps = 4.

---

## Cell 53 — Subsection Header: DPO 3 — GPU Selection and Command Building (Markdown)

Subsection label only.

---

## Cell 54 — GPU Auto-Selection and DPO Command Assembly

**What it does:**
1. Queries `nvidia-smi` for GPU memory usage and selects the `NUM_GPUS` least-loaded GPUs.
2. Picks a random `MASTER_PORT`.
3. Builds the full `deepspeed ... openrlhf.cli.train_dpo` command including all flags: model path, reference model path, preference data path, DPO beta, LR, batch sizes, ZeRO stage, dtype, and optional flags for LoRA, gradient checkpointing, packing, ring attention, offloading, and Liger kernel.

**Input:** All config from Cells 50 and 52; `PATH_DPO_INIT`, `JSONL_PREF_DPO`, `CKPT_DPO`.

**Output:** `cmd` (list of strings), `DS_INCLUDE` (GPU selection string), `MASTER_PORT`; prints config summary.

**Edit:** 🔴 **Two values are allowed edits**, both inside `cmd`:
- `--model.beta "0.25"` → **DPO beta** (KL regularization strength).
  - Higher beta (e.g. `0.5`) → more conservative, stays closer to the SFT model.
  - Lower beta (e.g. `0.1`) → more aggressive optimization.
- `--adam.lr "5e-7"` → **DPO learning rate**.
  - Lower LR is safer for small or noisy preference datasets.
  - Recommended range: `1e-7` to `1e-6`.

---

## Cell 55 — Subsection Header: DPO 4 — Run DPO Training (Markdown)

Subsection label only.

---

## Cell 56 — Run DPO Training

**What it does:** Prints a one-line config summary (key settings visible at a glance), then executes the DeepSpeed DPO command built in Cell 54 via `get_ipython().system(cmd_str)`.

**Input:** `cmd` from Cell 54.

**Output:** DPO checkpoint saved to `checkpoints/assn2-dpo/`. Training logs to notebook (~10 min on 4× RTX 3090).

**Edit:** 🟢 Do not edit.

---

## Cell 57 — Subsection Header: DPO 5 — Evaluate DPO Checkpoint (Markdown)

Subsection label only.

---

## Cell 58 — Evaluate DPO Checkpoint

**What it does:**
1. Searches `checkpoints/assn2-dpo/` for the most-recently-modified HF checkpoint. Prefers LoRA adapter dirs (if DPO was run with LoRA), falls back to merged dirs.
2. Calls `eval_model(hf_model_dir, eval_examples)`.
3. Prints `tuned_acc` and the delta vs. `base_acc`.
4. Frees GPU memory and cleans up distributed state.

**Input:** `CKPT_DPO`, `eval_examples`, `base_acc`, `eval_model`.

**Output:** `tuned_acc` (float, printed). GPU memory freed.

**Edit:** 🟢 Do not edit. To evaluate a different checkpoint:
```python
import os; os.environ["ASSN2_TUNED_OUT"] = "/path/to/checkpoint"
```

---

## Cell 59 — Subsection Header: DPO 6 — Merge and Upload (Markdown)

Subsection label only.

---

## Cell 60 — Sub-header: Helper Functions (Markdown)

Sub-subsection label only.

---

## Cell 61 — DPO Upload Helper Functions

**What it does:** Defines 4 utility functions used by Cell 63:

| Function | Purpose |
|---|---|
| `has_adapter_artifacts(p)` | Returns True if the dir contains `adapter_config.json` or adapter weight files |
| `has_full_weights(p)` | Returns True if the dir contains `model.safetensors` or sharded weight files |
| `assert_uploadable_full_model(p)` | Raises an error if the dir is missing, adapter-only, or has no weights |
| `merged_dir_for_upload(hf_src)` | If `hf_src` is an adapter, merges it with the base model and saves to `checkpoints/assn2-dpo-merged-hf/`; if already full model, returns as-is |

**Input:** None (function definitions).

**Output:** 4 functions available.

**Edit:** 🟢 Do not edit.

---

## Cell 62 — Sub-header: Find Checkpoint, Merge, and Upload (Markdown)

Sub-subsection label only.

---

## Cell 63 — Find DPO Checkpoint, Merge, and Upload to HF Hub

**What it does:**
1. Searches `checkpoints/assn2-dpo/` for the latest adapter or full-model checkpoint.
2. Prompts for a Hub repo id (or reads `HF_REPO_ID` env var).
3. Calls `merged_dir_for_upload(hf_src)` — merges adapter into base if needed.
4. Validates the merged directory with `assert_uploadable_full_model`.
5. Creates the HF model repo and uploads all files.

**Input:** `CKPT_DPO`, `CKPT_DPO_MERGED`, HF token, `HF_REPO_ID` env var or interactive prompt.

**Output:** Full merged DPO model uploaded to `https://huggingface.co/<repo_id>`.

**Edit:** 🟢 Do not edit the code. Set env vars to avoid interactive prompts:
```python
import os
os.environ["HF_REPO_ID"] = "your-username/assn2-dpo"
os.environ["ASSN2_DPO_MERGED_OUT"] = "/custom/merge/path"  # optional
```

---

## Cell 64 — Subsection Header: DPO 7 — Upload Preference Dataset (Markdown)

Subsection label only.

---

## Cell 65 — Upload Preference Dataset to HF Hub

**What it does:** Uploads `data/assn2_pref_dpo.jsonl` (your synthetic preference dataset from Section 3) to a HF **dataset** repository as `assn2_pref.jsonl`.

**Input:** `JSONL_PREF_DPO` (file on disk), HF token, `HF_PREF_REPO_ID` or `HF_DATASET_REPO_ID` env var or interactive prompt.

**Output:** Preference JSONL uploaded to `https://huggingface.co/datasets/<repo_id>`.

**Edit:** 🟢 Do not edit the code. Set env var to skip prompt:
```python
import os; os.environ["HF_PREF_REPO_ID"] = "your-username/assn2-prefs"
```

---

## Cell 66 — Section Header: SimPO Pipeline (Markdown)

Original section header from the assignment. No code.

---

## Cell 67 — SimPO Implementation (EDIT HERE — 25% of grade)

**What it does:** Currently an empty stub with suggested structure. You must implement the full SimPO pipeline here.

**Input:** `eval_examples`, `eval_model`, `CKPT_SIMPO`, `CKPT_SIMPO_MERGED`, and `JSONL_PREF_DPO` (can reuse the same preference data as DPO).

**Output (you implement):** SimPO checkpoint, evaluation score, uploaded model.

**Edit:** 🔴 **Implement the full pipeline:**

### Step 1 — SimPO Algorithm in OpenRLHF

SimPO (Simple Preference Optimization) removes the reference model and replaces per-token log-prob rewards with *average* log-prob rewards, plus a target-reward-margin term:

```
loss = -log σ(β * (avg_logp(y_w|x) - avg_logp(y_l|x) - γ))
```

where `γ > 0` is the margin (target reward gap). Implement this in OpenRLHF — options:
- Add `openrlhf/assn2/simpo_trainer.py` (recommended: copy `openrlhf/trainer/dpo_trainer.py` and change the loss).
- Or modify `openrlhf/trainer/dpo_trainer.py` to accept a `loss_type="simpo"` flag.

Key difference from DPO: no `ref_model` logprobs needed; divide log-probs by sequence length before computing the difference.

### Step 2 — Training Command

Follow the same pattern as Cells 54 and 56. Use your SimPO trainer module:
```python
cmd = ["deepspeed", ..., "--module", "openrlhf.cli.train_simpo", ...]
```
Save to `CKPT_SIMPO`.

### Step 3 — Merge LoRA (if applicable)

Same pattern as Cell 43 — merge adapter into base, save to `CKPT_SIMPO_MERGED`.

### Step 4 — Evaluate

```python
simpo_acc = eval_model(str(CKPT_SIMPO_MERGED), eval_examples)
print("simpo_acc:", simpo_acc)
print("delta vs base:", simpo_acc - base_acc)
```

### Step 5 — Upload to HF Hub

Same pattern as Cell 63. Use a new repo id (e.g. `your-username/assn2-simpo`).

---

## Quick Reference: What You Can Edit

| Cell | What to change | Notes |
|---|---|---|
| **04** | `build_prompt`, `extract_final` | Change prompt format and answer extraction |
| **13** | `NUM_SAMPLES`, `GEN_TEMP`, `MAX_NEW`, `MAX_SEED`, `MAX_DPO_PAIRS` | Preference data generation config |
| **15** | `fallback_positive/negative`, `load_vllm` kwargs | Optional helper improvements |
| **17** | Dataset source, field names | Use a different dataset if desired |
| **19** | Pair selection logic, filtering strategy | Most grade-relevant for preference data |
| **41** | `--adam.lr` (SFT learning rate) | **Only this one value** |
| **52** | `TRAIN_BATCH_SIZE`, `MICRO_BATCH_SIZE` | Keep batch ≥ NUM_GPUS × micro |
| **54** | `--model.beta`, `--adam.lr` (DPO) | **Only these two values** |
| **67** | Everything | Implement full SimPO pipeline |
