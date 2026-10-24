# ASSN2 — vLLM + OpenRLHF for Efficient Math Reasoning

**TA:** Minju Gwak (mjgwak@yonsei.ac.kr)

## Goal

Train and evaluate preference-optimized math models using **OpenRLHF**, **vLLM**, and **Hugging Face**. Students edit only sections marked `### EDIT HERE ###`.

---

## Grading

| Component | Weight |
|---|---|
| Environment + notebook execution | 10% |
| HF upload + leaderboard (SFT/DPO) | 10% |
| Performance improvement | 10% |
| Original synthetic preference-data pipeline | 15% |
| SimPO implementation + training/eval/upload | 25% |
| Analysis and justification | 30% |

---

## Pipeline Overview

| Step | Description |
|---|---|
| 1. Setup | Install OpenRLHF (editable), configure paths |
| 2. HF Login | Authenticate for gated models and uploads |
| 3. Preference Data | Generate `chosen`/`rejected` pairs via vLLM |
| 4. Eval Data | Download GSM8K test split → `data/assn2_eval.jsonl` |
| 5. Eval Infra | Define `eval_model(...)` (vLLM or HF backend) |
| 6. Base Eval | Establish baseline accuracy |
| 7. SFT | Train LoRA adapter on GSM8K; merge into HF model |
| 8. DPO | Train on generated preference pairs |
| 9. Upload | Push models/dataset to HF Hub + leaderboard |
| 10. SimPO | Implement, train, evaluate, and upload |

---

## Datasets and Models

| Stage | Source | Local Path | Purpose |
|---|---|---|---|
| Base model | `meta-llama/Llama-3.2-1B-Instruct` | `MODEL_BASE_ID` | Starting point |
| SFT data | `openai/gsm8k` (train) | `data/train.jsonl` | Supervised fine-tuning |
| Preference data | `talzoomanzoo/gsm8k` + vLLM | `data/assn2_pref_dpo.jsonl` | DPO/SimPO pairs |
| Label data | Same vLLM generations | `data/assn2_pref_labels.jsonl` | Optional analysis |
| Eval data | Course-provided | `data/assn2_eval.jsonl` | Held-out evaluation |

### Checkpoints

| Name | Path |
|---|---|
| SFT LoRA adapter | `checkpoints/assn2-sft` |
| Merged SFT model | `checkpoints/assn2-sft-merged-hf` |
| DPO checkpoint | `checkpoints/assn2-dpo` |
| Merged DPO model | `checkpoints/assn2-dpo-merged-hf` |
| SimPO checkpoint | `checkpoints/assn2-simpo` |
| Merged SimPO model | `checkpoints/assn2-simpo-merged-hf` |

---

## Section Details

### 1. Environment Setup

- Installs OpenRLHF as editable package in `third_party/OpenRLHF/`
- Injects `openrlhf/assn2/reasoning.py` with two helpers:
  - `build_prompt(question)` — consistent chain-of-thought prompt
  - `extract_final(text)` — extracts `#### <answer>` from model output
- Sets HF cache dirs inside the repo (`.hf/`)

### 2. Hugging Face Authentication

- Reads `HF_TOKEN` env var or prompts interactively
- Calls `huggingface_hub.login()`

### 3. Preference Data Generation *(EDIT HERE)*

Generates DPO training pairs using vLLM:

**Config knobs (editable):**
- `NUM_SAMPLES = 8` — completions per question
- `GEN_TEMP = 0.9` — sampling temperature
- `MAX_NEW = 4096` — max tokens per completion
- `MAX_SEED = 1000` — questions to sample from dataset
- `MAX_DPO_PAIRS = 1000` — cap on saved DPO pairs

**Logic:**
1. Load `talzoomanzoo/gsm8k` train split (shuffled)
2. For each question, generate `NUM_SAMPLES` completions via vLLM
3. Classify: correct answer → `positives`; wrong extractable answer → `negatives`
4. Save label rows (all responses with `True`/`False` labels)
5. Save strict DPO pairs: real correct `chosen` + wrong `rejected`

**Outputs:** `assn2_pref_dpo.jsonl`, `assn2_pref_labels.jsonl`

### 4. Evaluation Dataset

- Loads `talzoomanzoo/gsm8k` test split (100 examples, shuffled seed=42)
- Normalizes answers (rounds floats, strips `####` prefix)
- Saves to `data/assn2_eval.jsonl` with `{question, answer}` fields

### 5. Evaluation Infrastructure

Two backends selectable via `ASSN2_EVAL_BACKEND`:

**vLLM (default):** Batched generation; supports LoRA via `LoRARequest`

**HF Transformers (`=hf`):** 4-bit QLoRA or full-precision; CPU-compatible

Answer matching: tries `\boxed{...}` format first, falls back to `#### <answer>`.

Key env vars:
- `ASSN2_EVAL_BACKEND` — `vllm` or `hf`
- `ASSN2_EVAL_MAX_NEW` — max tokens (default 768)
- `ASSN2_VLLM_GPU_MEM_UTIL` — GPU memory fraction (default 0.88)

### 6. Base Model Evaluation

```python
base_acc = eval_model(MODEL_BASE_ID, eval_examples)
```

### 7. SFT Training Pipeline *(EDIT HERE: LR only)*

**Data prep:** Converts `openai/gsm8k` to `{problem, solution}` JSONL; wraps final answer in `\boxed{...}`.

**Config (mostly fixed):**
- `MAX_LEN = 4096`, `TRAIN_BATCH = 4`, `MICRO_BATCH = 1`, `MAX_EPOCHS = 2`
- LoRA rank 16, alpha 32 (4-bit QLoRA)
- DeepSpeed ZeRO stage 2

**Editable:**
- `--adam.lr` (default `5e-6`) — **the only intended edit**

**Merge:** Uses `peft.PeftModel.merge_and_unload()` → saves full HF model.

**Upload:** Validates no adapter artifacts remain before pushing to Hub.

### 8. DPO Training Pipeline *(EDIT HERE: LR, beta, batch sizes)*

**Init model:** Merged SFT checkpoint (`checkpoints/assn2-sft-merged-hf`)

**Editable hyperparameters:**
- `--adam.lr` (default `5e-7`)
- `--model.beta` (default `0.25`) — KL penalty strength
- `TRAIN_BATCH_SIZE` (default 16)
- `MICRO_BATCH_SIZE` (default 1)

**Key features:**
- Auto-selects least-loaded GPUs via `nvidia-smi`
- Reference model CPU offload (`--ref.offload`)
- Adam optimizer CPU offload (`--ds.adam_offload`)
- Optional ring attention, packing, Liger kernel

**Eval:** Finds latest HF checkpoint under `checkpoints/assn2-dpo`, runs `eval_model(...)`.

**Upload:** Merges adapter if needed, then pushes to Hub. Also uploads preference dataset separately.

### 9. SimPO Pipeline *(EDIT HERE: full implementation)*

```python
# ### EDIT HERE ### SIMPO pipeline area
# Suggested structure:
# 1) Add/modify OpenRLHF SimPO trainer logic
# 2) Build SimPO launch command (similar to DPO cell)
# 3) Evaluate merged SimPO model with eval_model(...)
```

---

## What To Edit

| Section | What to change |
|---|---|
| SFT | Learning rate (`--adam.lr`) only |
| Preference data | Prompt, model, decoding settings, filters, pair construction (full freedom) |
| DPO | Learning rate, beta, train batch size, micro batch size only |
| SimPO | Full implementation: trainer logic, launch command, eval, upload |

---

## Notes

- Expected runtime: SFT ~25 min (1× RTX 3090); preference opt ~10 min (4× RTX 3090)
- Get HF access approval for `meta-llama/Llama-3.2-1B-Instruct` before running
- Restart kernel if CUDA memory remains allocated after a failed run
- Submit only one final model to the leaderboard
- Leaderboard uses a different dataset than the held-out eval set
