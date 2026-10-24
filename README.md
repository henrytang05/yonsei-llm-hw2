# HW2: Preference Optimization for Mathematical Reasoning

**Course:** CAS4133 — Large Language Models (Yonsei University)  
**Model:** LLaMA-3.2-1B-Instruct fine-tuned on GSM8K with DPO and SimPO

---

## Overview

This assignment builds a full pipeline for improving mathematical reasoning in LLMs using **preference optimization** with [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and **vLLM** for efficient inference.

### Pipeline

| Step | Description |
|---|---|
| 1. SFT | Supervised fine-tuning on GSM8K training data (up to 5,000 samples) |
| 2. Preference data | Generate `chosen`/`rejected` pairs via vLLM sampling |
| 3. DPO | Direct Preference Optimization on generated pairs |
| 4. SimPO | Simple Preference Optimization (no reference model) |
| 5. Eval | Evaluate all checkpoints on GSM8K test split |
| 6. Upload | Push models and dataset to HuggingFace Hub |

### Algorithms

**DPO (Direct Preference Optimization):**  
Directly optimizes a Bradley-Terry preference model without a separate reward model:
```
L_DPO = -E[log σ(β log(π(y_w|x)/π_ref(y_w|x)) - β log(π(y_l|x)/π_ref(y_l|x)))]
```

**SimPO (Simple Preference Optimization):**  
Eliminates the reference model by using length-normalized log-likelihood as implicit reward:
```
L_SimPO = -E[log σ((β/|y_w|)log π(y_w|x) - (β/|y_l|)log π(y_l|x) - γ)]
```

---

## Grading Breakdown

| Component | Weight |
|---|---|
| Environment + notebook execution | 10% |
| HF upload + leaderboard (SFT/DPO) | 10% |
| Performance improvement | 10% |
| Original synthetic preference-data pipeline | 15% |
| SimPO implementation + training/eval/upload | 25% |
| Analysis and justification | 30% |

---

## Project Structure

```
CAS4133-assn2/
├── assn2_clean.ipynb            # Clean notebook template
├── assn2_stu_refactor.ipynb     # Refactored implementation
├── assn2_stu_v2.ipynb           # Final submission version
├── assn2_summary.md             # Summary of key design decisions
├── instruction.md               # Assignment instructions
├── setup_script.sh              # Environment setup script
├── requirements.txt             # Python dependencies
├── third_party/                 # (git-ignored) Clone OpenRLHF here
└── latex/
    └── report.tex               # Report source
```

---

## Setup

### 1. Create conda environment

```bash
conda create -n hw2 python=3.11 -y
conda activate hw2
python -m pip install -U pip
```

### 2. Clone OpenRLHF

```bash
mkdir -p third_party
git clone https://github.com/OpenRLHF/OpenRLHF.git third_party/OpenRLHF
```

### 3. Install PyTorch + OpenRLHF with vLLM

```bash
# Match your CUDA version (cu121, cu124, cu128 — see pytorch.org)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e "third_party/OpenRLHF[vllm]" --no-build-isolation
```

### 4. Notebook dependencies

```bash
python -m pip install -U jupyter ipykernel transformers datasets accelerate deepspeed
python -m ipykernel install --user --name hw2 --display-name "Python (hw2)"
```

### 5. Set HuggingFace token

```bash
export HF_TOKEN=your_token_here
```

---

## Running

```bash
jupyter lab
# Open assn2_stu_v2.ipynb
```

The notebook covers:
1. Environment setup and HF authentication
2. Preference data generation (custom pipeline — key contribution)
3. SFT training with LoRA on GSM8K
4. DPO training on generated preference pairs
5. SimPO implementation and training
6. Evaluation on GSM8K test split
7. Model upload to HuggingFace Hub

---

## Key Design Decisions

See [assn2_summary.md](assn2_summary.md) for detailed analysis.

Key contributions:
- **Preference data generation:** vLLM samples multiple CoT solutions per problem; select `chosen` (correct answer) vs `rejected` (incorrect) pairs
- **SimPO implementation:** Reference-free objective with length normalization and margin γ
- **Chain-of-thought prompting:** Consistent `build_prompt()` template for reproducible generation
