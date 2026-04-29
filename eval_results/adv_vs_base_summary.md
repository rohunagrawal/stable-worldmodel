# ADV vs Base Planning Results Summary

Adversarial fine-tuning uses FGSM in embedding space (perturbing `ctx_emb` and `ctx_act` before the prediction step).  
All results: `horizon=5`, `num_eval=50`.  
No CEM results available for ADV checkpoints (CEM evals are base only).

---

## PushT

**Checkpoints**: `lewm-pusht-adv` vs `lewm-pusht`  
**Goal offset**: 25 steps | **Solver**: Adam (`n_steps=30`, `lr=0.1`)

| Model | Epoch | Solver | Seeds | Mean | Std | Per-seed rates |
|-------|------:|--------|------:|-----:|----:|----------------|
| Base  | —     | Adam   | 5     | 66.4% | 9.1% | 80%, 60%, 54%, 72%, 66% |
| ADV   | 1     | Adam   | 5     | 64.4% | 8.2% | 80%, 62%, 56%, 60%, 64% |

Training was stopped after epoch 1; further epochs not evaluated. No CEM results for PushT.

---

## Reacher

**Checkpoints**: `lewm-reacher-adv` vs `lewm-reacher`  
**Goal offset**: 25 steps

| Model | Epoch | Solver | Seeds | Mean | Std | Per-seed rates |
|-------|------:|--------|------:|-----:|----:|----------------|
| Base  | —     | CEM (`n_steps=30`, `topk=30`, `num_samples=300`) | 1 | 76.0% | — | 76% (seed 42) |
| Base  | —     | Adam (`n_steps=30`, `lr=0.1`) | 5 | 60.0% | 3.3% | 56%, 60%, 60%, 66%, 58% |
| ADV   | 3     | Adam (`n_steps=30`, `lr=0.1`) | 5 | **62.0%** | 7.5% | 52%, 60%, 74%, 66%, 58% |

Best ADV (epoch 3) is **+2pp** over Adam base, but **-14pp** vs CEM base (single seed). ADV performance degrades past epoch 3.

---

## Cube (OGBench)

**Checkpoints**: `lewm-ogbcube-adv` vs `lewm-cube`  
**Solver**: CEM (`n_steps=30`, `topk=30`, `num_samples=300`, `batch_size=1`) or Adam (`n_steps=30`, `lr=0.1`)

### Goal offset = 25 steps (easier)

| Model | Epoch | Solver | Seeds | Mean | Std | Per-seed rates |
|-------|------:|--------|------:|-----:|----:|----------------|
| Base  | —     | Adam   | 3     | 75.3% | 6.9% | 66%, 78%, 82% (seeds 42, 123, 456) |
| Base  | —     | CEM    | 2     | 65.0% | 1.0% | 66%, 64% (seeds 42, 2) |

### Goal offset = 50 steps (harder)

| Model | Epoch | Solver | Seeds | Mean | Std | Per-seed rates |
|-------|------:|--------|------:|-----:|----:|----------------|
| Base  | —     | Adam   | 5     | 51.6% | 5.0% | 54%, 44%, 58%, 54%, 48% |
| Base  | —     | CEM    | 3     | 51.3% | 3.3% | 48%, 50%, 56% (seeds 42, 123, 456) |
| ADV   | 2     | Adam   | 3*    | **56.0%** | 5.7% | 52%, 64%, 52% (seeds 42, 456, 789) |

*Seeds 123 and 1024 failed due to container errors; ADV estimate is from 3/5 seeds.

At goal_offset=50, Adam and CEM are roughly equivalent for the base model (~51%). ADV epoch 2 is **+4.4pp** over base Adam and **+4.7pp** over base CEM, though with fewer seeds.
