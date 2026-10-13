# Image continual learning (Split-FMNIST / Split-CIFAR10)

Unconditional CFM on Masip's `diffusers.UNet2DModel`: `ddim_medium` for
Fashion-MNIST and `ddim_medium_3ch` for CIFAR-10.

- **naive** — train only on the current task
- **generative** — generative replay (synthetic old endpoints mixed by λ)
- **trajectory** — trajectory replay (field match to frozen teacher + CFM on new)
## λ selection (paper §2.4)

`--lambda-mode grid` (default): after task 0, each later task tries `--lambda-grid`
(default `0.1 0.25 0.5 0.75 0.9`). The candidate with the best **macro-average
validation FID** over all tasks seen so far is kept.

`--lambda-mode equal`: no sweep. At task \(k\) (with \(n=k+1\) tasks seen), use
fixed \(\lambda=(n-1)/n\) on the old/teacher term (equal weight per task). Useful
ablation vs selected λ.

Test FID / stream KLD are reporting-only (never used for λ).

Class orders for seeds `42 / 69 / 1714` match Avalanche `SplitFMNIST` and
`SplitCIFAR10`.

Default train/ODE knobs follow Masip:

| Dataset | epochs | batch | ODE steps |
|---------|--------|-------|-----------|
| `split_fmnist` | 100 | 128 | 10 (`--ode-step-size 0.1`) |
| `split_cifar10` | 200 | 64 | 20 (`--ode-step-size 0.05`) |

## Metrics

- **Test FID** — stream of all tasks seen so far (reporting).
- **Stream KLD** — Masip class-histogram KLD via the dataset-specific
  `weights/cnn_{fmnist,cifar10}/resnet.pth`
  (disable with `--no-kld`).
- **Timing** — `timing.json` / per-task `tasks/k/timing.json` with single-FT train
  walls, λ-sweep walls, and MeanFlow distill walls.

## Replay backend (paper §2.3)

`--replay-backend ode` (default): teacher sampling via Euler ODE during FT.

`--replay-backend mean_flow`:
- **Task 0:** train CFM, then train `MF^0` on the **same real data** with OT
  velocity `v = x1 - x0` (not distilled from the CFM field).
- **Tasks k≥1:** jointly train CFM + MeanFlow (warm-start from `MF^{k-1}`).
  - **Trajectory:** one shared `MF^{k-1}` rollout to `r` (`--mean-flow-steps` /
    `MEAN_FLOW_STEPS` jumps, table rows use **N=1** and **N=4**); evaluate
    frozen `v^{k-1}`; CFM field match + MF identity share that state; MF also
    samples `t≥r`. New branch = real data.
  - **Generative:** `MF^{k-1}` samples endpoints; both CFM and MF use
    data-conditioned losses on those samples and on real task data.

FID/KLD still use ODE on the CFM UNet.

MeanFlow defaults follow zhuyu CIFAR/ImageNet schedules. **Adaptive L2 applies to
task 0 only** (`p=0.75` FMNIST/CIFAR, `p=1.0` ImageNet) — the joint k≥1 loss keeps
unweighted MSE, because adaptive weighting self-normalizes each branch and `λ`
would stop mixing comparable squared errors. Task-0 MF is data-conditioned (OT `v`);
task-0 and joint CL share the MF optimizer: **AdamW, lr `1e-4`, clip `1.0`, β₂=`0.95`**.
Task-0 batch size matches CFM (`128` FMNIST, `64` CIFAR). CFM `--lr` stays `2e-4` on FMNIST/CIFAR.

The reference uses data→noise time, while this repo uses noise→data. Therefore
its logit-normal means are mirrored (`μ_repo = -μ_reference`); `σ` and the
diagonal mass are unchanged.

| Dataset | times | `diag_fraction` | `adaptive_p` (task 0) | `adaptive_p` (joint) | MF optimizer |
|---------|-------|-----------------|-----------------------|----------------------|--------------|
| FMNIST / CIFAR-10 | `μ=+2, σ=2` | `0.25` | `0.75` | off | AdamW `1e-4`, clip `1.0` |
| ImageNet-100 | `μ=+0.4, σ=1.0` | `0.75` | `1.0` | off | AdamW `1e-4`, clip `1.0` |

```bash
# FMNIST MeanFlow table grid (24 jobs: gen/traj × grid/equal × N=1,4 × 3 seeds):
DRY_RUN=1 ./sbatch/image_cl/submit_fmnist_meanflow.sh
./sbatch/image_cl/submit_fmnist_meanflow.sh
# → results_dir/image_cl/fmnist/{generative,trajectory}_mf_n{1,4}[_eq]_s{seed}/

# CIFAR-10 MeanFlow (same 24-job layout; λ-grid {0.25,0.5,0.75} like ODE g3):
DRY_RUN=1 ./sbatch/image_cl/submit_cifar10_meanflow.sh
./sbatch/image_cl/submit_cifar10_meanflow.sh
# → results_dir/image_cl/cifar10/{generative,trajectory}_mf_n{1,4}_g3_s{seed}/
# → results_dir/image_cl/cifar10/{generative,trajectory}_mf_n{1,4}_eq_s{seed}/

# ImageNet-100 MeanFlow (same 24-job layout; zhuyu IN schedule, class-cond recipe):
DRY_RUN=1 ./sbatch/image_cl/submit_imagenet100_meanflow.sh
./sbatch/image_cl/submit_imagenet100_meanflow.sh
# → results_dir/image_cl/imagenet100/{generative,trajectory}_mf_n{1,4}_g3_s{seed}/
# → results_dir/image_cl/imagenet100/{generative,trajectory}_mf_n{1,4}_eq_s{seed}/
```

## ImageNet-100 class-conditional protocol

Locked recipe (defaults in `submit_imagenet100_min.sh` / `train_cl.sbatch`):

- **3 tasks** (34/33/33); override with `N_EXPERIENCES=5` for Masip 5×20
- class-conditional + CFG 1.5, UNet **large**, lr **1e-4**, grad clip **1.0**
- **~100k Adam steps/task** (`EPOCHS=667` @ batch 256; 100k beat 200k on FID in e3 pilots)
- Euler **~50 NFEs** (`ODE_STEP_SIZE=0.02`); stream FID **10000**; per-class FID **off**; stream KLD off
- **cumulative schedule=`steps`**: keep that step budget fixed while the real-data
  union grows (epochs shrink). FMNIST/CIFAR keep Masip-style fixed epochs.

Run names match CIFAR/FMNIST (`naive_s42`, `cumulative_s42`, …). Pilot /
ablation dirs live under `results_dir/image_cl/imagenet100/pilot/`.

```bash
# Finetune + joint upper bound (3 seeds):
STRATEGIES="naive cumulative" ./sbatch/image_cl/submit_imagenet100_min.sh

# Table 3 FM replay rows (ODE first; then MeanFlow):
STRATEGIES="generative trajectory" LAMBDA_MODE=grid RUN_SUFFIX=g3 \
  TIME=72:00:00 ./sbatch/image_cl/submit_imagenet100_min.sh
STRATEGIES="generative trajectory" LAMBDA_MODE=equal \
  TIME=48:00:00 ./sbatch/image_cl/submit_imagenet100_min.sh
./sbatch/image_cl/submit_imagenet100_meanflow.sh

# Local / smoke:
uv run python experiments/image_cl/train_cl.py \
  --dataset split_imagenet100 --strategy cumulative --seed 42 \
  --class-conditional --unet-size large --max-grad-norm 1.0 --lr 1e-4 \
  --smoke --run-name image_cl/imagenet100/cumulative_smoke_s42

# Class-cond generative smoke (ODE, equal λ):
uv run python experiments/image_cl/train_cl.py \
  --dataset split_imagenet100 --strategy generative --seed 42 \
  --class-conditional --unet-size medium --max-grad-norm 1.0 --lr 1e-4 \
  --lambda-mode equal --smoke --max-tasks 2 \
  --run-name image_cl/imagenet100/smoke_gen_eq_s42
```

`--class-conditional` supports `naive` / `cumulative` / `generative` / `trajectory`
(ODE or MeanFlow replay).

## Quick smoke test

```bash
uv run python experiments/image_cl/train_cl.py \
  --strategy trajectory --seed 42 --smoke \
  --run-name image_cl/fmnist/smoke_traj_s42

# MeanFlow replay path (N=1 jumps; use --mean-flow-steps 4 for N=4):
uv run python experiments/image_cl/train_cl.py \
  --strategy trajectory --seed 42 --smoke \
  --replay-backend mean_flow --mean-flow-steps 1 \
  --run-name image_cl/fmnist/smoke_traj_mf_n1_s42
```

`--run-name` is optional: omitting it defaults to
`image_cl/fmnist/<strategy>[_mf][_eq]_s<seed>`. Legacy flat names like
`image_cl/trajectory_s42` are rewritten under `image_cl/fmnist/`.

## Full run (cluster)

```bash
sbatch --export=ALL,STRATEGY=trajectory,SEED=42 \
  sbatch/image_cl/train_cl.sbatch

# MeanFlow-accelerated trajectory:
sbatch --export=ALL,STRATEGY=trajectory,SEED=42,REPLAY_BACKEND=mean_flow \
  sbatch/image_cl/train_cl.sbatch

# or the 3×3 ODE grid:
./sbatch/image_cl/submit_fmnist_min.sh

# CIFAR-10 (default 9 jobs: naive/gen/traj × ODE × λ-grid × 3 seeds):
./sbatch/image_cl/submit_cifar10_min.sh
# → results_dir/image_cl/cifar10/{naive,generative,trajectory}_s{seed}/
# Expand to 15 (+ MeanFlow replay on gen/traj):
REPLAY_BACKENDS="ode mean_flow" ./sbatch/image_cl/submit_cifar10_min.sh

# ODE vs MeanFlow on trajectory × 3 seeds:
STRATEGIES="trajectory" REPLAY_BACKENDS="ode mean_flow" \
  ./sbatch/image_cl/submit_fmnist_min.sh

# Fixed equal-weight λ (no sweep), trajectory × 3 seeds:
STRATEGIES="trajectory" LAMBDA_MODE=equal SEEDS="42 69 1714" \
  ./sbatch/image_cl/submit_fmnist_min.sh
# → results_dir/image_cl/fmnist/trajectory_eq_s{seed}/

## Outputs

```
results_dir/image_cl/<dataset-slug>/<run>/
  # dataset-slug is fmnist | cifar10
  config.json
  metrics.json          # full trace + per-task test FID / KLD + timing_summary
  timing.json           # mean single-FT wall, distill walls
  lambda_trace.json     # chosen λ_k per task (for cross-seed plots)
  checkpoint.pt         # last task
  tasks/00..04/
    checkpoint.pt
    mean_flow.pt        # if --replay-backend mean_flow
    lambda_sweep.json
    metrics.json
    timing.json
  samples/taskXX_gen.pt
```

## Table audit

Run the checked-in three-seed audit before copying values into the paper:

```bash
uv run python experiments/image_cl/audit_tables.py
```

It verifies run completeness, finite final stream FID, task count, dataset/seed
provenance, and validation-selected/equal-task λ choices, then writes
`results_dir/image_cl/table_audit.json` with mean, SD, and SE and regenerates:

- `experiments/image_cl/tables/fmnist_cl_fm.tex` (FID and KLD)
- `experiments/image_cl/tables/cifar10_cl_fm.tex` (FID and KLD)
- `experiments/image_cl/tables/imagenet100_cl_fm.tex` (FID)
- `experiments/image_cl/tables/image_cl_all_datasets.tex` (paper-ready unified table)

For the existing FMNIST MeanFlow campaign, the audit intentionally selects the
corrected `*_tfix_*` runs rather than the stale unsuffixed runs.
