# Phenomics continual learning

The PX pipeline trains native-channel `96×96×6` flow-matching models on RxRx1.

## Baseline model

`main.py` supports the microscopy transformer variants and the Config-B
ADM U-Net:

```bash
uv run python experiments/px/main.py --help
```

The ADM defaults to approximately 56M parameters and follows the paper recipe:
noise-to-data linear flow, no minibatch OT, one-hot perturbation and experiment
conditioning, independent conditioning dropout, Adam `1e-4`, gradient clipping
`0.5`, and power-function EMA.

Cell-type conditioning is intentionally disabled: every RxRx1 crop in this
benchmark is U2OS.

## Continual-learning stream

`train_cl.py` uses the fixed experiment order:

1. `U2OS-01`
2. `U2OS-02`
3. `U2OS-03`

Each task contains the same perturbation vocabulary. A deterministic 10% split
is carved from each experiment's training rows for validation. The official
`test & is_eval_pert` subset remains reporting-only.

The locked training budget is **25,000 optimizer steps per task** with local
batch size 64 on one H100. This exposes the model to roughly 39 epochs of each
experiment. All methods use the same per-task update budget.

Implemented ODE strategies:

- `naive`
- `cumulative`
- `real`
- `generative`
- `trajectory`

Equal-task replay uses λ=1/2 at task 1 and λ=2/3 at task 2. ODE replay
uses Euler step size `0.05` (20 endpoint NFEs).
`--lambda-mode grid` selects from `{0.25, 0.5, 0.75}` using macro validation
FID across experiments seen so far; validation sampling uses a fixed Euler
budget and never reads the official test split.

ADM MeanFlow replay is available with `--replay-backend mean_flow` and
`--mean-flow-n-steps {1,4}`. Task 0 trains a data-conditioned MeanFlow; later
tasks jointly update the CFM and MeanFlow while replay uses the frozen previous
MeanFlow. Time pairs use the image-CL logit-normal schedule
(`μ=2`, `σ=2`, diagonal fraction `0.25`).

Submit a one-GPU compute-node smoke:

```bash
sbatch --export=ALL,STRATEGY=generative sbatch/px/run_cl_smoke.sbatch
```

Submit the core campaign after committing the implementation:

```bash
STRATEGIES="naive cumulative real generative trajectory" \
SEEDS="42 69 1714" \
bash sbatch/px/submit_cl_core.sh
```

Submit the MeanFlow matrix with:

```bash
bash sbatch/px/submit_cl_mean_flow.sh
```

The generic launcher creates immutable git snapshots by default. Use
`--launcher.no_snapshot` only for development runs with untracked files.

## Evaluation

Production runs enable deterministic composition-matched evaluation separately
for every experiment seen so far using the EMA model. It reports Inception
FID/KID, DINOv2 Fréchet/kernel distances, and sampler NFE. Generated and real
six-channel tensors are projected through the same fixed RGB mapping. The same
generated feature sets are concatenated to report pooled seen-experiment FID;
the final pooled score uses all `2,942` treated benchmark test images. The 120
untreated test controls are auxiliary and are not included in headline FID.

Before final reporting, reproduce the paper's composition-matched real-vs-real
FID reference (approximately `8.11`) and pin the DINOv2 source revision.

## Matched Masip diffusion baseline

`train_masip_cl.py` uses the identical ADM, experiment stream, conditions,
training budget, EMA, and evaluation pipeline, but replaces continuous flow
matching with DDIM epsilon prediction. It implements:

- diffusion fine-tuning
- cumulative diffusion training
- endpoint generative replay without distillation
- full generation distillation

Teacher replay uses 20 DDIM steps and final evaluation uses 50 DDIM steps.
Following Masip et al., replay methods use a full current batch plus a full
generated batch (32 real + 32 replay, preserving the PX total batch budget of
64), task-proportional weighting, replay coefficient 0.75, the
post-task online model as teacher, zero-terminal-SNR DDIM, and gradient clip 1.
Run `sbatch/px/submit_masip_all.sh` for the three-seed matrix. Task 0 is trained
once per seed and shared across all four methods.
