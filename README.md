# Trajectory replay

Trajectory replay updates a flow-matching model on a new task by matching the historical vector field along the probability path, rather than storing past samples or treating generated endpoints as ordinary data.

This repository contains the library and the experiment entry points for the controlled simulations, class-incremental image generation, and single-cell perturbation streams.

## Install

Requires Python 3.13.

```bash
uv sync
```

Optional evaluation extras:

```bash
mkdir -p external
git clone https://github.com/valence-labs/vcb.git -b main external/vcb
uv sync --extra vcb
uv sync --extra cell-eval
```

The `vcb` extra is a path dependency, so the clone has to exist before `uv sync --extra vcb`.

## Layout

```text
src/trajectory_replay/   models, training, data, and metrics
experiments/             experiment entry points
sbatch/                  SLURM launchers
configs/                 transcriptomics run tables
tests/
notebooks/
baselines/               diffusion continual-learning baselines
third_party/             Arc STATE and SCimilarity embedding tools
```

Datasets and run outputs are not tracked. Place data under `data_dir/` and write runs to `results_dir/` at the repository root.

## Experiments

### Mixture recovery

Two-mode Gaussian mixtures. Details are in [`experiments/sim_mixture/README.md`](experiments/sim_mixture/README.md).

```bash
uv run python experiments/sim_mixture/run_top_panel.py --help
```

### Sequential Gaussian rings

Five-task ring. Details are in [`experiments/sim_sequential/README.md`](experiments/sim_sequential/README.md).

```bash
uv run python experiments/sim_sequential/build_datasets.py
uv run python experiments/sim_sequential/train_cl.py --help
```

### Class-incremental images

Split Fashion-MNIST, Split CIFAR-10, and Split ImageNet-100. Details are in [`experiments/image_cl/README.md`](experiments/image_cl/README.md).

```bash
uv run python experiments/image_cl/train_cl.py --help
```

### Single-cell perturbation streams

Replogle-Nadig and Parse PBMC continual learning. Details are in [`experiments/tx/README.md`](experiments/tx/README.md).

```bash
uv run python experiments/tx/train_cl.py --help
```

## Other tracks

- MNIST holdout: [`experiments/mnist/README.md`](experiments/mnist/README.md)
- RxRx1 phenomics: [`experiments/px/README.md`](experiments/px/README.md). Metadata defaults to `data_dir/rxrx1/rxrx1_metadata.csv`; override with `RXRX1_METADATA_PATH`.
- Simulation holdout sweeps: [`experiments/sim/README.md`](experiments/sim/README.md)

## Cluster

Scripts under `sbatch/` target our SLURM setup (`--qos=hooke-predict`, `--wckey=hooke-predict`, partitions `cpu`, `def`, or `hopper`). Change those directives before submitting on another site.

The RxRx1 submitit launcher uses the same defaults and accepts `--launcher.partition`, `--launcher.qos`, and `--launcher.wckey`. The environment variables `TR_SLURM_PARTITION`, `TR_SLURM_QOS`, and `TR_SLURM_WCKEY` override those defaults.

RxRx1 Weights & Biases projects default to `trajectory-replay-px`, `trajectory-replay-px-cl`, and `trajectory-replay-px-masip`. Set `WANDB_PROJECT` to use another project.

## Third-party code

- UNet blocks under `src/trajectory_replay/model/unet/` are adapted from [openai/guided-diffusion](https://github.com/openai/guided-diffusion).
- The RxRx1 transformer is adapted from [facebookresearch/DiT](https://github.com/facebookresearch/DiT) (MIT).
- Diffusion continual-learning baselines: [`baselines/README.md`](baselines/README.md).
- Arc STATE and SCimilarity: [`third_party/README.md`](third_party/README.md). Arc STATE model weights are released under a separate non-commercial license.

## License

Apache License 2.0. Third-party components keep their own licenses.
