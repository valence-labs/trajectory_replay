# Sequential Gaussian-ring simulation

Unconditional continual-learning test bed with five isotropic Gaussian tasks.
Task 0 is the top component; subsequent tasks arrive clockwise around the ring.

The data bundle contains independent train, validation, and test samples for
each task. Validation data is reserved for selecting the replay weight in the
future continual-learning experiment.

```bash
uv run python experiments/sim_sequential/build_datasets.py
uv run python experiments/sim_sequential/plot_data.py
```

Defaults:

- 5 tasks in 2D
- ring radius 3.0
- isotropic standard deviation 0.35
- 10,000 / 2,000 / 5,000 samples per task for train / validation / test
- data seed 0

Artifacts:

```text
results_dir/sim_sequential/
├── data/dim2/
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── meta.json
└── figures/
    ├── task_ring.png
    └── task_ring.pdf
```

## Continual-learning campaign

Each method trains for 20,000 requested optimizer steps per task (20,007 actual
steps with 10,000 examples, batch 256, and `drop_last=True`). The final campaign
uses seeds 0, 1, and 2 and clips the global gradient norm at 1.0.

Replay policies:

- `constant`: fixed historical weight in `{0.25, 0.5, 0.75, 0.9}`
- `equal`: historical weight `k/(k+1)` when zero-based task `k` arrives
- `adaptive`: choose from `{0.25, 0.5, 2/3, 0.75, 0.8, 0.9}` by validation
  energy to the uniform mixture of tasks seen

The same policies are applied to generative and trajectory replay. Naive
fine-tuning has no replay weight.

```bash
DRY_RUN=1 ./sbatch/sim_sequential/submit_campaign.sh
./sbatch/sim_sequential/submit_campaign.sh
```

After all runs finish:

```bash
uv run python experiments/sim_sequential/plot_results.py
```

Evaluation records forward categorical KL from uniform task mass to generated
task mass, full-mixture energy distance, per-mode energy, latest-task fit, and
historical retention after every task. Generated masses use a Jeffreys
pseudocount of 0.5 to keep KL finite when a mode receives no samples.
