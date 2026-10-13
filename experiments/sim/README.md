# Simulation track

Gaussian single-cell-style data and holdout/finetuning experiments.

## Scripts

| Script | Purpose |
|--------|---------|
| `train_flow_matching.py` | Noise-to-data FM on simulated data |
| `train_flow_matching_data_to_data.py` | Data-to-data FM |
| `train_perturbation_forgetting.py` | Holdout + finetune |
| `train_combo_holdout_compare.py` | Combo holdout comparisons |

Plotting (sweep / holdout / replay metrics): `plotting/`.

Data generator: `data/simulate.py`. Shared helpers: `sim_common.py`.

Notebooks: `notebooks/simulations/`.

## Quick start

```bash
python experiments/sim/train_flow_matching.py --dim 2 --epochs 200 --eval-ode --run-name demo
```

Results: `results_dir/<run-name>/` at the repo root (often a symlink).

```bash
python experiments/sim/plotting/plot_sweep_metrics.py \
  results_dir/sweep_ft_small_pert_effect \
  results_dir/sweep_ft_small_pert_effect/metrics_plots
```
