# Slurm scripts

Run all commands from the **repo root**. Logs go to `sbatch/logs/`.

```
sbatch/
  logs/          # Slurm stdout/stderr (shared)
  tx/            # PBMC/Replogle continual-learning campaigns and evaluation
  sim/           # Synthetic simulation: train, sweeps, plots
  sim_mixture/   # Unconditional MoG mixture / trajectory-replay experiments
  mnist/         # MNIST pretrain / finetune
```

## Transcriptomics (`sbatch/tx/`)

Current CL campaigns:

```bash
./sbatch/tx/submit_cl_pca100_pbmc.sh
./sbatch/tx/submit_cl_pca50.sh
```

The generic campaign wrapper is `submit_cl_campaign.sh`; jobs run through
`train_cl.sbatch`. Post-hoc evaluation uses `eval_cl_heldout.sbatch` and
`eval_cl_celleval.sbatch`.

## Simulation (`sbatch/sim/`)

| Script | Purpose |
|--------|---------|
| `submit_sweep_ft.sh` | Small perturbation-effect finetune sweep |
| `submit_sweep_ft_harder.sh` | Large perturbation-effect sweep |
| `submit_sweep_ft_replay.sh` | Replay finetune sweep |
| `submit_sweep_holdout_cmp.sh` | Holdout comparison sweep |
| `submit_sweep_holdout_control_subsample.sh` | Control subsample holdout sweep |
| `submit_plot_sweep_metrics.sh` | Sweep metric plots |
| `submit_plot_replay_seen_metrics.sh` | Replay “seen” metric plots |
| `submit_plot_holdout_cmp_sweep.sh` | Holdout comparison plots |

Simulation run outputs use `results_dir/simulation_results/<sweep_name>/...` (no tx bucket).

See `experiments/tx/README.md` for tx training details.
