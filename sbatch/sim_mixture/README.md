# sim_mixture Slurm jobs

Run from **repo root**. Logs: `sbatch/logs/`.

| Script | Purpose |
|--------|---------|
| `submit_replicates.sh` | Triplicates: **shared** pretrain + FT/oracle seeds 0/1/2 |
| `submit_pretrain.sh` | Pretrain on task 0 (usually once, seed 0) |
| `submit_ft_lambda_grid.sh` | FT grid; seeds vary FT only |
| `submit_ft_throttled.sh` | Same FT grid, ≤`MAX_CONCURRENT` jobs at a time |
| `submit_finetune.sh` | One FT job |
| `submit_oracle_mixture.sh` | FM on mixture (`ALL_LAMBDAS=1`, `SEEDS=...`) |

```bash
# FT + oracle triplicates (uses existing pretrain_task0_seed0)
./sbatch/sim_mixture/submit_replicates.sh

# Or pieces
SEEDS="0 1 2" EPOCHS=200 ./sbatch/sim_mixture/submit_ft_lambda_grid.sh
ALL_LAMBDAS=1 SEEDS="0 1 2" ./sbatch/sim_mixture/submit_oracle_mixture.sh

# Refresh shared pretrain then FT
RUN_PRETRAIN=1 ./sbatch/sim_mixture/submit_replicates.sh
```

FT/oracle run names include `_seed{S}`. Pretrain stays fixed at `pretrain_task0_seed0/`.
