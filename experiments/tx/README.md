# Transcriptomics experiments

The active TX pipeline covers sequential continual learning and matched
one-shot experiments on PBMC and Replogle.

## Continual learning

`train_cl.py` implements the current task streams and objectives:

- standard and gradual fine-tuning
- real, generative endpoint, and trajectory replay
- ODE and MeanFlow replay backends
- full, embedding-only, and LoRA update scopes

PBMC uses T0 pretraining donors followed by Donor1, Donor4, Donor9, and
Donor12. Replogle uses the three task-0 cell lines followed by HepG2.
Test cells are evaluation-only.

Submit campaigns from the repository root:

```bash
./sbatch/tx/submit_cl_pca100_pbmc.sh
./sbatch/tx/submit_cl_pca50.sh
```

The generic launcher is `sbatch/tx/submit_cl_campaign.sh`; training runs
through `sbatch/tx/train_cl.sbatch`.

## Evaluation

Comparable benchmark evaluation generates exactly `K=500` cells per
non-control stratum. The same fixed budget applies to Pearson Delta,
residual Pearson Delta, and CellEval. Controls may use decoded observed
control latents.

Current entrypoints:

- `eval_cl_heldout.py` — heldout performance after each sequential task
- `eval_cl_celleval.py` — post-hoc donor-aware CellEval
- `summarize_cl.py` — aggregate current CL results
- `plot_pbmc_cl_fit_retention_heldout.py` — PBMC CL overview

Final run metrics are stored in `final_metrics.json`; sequential heldout
metrics are stored in `heldout_curve.json`.

## One-shot experiments

- `train_pca_oneshot_hepg2.py`
- `train_se_oneshot_hepg2.py`
- `train_gene_oneshot_hepg2.py`

PCA one-shot jobs use `sbatch/tx/train_pca_oneshot_hepg2.sbatch`.

## Decoder and baseline utilities

- `prepare_cl_decoders.py` — construct the affine decoder bank
- `diag_decoded_geneblind_heldout.py` — Replogle heldout references
- `diag_decoded_geneblind_heldout_pbmc.py` — PBMC heldout references
- `diag_oneshot_decoder_oracle.py` — one-shot decoder ceiling
- `diag_pca_affine_bank_oracle.py` — affine-bank ceiling

Run outputs belong under `results_dir/tx_results/`, not under
`experiments/tx/`.
