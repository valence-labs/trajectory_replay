"""Unconditional 2-mode MoG continual-learning / mixture-recovery experiments.

Top panel (dim=2):
  1. ``build_datasets.py`` — task 0/1 + reference mixtures at λ ∈ {0.1,0.3,0.5,0.7,0.9}
  2. ``train_on_mixture.py`` — oracle FM trained on each mixture dataset
  3. ``train_pretrain.py`` — FM on task 0 only
  4. ``train_finetune.py`` — naive / generative / trajectory (/ real) on task 1
     - ``--replay-backend ode`` (default): instantaneous CFM only
     - ``--replay-backend mean_flow``: jointly train instantaneous CFM + dual-time
       MeanFlow (two-branch identity; old branch rolls out frozen CFM)
  5. ``train_distill_meanflow.py`` — optional post-hoc distill from a frozen CFM
     (legacy convenience; joint training above is the CL MeanFlow path)

Orchestrator: ``run_top_panel.py``.
"""
