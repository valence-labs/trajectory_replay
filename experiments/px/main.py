import argparse
import logging
import os
import pathlib

import submitit
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

import experiments.px.dataset as ds
from experiments.px.model import get_model_cls
from experiments.px.trainer import TrainState, train
from experiments.px.utils.distributed import Distributed
from experiments.px.utils.ema import KarrasEMA
from experiments.px.utils.launcher import launch
from experiments.px.utils.name_run import generate_random_name

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_NAMES = (
    "ADM",
    "DiT-S/2",
    "DiT-S/4",
    "DiT-S/8",
    "DiT-B/2",
    "DiT-B/4",
    "DiT-B/8",
    "DiT-L/2",
    "DiT-L/4",
    "DiT-L/8",
    "DiT-XL/2",
    "DiT-XL/4",
    "DiT-XL/8",
)


def add_px_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("phenomics")
    group.add_argument(
        "--model-name",
        choices=MODEL_NAMES,
        default="ADM",
        help="Model backend. ADM is the paper's Config-B U-Net.",
    )
    group.add_argument("--batch-size", type=int, default=16)
    group.add_argument("--num-steps", type=int, default=100_000)
    group.add_argument(
        "--eval-every-n-steps",
        type=int,
        default=5_000,
        help="Validation/sample interval; 0 disables it.",
    )
    group.add_argument(
        "--checkpoint-every-n-steps",
        type=int,
        default=25_000,
        help="Checkpoint interval; 0 disables it.",
    )
    group.add_argument(
        "--metrics-every-n-steps",
        type=int,
        default=25_000,
        help="FID/KID/FD evaluation interval; 0 disables it.",
    )


def prng(rank: int, seed: int = 42):
    local_seed = seed + rank
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    log.info(f"Rank {rank} setting seed to {local_seed}")


def main(args: argparse.Namespace):
    if args.batch_size < 1 or args.num_steps < 1:
        raise ValueError("--batch-size and --num-steps must be positive")
    for name in (
        "eval_every_n_steps",
        "checkpoint_every_n_steps",
        "metrics_every_n_steps",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")

    with Distributed() as D:
        job_env = submitit.JobEnvironment()

        output_dir = str(pathlib.Path(args.launch_dir) / job_env.job_id)
        name = generate_random_name(output_dir)

        if D.rank == 0:
            wandb.init(
                name=name,
                id=name,
                dir=output_dir,
                notes=f"Outputs saved to: {output_dir}",
                project=os.environ.get("WANDB_PROJECT", "trajectory-replay-px"),
                resume="allow",
                save_code=False,
                config=vars(args),
            )

        log.info(f"Running job ID {job_env.job_id}")
        log.info(f"{output_dir=}")
        log.info(f"{name=}")
        log.info(
            "Phenotype config: model=%s batch=%d steps=%d eval=%d ckpt=%d metrics=%d",
            args.model_name,
            args.batch_size,
            args.num_steps,
            args.eval_every_n_steps,
            args.checkpoint_every_n_steps,
            args.metrics_every_n_steps,
        )

        prng(D.rank)
        torch.set_float32_matmul_precision("medium")

        model_cls = get_model_cls(name=args.model_name)
        net = model_cls(
            in_channels=ds.NUM_CHANNELS,
            input_size=ds.IMG_SIZE,
            y_dim=ds.NUM_PERTURBATIONS,
            e_dim=ds.NUM_EXPERIMENTS,
            c_dim=ds.NUM_CELL_TYPES,
        )
        n_params = sum(parameter.numel() for parameter in net.parameters())
        log.info("Model parameters: %.2fM", n_params / 1e6)
        net.to(D.device)
        net.compile()
        ddp = DDP(net)
        ema = KarrasEMA(net)
        opt = torch.optim.Adam(
            net.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        state = TrainState(ddp=ddp, ema=ema, opt=opt, global_step=0)

        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info(f"Using checkpoint directory: {ckpt_dir}")
        state.load_latest_ckpt(dir=ckpt_dir, device=D.device)

        train_loader, iid_loader = ds.get_dataloaders(
            batch_size=args.batch_size
        )
        train(
            state=state,
            train_loader=train_loader,
            val_loader=iid_loader,
            test_loaders={"iid": iid_loader},
            output_dir=output_dir,
            D=D,
            num_steps=args.num_steps,
            eval_every_n_steps=args.eval_every_n_steps,
            ckpt_every_n_steps=args.checkpoint_every_n_steps,
            metrics_every_n_steps=args.metrics_every_n_steps,
        )

        if D.rank == 0:
            wandb.finish()


if __name__ == "__main__":
    launch(main, add_args=add_px_args)
