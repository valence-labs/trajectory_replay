import argparse
import contextlib
import logging
import os
import pathlib
import sys
import time
from collections.abc import Callable

import submitit

log = logging.getLogger(__name__)


def prepare_submission(fn: Callable) -> Callable:
    # attach a dummy checkpoint method so submitit knows that it should auto-requeue on timeout
    def checkpoint(*args, **kwargs):
        return submitit.helpers.DelayedSubmission(fn, *args, **kwargs)

    setattr(fn, "checkpoint", checkpoint)
    return fn


def launch(
    fn: Callable,
    *,
    add_args: Callable[[argparse.ArgumentParser], None] | None = None,
):
    """Thin wrapper that launches the main function with submitit."""

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("launcher")
    group.add_argument("--launcher.nodes", dest="nodes", type=int, default=1)
    group.add_argument("--launcher.gpus", dest="gpus", type=int, default=1)
    group.add_argument("--launcher.cpus", dest="cpus", type=int, default=12)
    group.add_argument("--launcher.ram", dest="ram", type=int, default=64)
    group.add_argument("--launcher.timeout", dest="timeout", type=int, default=1440)
    group.add_argument(
        "--launcher.partition",
        dest="partition",
        type=str,
        default=os.environ.get("TR_SLURM_PARTITION", "hopper"),
    )
    group.add_argument(
        "--launcher.qos",
        dest="qos",
        type=str,
        default=os.environ.get("TR_SLURM_QOS", "hooke-predict"),
    )
    group.add_argument(
        "--launcher.wckey",
        dest="wckey",
        type=str,
        default=os.environ.get("TR_SLURM_WCKEY", "hooke-predict"),
    )
    group.add_argument(
        "--launcher.output_dir", dest="output_dir", type=str, default="./outputs/"
    )
    group.add_argument(
        "--launcher.cluster",
        dest="cluster",
        choices=("debug", "local", "slurm"),
        default="debug",
    )
    group.add_argument(
        "--launcher.no_snapshot",
        dest="no_snapshot",
        action="store_true",
        help="Run directly from the working tree (development only).",
    )
    if add_args is not None:
        add_args(parser)
    args = parser.parse_args()
    output_dir = os.path.abspath(
        os.path.join(args.output_dir, time.strftime("%Y-%m-%d/%H-%M-%S", time.gmtime()))
    )
    args.launch_dir = output_dir
    executor = submitit.AutoExecutor(folder=output_dir, cluster=args.cluster)
    executor.update_parameters(
        nodes=args.nodes,
        tasks_per_node=args.gpus,  # set ntasks = ngpus
        gpus_per_node=args.gpus,
        cpus_per_task=args.cpus,
        slurm_mem_per_gpu=f"{args.ram}G",
        timeout_min=args.timeout,
        stderr_to_stdout=True,
        slurm_signal_delay_s=120,
        slurm_qos=args.qos,
        slurm_partition=args.partition,
        wckey=args.wckey,
    )

    os.makedirs(output_dir, exist_ok=True)
    launch_cmd = f"{sys.executable} {' '.join(sys.argv)}"
    with open(os.path.join(output_dir, "launch_cmd.txt"), "w") as f:
        f.write(launch_cmd)

    snapshot_dir = os.path.join(output_dir, "snapshot")
    snapshot = (
        contextlib.nullcontext()
        if args.no_snapshot
        else submitit.helpers.RsyncSnapshot(pathlib.Path(snapshot_dir))
    )
    with snapshot:
        fn = prepare_submission(fn)
        job = executor.submit(fn, args)
        log.info(f"Submitted {job=}")

        # if local or debug, wait for job to finish, otherwise exit script as soon as job is submitted
        if args.cluster == "local":
            log.info("Running job(s) locally using multiprocessing...")
            log.info(f"stdout and stderr for each process are logged to {output_dir}.")
            log.info("The job is in another process so you won't see anything here.")
            log.info("(But ctrl-c will still kill the job.)")
            _ = job.results()[0]

        elif args.cluster == "debug":
            log.info("Running job(s) in this process in debug mode...")
            log.info("pdb will open automatically on crash.")
            log.info("It's best to only use 1 GPU in this mode.")
            _ = job.results()[0]
