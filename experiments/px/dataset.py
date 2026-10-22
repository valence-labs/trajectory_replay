import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import v2

# Metadata IDs are contiguous and zero-based.
NUM_PERTURBATIONS = 1071
NUM_EXPERIMENTS = 3
NUM_CELL_TYPES = 1
IMG_SIZE = 96
NUM_CHANNELS = 6

_REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = os.environ.get(
    "RXRX1_METADATA_PATH",
    str(_REPO_ROOT / "data" / "rxrx1" / "rxrx1_metadata.csv"),
)


def uniform_dq(img, scale=1 / 128.0):
    """Dequantize by adding uniform noise to the input.
    This is a common trick to improve performance of generative models.
    Explained well in Sec. 3.1.1. of https://arxiv.org/abs/1902.00275."""
    return img + (torch.rand_like(img) * scale)


def cp2rgb(images):
    """Convert 6-channel images to 3-channel RGB composite images.
    Input: torch.float32, shape B,6,H,W.
    Output: torch.float32, shape B,3,H,W.
    """
    # Forked from: https://github.com/yuhui-zh15/CellFlux/blob/main/training/data_utils.py
    weights = torch.tensor(
        [
            [0, 0, 1],  # Channel 1 -> Blue
            [0, 1, 0],  # Channel 2 -> Green
            [1, 0, 0],  # Channel 3 -> Red
            [0, 0.5, 0.5],  # Channel 4 -> Cyan (lower intensity)
            [0.5, 0, 0.5],  # Channel 5 -> Magenta (lower intensity)
            [0.5, 0.5, 0],  # Channel 6 -> Yellow (lower intensity)
        ],
        dtype=images.dtype,
        device=images.device,
    )
    images_rgb = torch.einsum("bchw,cn->bnhw", images, weights)
    return images_rgb.contiguous()


def encode(x):
    """torch.uint8 [0,255) -> torch.float32 [-1,1]"""
    return x.to(torch.float32) / 127.5 - 1


def decode(x):
    """torch.float32 [-1,1] -> torch.uint8 [0,255)"""
    return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)


def get_transforms() -> Callable[[torch.Tensor], torch.Tensor]:
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
        ]
    )


class CellDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata: pl.DataFrame,
        train: bool,
        load_control: bool = True,
    ):
        required_cols = [
            "path",
            "experiment_id",
            "perturbation_id",
        ]
        assert all(col in metadata.columns for col in required_cols), (
            f"metadata must have the following columns: {required_cols}"
        )
        # ctrl_metadata = metadata.filter(pl.col("ANNOT") == "DMSO")
        ctrl_metadata = metadata.filter(pl.col("ANNOT") == "negative_control")
        # pert_metadata = metadata.filter(pl.col("ANNOT") == "treated")
        pert_metadata = metadata  # allow controls to show up as perts

        self.pert_metadata = pert_metadata
        self.ctrl_subsets = {
            e: ctrl_metadata.filter(pl.col("experiment_id") == e)
            for e in list(ctrl_metadata["experiment_id"].unique())
        }

        self.transforms = get_transforms()
        self.train = train
        self.load_control = bool(load_control)

    def __len__(self):
        return len(self.pert_metadata)

    def __getitem__(self, index: int):
        row = self.pert_metadata.row(index, named=True)
        pert_path = row["path"]
        perturbation_id = row["perturbation_id"]
        experiment_id = row["experiment_id"]

        pert_array = np.load(pert_path)  # np.uint8 (96x96x6)
        pert_tensor = torch.from_numpy(pert_array).permute(2, 0, 1)  # uint8 (6x96x96)
        ctrl_tensor = None
        if self.load_control:
            ctrl_subset = self.ctrl_subsets[experiment_id]
            ctrl_idx = int(torch.randint(0, len(ctrl_subset), (1,)).item())
            ctrl_path = ctrl_subset.row(ctrl_idx, named=True)["path"]
            ctrl_array = np.load(ctrl_path)  # np.uint8 (96x96x6)
            ctrl_tensor = torch.from_numpy(ctrl_array).permute(2, 0, 1)

        with torch.inference_mode():
            pert_tensor = encode(pert_tensor)  # float32 [-1,1]
            if ctrl_tensor is not None:
                ctrl_tensor = encode(ctrl_tensor)  # float32 [-1,1]

            if self.train:
                pert_tensor = self.transforms(pert_tensor)
                pert_tensor = uniform_dq(pert_tensor)
                if ctrl_tensor is not None:
                    ctrl_tensor = self.transforms(ctrl_tensor)
                    ctrl_tensor = uniform_dq(ctrl_tensor)

        result = {
            "pert": pert_tensor,
            "perturbation_id": perturbation_id,
            "experiment_id": experiment_id,
            "cell_type_id": 0,  # all same cell type
        }
        if ctrl_tensor is not None:
            result["ctrl"] = ctrl_tensor
        return result


def get_dataloaders(
    *,
    batch_size: int = 16,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    df = pl.read_csv(METADATA_PATH)
    train_df = df.filter(pl.col("SPLIT") == "train")
    val_df = df.filter(
        (pl.col("SPLIT") == "test")
        & pl.col("is_eval_pert")
        & (pl.col("ANNOT") == "treated")
    )

    train_ds = CellDataset(train_df, train=True)
    val_ds = CellDataset(val_df, train=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader
