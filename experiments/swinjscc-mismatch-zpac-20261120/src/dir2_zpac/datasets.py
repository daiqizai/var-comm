"""Sealed, deterministic dataset loaders for Direction 2 on A800 PFS.

Split boundaries (strictly disjoint):
- train20k : baseline_strength_20261011/inputs/rgb/train/shard_0000..0199.pt (20,000 images)
- dev64    : baseline_strength_20261011/inputs/rgb/calibration/shard_0000.pt[:64] (64 images, Stage 0 & rapid sanity)
- val256   : baseline_strength_20261011/inputs/rgb/calibration/shard_0001..0003.pt (indices 100..355, 256 images, checkpoint selection)
- common500: baseline_strength_20261011/common500_original_sources/assets/0000..0499.npz (500 held-out test images; NEVER used for selection)
- kodak24  : baseline_strength_20261011/kodak_final_20261013_v1/crops/kodim01..24.png (24 held-out test images; NEVER used for selection)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


BASELINE_DATA_ROOT = Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011")

SplitName = Literal["train20k", "dev64", "val256", "common500", "kodak24"]
HOLDOUT_SPLITS = frozenset({"common500", "kodak24"})


@dataclass(frozen=True)
class LoadedSplit:
    name: str
    is_holdout: bool
    images_u8: torch.Tensor  # (N, 3, 256, 256), uint8
    source_ids: list[str]


def load_split_u8(
    split: SplitName,
    data_root: Path | str = BASELINE_DATA_ROOT,
    *,
    allow_holdout: bool = False,
    max_train_shards: int | None = None,
) -> LoadedSplit:
    root = Path(data_root)
    if split in HOLDOUT_SPLITS and not allow_holdout:
        raise PermissionError(
            f"Split '{split}' is a sealed held-out test set. Pass allow_holdout=True ONLY in frozen test evaluation."
        )

    if split == "train20k":
        train_dir = root / "inputs/rgb/train"
        shards = sorted(train_dir.glob("shard_*.pt"))
        if max_train_shards is not None:
            shards = shards[:max_train_shards]
        if not shards:
            raise FileNotFoundError(f"No train shards found in {train_dir}")
        tensors: list[torch.Tensor] = []
        ids: list[str] = []
        for sp in shards:
            blob = torch.load(sp, map_location="cpu")
            t = blob["targets_u8"]
            assert t.dtype == torch.uint8 and t.ndim == 4 and tuple(t.shape[1:]) == (3, 256, 256)
            tensors.append(t)
            ids.extend(str(x) for x in blob["image_ids"])
        return LoadedSplit(name=split, is_holdout=False, images_u8=torch.cat(tensors, dim=0), source_ids=ids)

    if split == "dev64":
        sp = root / "inputs/rgb/calibration/shard_0000.pt"
        blob = torch.load(sp, map_location="cpu")
        t = blob["targets_u8"][:64].clone()
        ids = [str(x) for x in blob["image_ids"][:64]]
        assert t.shape == (64, 3, 256, 256) and t.dtype == torch.uint8
        return LoadedSplit(name=split, is_holdout=False, images_u8=t, source_ids=ids)

    if split == "val256":
        cal_dir = root / "inputs/rgb/calibration"
        tensors = []
        ids = []
        for idx in (1, 2, 3):
            blob = torch.load(cal_dir / f"shard_{idx:04d}.pt", map_location="cpu")
            tensors.append(blob["targets_u8"])
            ids.extend(str(x) for x in blob["image_ids"])
        cat_t = torch.cat(tensors, dim=0)[:256].clone()
        ids = ids[:256]
        assert cat_t.shape == (256, 3, 256, 256) and cat_t.dtype == torch.uint8
        return LoadedSplit(name=split, is_holdout=False, images_u8=cat_t, source_ids=ids)

    if split == "common500":
        assets_dir = root / "common500_original_sources/assets"
        arrays = []
        ids = []
        for i in range(500):
            p = assets_dir / f"{i:04d}.npz"
            with np.load(p, allow_pickle=False) as z:
                pix = z["pixels"].copy()
            assert pix.shape == (3, 256, 256) and pix.dtype == np.uint8
            arrays.append(torch.from_numpy(pix))
            ids.append(f"common500/{i:04d}")
        return LoadedSplit(name=split, is_holdout=True, images_u8=torch.stack(arrays, dim=0), source_ids=ids)

    if split == "kodak24":
        crops_dir = root / "kodak_final_20261013_v1/crops"
        arrays = []
        ids = []
        for i in range(1, 25):
            p = crops_dir / f"kodim{i:02d}.png"
            with Image.open(p) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            assert rgb.shape == (256, 256, 3)
            chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
            arrays.append(torch.from_numpy(chw))
            ids.append(f"kodak/kodim{i:02d}")
        return LoadedSplit(name=split, is_holdout=True, images_u8=torch.stack(arrays, dim=0), source_ids=ids)

    raise ValueError(f"Unknown split: {split}")


class TensorImageDataset(Dataset):
    """Lightweight dataset converting pre-loaded uint8 CHW tensors to float32 [0, 1] on the fly."""

    def __init__(self, split_data: LoadedSplit):
        self.split_data = split_data

    def __len__(self) -> int:
        return int(self.split_data.images_u8.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.split_data.images_u8[idx].to(dtype=torch.float32).div_(255.0)
        return img, int(idx)
