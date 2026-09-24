"""Shared PKL/NPZ motion archive loading for tracking and reference encoding."""

from pathlib import Path

import joblib
import numpy as np


def load_motion_dataset(path: Path) -> dict:
    if path.suffix.lower() != ".npz":
        return joblib.load(path)
    with np.load(path, allow_pickle=False) as data:
        if int(data["format_version"]) != 1:
            raise ValueError("unsupported motion NPZ format version")
        names = data["names"].tolist()
        fps = int(data["fps"])
        if not names or len(set(names)) != len(names) or fps <= 0:
            raise ValueError("motion NPZ requires unique names and positive fps")
        shapes = {"dof": (29,), "root_trans_offset": (3,), "root_rot": (4,), "pose_aa": (30, 3), "smpl_joints": (24, 3)}
        frames = data["dof"].shape[1]
        arrays = {}
        for key, tail in shapes.items():
            if key not in data and key in ("pose_aa", "smpl_joints"):
                continue
            array = data[key]
            if array.shape != (len(names), frames, *tail) or frames < 2 or (not np.isfinite(array).all()):
                raise ValueError(f"invalid motion NPZ array: {key}")
            arrays[key] = array
        return {
            name: {"fps": fps, **{key: value[i] for (key, value) in arrays.items()}} for (i, name) in enumerate(names)
        }
