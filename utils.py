import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.backends import cudnn

DEFAULT_NUM_FRAMES = 32
DEFAULT_ALPHA = 4
DEFAULT_EVAL_CONDITIONS = "0:0,1:0,1:45,1:90"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def parse_int_list(s: Optional[str]) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    return [int(x) for x in str(s).split(",") if str(x).strip() != ""]


def parse_sensor_size(s: str) -> Tuple[int, int]:
    parts = [int(x) for x in str(s).split(",")]
    if len(parts) != 2:
        raise ValueError(f"sensor_size must be 'W,H', got: {s}")
    return int(parts[0]), int(parts[1])


def parse_conditions(s: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if s is None or str(s).strip() == "":
        return out
    for item in str(s).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Each condition must be 'light:degree', got: {item}")
        a, b = item.split(":", 1)
        out.append((int(a), int(b)))
    return out


def condition_key(light: int, degree: int) -> str:
    return f"light{int(light)}_degree{int(degree)}"


def pid_from_path(path: str) -> int:
    return int(os.path.basename(os.path.dirname(path)))


def parse_meta_from_filename(path: str) -> Dict[str, Optional[int]]:
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("_")
    out: Dict[str, Optional[int]] = {"light": None, "degree": None, "num": None}
    if len(parts) >= 1:
        try:
            out["light"] = int(parts[0])
        except Exception:
            pass
    if len(parts) >= 2:
        try:
            out["degree"] = int(parts[1])
        except Exception:
            pass
    if len(parts) >= 3:
        try:
            out["num"] = int(parts[2])
        except Exception:
            pass
    return out


def safe_struct_field(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.dtype.names is None:
        raise ValueError("Expected structured np.ndarray with fields x,y,p,t.")
    if name in arr.dtype.names:
        return arr[name]
    for n in arr.dtype.names:
        if str(n).lower() == str(name).lower():
            return arr[n]
    raise KeyError(f"Field '{name}' not found in dtype names: {arr.dtype.names}")


def load_events_xypt(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)

    if arr.dtype.names is not None:
        x = safe_struct_field(arr, "x").astype(np.int64, copy=False)
        y = safe_struct_field(arr, "y").astype(np.int64, copy=False)
        p = safe_struct_field(arr, "p").astype(np.int64, copy=False)
        t = safe_struct_field(arr, "t").astype(np.float64, copy=False)
        return np.stack([x, y, p, t], axis=1)

    if arr.ndim == 2 and arr.shape[1] >= 4:
        x = arr[:, 0].astype(np.int64, copy=False)
        y = arr[:, 1].astype(np.int64, copy=False)
        p = arr[:, 2].astype(np.int64, copy=False)
        t = arr[:, 3].astype(np.float64, copy=False)
        return np.stack([x, y, p, t], axis=1)

    raise ValueError(f"Unsupported npy format at {path}, shape={arr.shape}, dtype={arr.dtype}")
