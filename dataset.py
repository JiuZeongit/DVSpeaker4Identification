import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_events_xypt, pid_from_path

_FILENAME_RE = re.compile(r"^(?P<light>[01])_(?P<degree>0|45|90)_(?P<num>[0-9])_.*\.npy$")


@dataclass(frozen=True)
class DVSpeakerFilters:
    lights: Sequence[int]
    degrees: Sequence[int]
    nums: Sequence[int]

    def accept(self, light: int, degree: int, num: int) -> bool:
        if self.lights and light not in self.lights:
            return False
        if self.degrees and degree not in self.degrees:
            return False
        if self.nums and num not in self.nums:
            return False
        return True


@dataclass(frozen=True)
class ConditionSplit:
    light: int
    degree: int
    gallery_paths: List[str]
    probe_paths: List[str]

    @property
    def key(self) -> str:
        return f"light{self.light}_degree{self.degree}"


def list_npy_paths(root: str, person_ids: Sequence[int], flt: DVSpeakerFilters) -> List[str]:
    paths: List[str] = []
    for pid in person_ids:
        pid_dir = os.path.join(root, str(pid))
        if not os.path.isdir(pid_dir):
            continue
        for fn in os.listdir(pid_dir):
            m = _FILENAME_RE.match(fn)
            if m is None:
                continue
            light = int(m.group("light"))
            degree = int(m.group("degree"))
            num = int(m.group("num"))
            if flt.accept(light, degree, num):
                paths.append(os.path.join(pid_dir, fn))
    paths.sort()
    return paths


def events_to_count_volume(
    ev_xypt: np.ndarray,
    sensor_size: Tuple[int, int],
    num_bins: int,
    log_count: bool = True,
) -> torch.Tensor:
    w, h = sensor_size
    t_bins = int(num_bins)
    if t_bins <= 0:
        raise ValueError(f"num_bins must be > 0, got {num_bins}")

    vol = np.zeros((2, t_bins, h, w), dtype=np.float32)
    if ev_xypt.size == 0:
        return torch.from_numpy(vol)

    x = ev_xypt[:, 0].astype(np.int64, copy=False)
    y = ev_xypt[:, 1].astype(np.int64, copy=False)
    p = ev_xypt[:, 2].astype(np.int64, copy=False)
    t = ev_xypt[:, 3].astype(np.float64, copy=False)

    x = x - int(x.min()) if x.size > 0 else x
    y = y - int(y.min()) if y.size > 0 else y
    t = t - float(t.min()) if t.size > 0 else t
    p = (p > 0).astype(np.int64, copy=False)

    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not np.any(valid):
        return torch.from_numpy(vol)

    x, y, p, t = x[valid], y[valid], p[valid], t[valid]

    if t.size == 0:
        return torch.from_numpy(vol)

    t_max = float(t.max())
    if t_max <= 0:
        tb = np.zeros_like(t, dtype=np.int64)
    else:
        tb = np.floor((t / t_max) * t_bins).astype(np.int64)
        tb = np.clip(tb, 0, t_bins - 1)

    np.add.at(vol, (p, tb, y, x), 1.0)

    if log_count:
        np.log1p(vol, out=vol)

    return torch.from_numpy(vol)


class DVSpeakerTrainDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[str],
        train_ids: Sequence[int],
        sensor_size: Tuple[int, int],
        num_frames: int,
        alpha: int,
        log_count: bool = True,
    ):
        self.paths = list(paths)
        self.sensor_size = tuple(sensor_size)
        self.fast_num_frames = int(num_frames)
        self.alpha = int(alpha)
        self.slow_num_frames = max(1, self.fast_num_frames // self.alpha)
        self.log_count = bool(log_count)
        self.pid2label = {int(pid): i for i, pid in enumerate(sorted([int(x) for x in train_ids]))}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.paths[idx]
        pid = pid_from_path(path)
        label = self.pid2label[pid]
        ev = load_events_xypt(path)
        slow = events_to_count_volume(ev, self.sensor_size, self.slow_num_frames, self.log_count)
        fast = events_to_count_volume(ev, self.sensor_size, self.fast_num_frames, self.log_count)
        return {
            "video": [slow, fast],
            "label": torch.tensor(label, dtype=torch.long),
            "pid": torch.tensor(pid, dtype=torch.long),
            "path": path,
        }


class DVSpeakerEvalDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[str],
        sensor_size: Tuple[int, int],
        num_frames: int,
        alpha: int,
        log_count: bool = True,
    ):
        self.paths = list(paths)
        self.sensor_size = tuple(sensor_size)
        self.fast_num_frames = int(num_frames)
        self.alpha = int(alpha)
        self.slow_num_frames = max(1, self.fast_num_frames // self.alpha)
        self.log_count = bool(log_count)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.paths[idx]
        pid = pid_from_path(path)
        ev = load_events_xypt(path)
        slow = events_to_count_volume(ev, self.sensor_size, self.slow_num_frames, self.log_count)
        fast = events_to_count_volume(ev, self.sensor_size, self.fast_num_frames, self.log_count)
        return {
            "video": [slow, fast],
            "pid": torch.tensor(pid, dtype=torch.long),
            "path": path,
        }


def build_train_paths(
    root: str,
    train_ids: Sequence[int],
    train_filters: DVSpeakerFilters,
) -> List[str]:
    return list_npy_paths(root, train_ids, train_filters)


def build_close_set_condition_splits(
    root: str,
    test_ids: Sequence[int],
    conditions: Sequence[Tuple[int, int]],
    enroll_per_id: int,
    split_seed: int = 0,
    nums: Optional[Sequence[int]] = None,
) -> Dict[str, ConditionSplit]:
    if enroll_per_id <= 0:
        raise ValueError(f"enroll_per_id must be > 0, got {enroll_per_id}")

    test_ids = [int(x) for x in test_ids]
    nums = [] if nums is None else [int(x) for x in nums]
    out: Dict[str, ConditionSplit] = {}

    for cond_index, (light, degree) in enumerate(conditions):
        gallery_paths: List[str] = []
        probe_paths: List[str] = []

        for pid in test_ids:
            flt = DVSpeakerFilters(lights=[int(light)], degrees=[int(degree)], nums=nums)
            pool = list_npy_paths(root, [pid], flt)
            if len(pool) < enroll_per_id + 1:
                raise ValueError(
                    f"pid={pid} condition=(light={light}, degree={degree}) has only {len(pool)} samples, "
                    f"but close-set evaluation needs at least enroll_per_id+1={enroll_per_id + 1}."
                )

            rng = random.Random(int(split_seed) + cond_index * 10007 + int(pid) * 97)
            picked = sorted(rng.sample(pool, enroll_per_id))
            picked_set = set(picked)
            remain = [p for p in pool if p not in picked_set]
            if len(remain) == 0:
                raise ValueError(
                    f"pid={pid} condition=(light={light}, degree={degree}) has no probe samples left after enrollment."
                )
            gallery_paths.extend(picked)
            probe_paths.extend(remain)

        split = ConditionSplit(
            light=int(light),
            degree=int(degree),
            gallery_paths=sorted(gallery_paths),
            probe_paths=sorted(probe_paths),
        )
        out[split.key] = split

    return out
