import argparse
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DVSpeakerEvalDataset, DVSpeakerFilters, list_npy_paths
from model import EventSlowFastClassifier
from train import collate
from utils import parse_int_list, parse_sensor_size, set_seed


def build_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    state = ckpt["model"]
    num_classes = int(state["classifier.weight"].shape[0])
    model = EventSlowFastClassifier(
        num_classes=num_classes,
        slow_base=int(args.get("slow_base", 32)),
        fast_base=int(args.get("fast_base", 16)),
        fuse_mode=str(args.get("fuse_mode", "add")),
        dropout=float(args.get("dropout", 0.0)),
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def extract_embeddings(model: EventSlowFastClassifier, loader: DataLoader, device: torch.device):
    model.eval()
    embs, pids, paths = [], [], []
    for batch in tqdm(loader, desc="Embed", dynamic_ncols=True, leave=False):
        slow = batch["video"][0].to(device, non_blocking=True)
        fast = batch["video"][1].to(device, non_blocking=True)
        emb = model.forward_embed([slow, fast])
        emb = F.normalize(emb, dim=1)
        embs.append(emb.detach().cpu().numpy().astype(np.float32))
        pids.extend([int(x) for x in batch["pid"].tolist()])
        paths.extend(batch["path"])
    if len(embs) == 0:
        return np.zeros((0, model.emb_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    return np.concatenate(embs, axis=0), np.asarray(pids, dtype=np.int64), paths


@torch.no_grad()
def close_set_rank1(model, gallery_loader, probe_loader, device):
    gallery_emb, gallery_pid, gallery_paths = extract_embeddings(model, gallery_loader, device)
    probe_emb, probe_pid, probe_paths = extract_embeddings(model, probe_loader, device)

    if gallery_emb.shape[0] == 0:
        raise RuntimeError("Gallery is empty.")
    if probe_emb.shape[0] == 0:
        raise RuntimeError("Probe is empty.")

    dist = (
        np.sum(probe_emb ** 2, axis=1, keepdims=True)
        + np.sum(gallery_emb ** 2, axis=1, keepdims=True).T
        - 2.0 * np.matmul(probe_emb, gallery_emb.T)
    )
    dist = np.maximum(dist, 0.0)
    nn_index = np.argmin(dist, axis=1)
    pred_pid = gallery_pid[nn_index]
    rank1 = float(np.mean(pred_pid == probe_pid))

    return {
        "rank1": rank1,
        "num_gallery": int(gallery_emb.shape[0]),
        "num_probe": int(probe_emb.shape[0]),
        "pred_pid": pred_pid.tolist(),
        "gt_pid": probe_pid.tolist(),
        "probe_paths": probe_paths,
        "gallery_paths": gallery_paths,
        "min_distance": dist[np.arange(dist.shape[0]), nn_index].astype(np.float64).tolist(),
    }



def _same_filter(a: DVSpeakerFilters, b: DVSpeakerFilters) -> bool:
    return list(a.lights) == list(b.lights) and list(a.degrees) == list(b.degrees) and list(a.nums) == list(b.nums)



def build_cross_condition_split(
    root: str,
    test_ids: Sequence[int],
    gallery_filter: DVSpeakerFilters,
    probe_filter: DVSpeakerFilters,
    enroll_per_id: int,
    split_seed: int,
) -> Dict[str, object]:
    if enroll_per_id <= 0:
        raise ValueError(f"enroll_per_id must be > 0, got {enroll_per_id}")

    test_ids = [int(x) for x in test_ids]
    gallery_paths: List[str] = []
    probe_paths: List[str] = []
    skipped_ids: List[int] = []

    for pid in test_ids:
        gallery_pool = list_npy_paths(root, [pid], gallery_filter)
        if len(gallery_pool) < enroll_per_id:
            skipped_ids.append(pid)
            continue

        rng = random.Random(int(split_seed) + int(pid) * 97)
        picked_gallery = sorted(rng.sample(gallery_pool, enroll_per_id))
        picked_gallery_set = set(picked_gallery)

        probe_pool = list_npy_paths(root, [pid], probe_filter)
        if _same_filter(gallery_filter, probe_filter):
            probe_pool = [p for p in probe_pool if p not in picked_gallery_set]
        else:
            # Still prevent exact same file appearing in both sides if filters overlap partially.
            probe_pool = [p for p in probe_pool if p not in picked_gallery_set]

        if len(probe_pool) == 0:
            skipped_ids.append(pid)
            continue

        gallery_paths.extend(picked_gallery)
        probe_paths.extend(sorted(probe_pool))

    valid_ids = sorted(set(int(os.path.basename(os.path.dirname(p))) for p in gallery_paths))
    if len(valid_ids) == 0:
        raise RuntimeError("No valid test identities remain after applying gallery/probe filters.")

    return {
        "gallery_paths": sorted(gallery_paths),
        "probe_paths": sorted(probe_paths),
        "valid_ids": valid_ids,
        "skipped_ids": sorted(skipped_ids),
    }



def build_loader(paths, sensor_size, num_frames, alpha, log_count, batch_size, num_workers):
    ds = DVSpeakerEvalDataset(
        paths=paths,
        sensor_size=sensor_size,
        num_frames=num_frames,
        alpha=alpha,
        log_count=log_count,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SlowFast DVSpeaker model with user-controlled gallery/probe conditions.")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--test-ids", required=True, type=str, help="Comma-separated test IDs.")
    parser.add_argument("--sensor-size", default="", type=str, help="Override checkpoint sensor_size, e.g. 200,160")
    parser.add_argument("--num-frames", default=-1, type=int)
    parser.add_argument("--alpha", default=-1, type=int)
    parser.add_argument("--no-log-count", action="store_true")

    parser.add_argument("--gallery-lights", default="", type=str)
    parser.add_argument("--gallery-degrees", default="", type=str)
    parser.add_argument("--gallery-nums", default="", type=str)
    parser.add_argument("--probe-lights", default="", type=str)
    parser.add_argument("--probe-degrees", default="", type=str)
    parser.add_argument("--probe-nums", default="", type=str)

    parser.add_argument("--enroll-per-id", default=3, type=int)
    parser.add_argument("--split-seed", default=0, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--save-json", default="", type=str)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model, ckpt = build_model_from_ckpt(args.ckpt, device)
    ckpt_args = ckpt.get("args", {})

    sensor_size = parse_sensor_size(args.sensor_size or str(ckpt_args.get("sensor_size", "200,160")))
    num_frames = int(args.num_frames if args.num_frames > 0 else ckpt_args.get("num_frames", 32))
    alpha = int(args.alpha if args.alpha > 0 else ckpt_args.get("alpha", 4))
    log_count = not (args.no_log_count or bool(ckpt_args.get("no_log_count", False)))
    test_ids = parse_int_list(args.test_ids)
    if len(test_ids) == 0:
        raise ValueError("No test IDs provided. Please set --test-ids.")

    gallery_filter = DVSpeakerFilters(
        lights=parse_int_list(args.gallery_lights),
        degrees=parse_int_list(args.gallery_degrees),
        nums=parse_int_list(args.gallery_nums),
    )
    probe_filter = DVSpeakerFilters(
        lights=parse_int_list(args.probe_lights),
        degrees=parse_int_list(args.probe_degrees),
        nums=parse_int_list(args.probe_nums),
    )

    if len(gallery_filter.lights) == 0 and len(gallery_filter.degrees) == 0 and len(gallery_filter.nums) == 0:
        raise ValueError("Please specify at least one gallery filter, e.g. --gallery-lights 1 --gallery-degrees 0")
    if len(probe_filter.lights) == 0 and len(probe_filter.degrees) == 0 and len(probe_filter.nums) == 0:
        raise ValueError("Please specify at least one probe filter, e.g. --probe-lights 0 --probe-degrees 0")

    split = build_cross_condition_split(
        root=args.data_root,
        test_ids=test_ids,
        gallery_filter=gallery_filter,
        probe_filter=probe_filter,
        enroll_per_id=args.enroll_per_id,
        split_seed=args.split_seed,
    )

    gallery_loader = build_loader(
        split["gallery_paths"], sensor_size, num_frames, alpha, log_count, args.batch_size, args.num_workers
    )
    probe_loader = build_loader(
        split["probe_paths"], sensor_size, num_frames, alpha, log_count, args.batch_size, args.num_workers
    )

    metrics = close_set_rank1(model, gallery_loader, probe_loader, device)
    metrics["test_ids"] = test_ids
    metrics["valid_ids"] = split["valid_ids"]
    metrics["skipped_ids"] = split["skipped_ids"]
    metrics["gallery_filter"] = {
        "lights": list(gallery_filter.lights),
        "degrees": list(gallery_filter.degrees),
        "nums": list(gallery_filter.nums),
    }
    metrics["probe_filter"] = {
        "lights": list(probe_filter.lights),
        "degrees": list(probe_filter.degrees),
        "nums": list(probe_filter.nums),
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
