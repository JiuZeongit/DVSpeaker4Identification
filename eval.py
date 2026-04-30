import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import DVSpeakerEvalDataset, build_close_set_condition_splits
from model import EventSlowFastClassifier
from train import collate, evaluate_conditions
from utils import DEFAULT_EVAL_CONDITIONS, parse_conditions, parse_int_list, parse_sensor_size, set_seed


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


def build_condition_loaders(
    condition_splits,
    sensor_size,
    num_frames,
    alpha,
    log_count,
    batch_size,
    num_workers,
):
    out = {}
    for cond_key, split in condition_splits.items():
        gallery_ds = DVSpeakerEvalDataset(
            paths=split.gallery_paths,
            sensor_size=sensor_size,
            num_frames=num_frames,
            alpha=alpha,
            log_count=log_count,
        )
        probe_ds = DVSpeakerEvalDataset(
            paths=split.probe_paths,
            sensor_size=sensor_size,
            num_frames=num_frames,
            alpha=alpha,
            log_count=log_count,
        )
        out[cond_key] = (
            DataLoader(gallery_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate),
            DataLoader(probe_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate),
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SlowFast DVSpeaker model with close-set identification Rank-1.")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--test-ids", default="", type=str, help="Override checkpoint test IDs if needed.")
    parser.add_argument("--sensor-size", default="", type=str, help="Override checkpoint sensor_size, e.g. 200,160")
    parser.add_argument("--num-frames", default=-1, type=int)
    parser.add_argument("--alpha", default=-1, type=int)
    parser.add_argument("--no-log-count", action="store_true")
    parser.add_argument("--test-nums", default="", type=str)
    parser.add_argument("--eval-conditions", default="", type=str, help="Default: use checkpoint value or 0:0,1:0,1:45,1:90")
    parser.add_argument("--enroll-per-id", default=-1, type=int)
    parser.add_argument("--split-seed", default=-1, type=int)
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
    test_ids = parse_int_list(args.test_ids) if str(args.test_ids).strip() else [int(x) for x in ckpt.get("test_ids", [])]
    if len(test_ids) == 0:
        raise ValueError("No test IDs available. Provide --test-ids or use a checkpoint that saved test_ids.")
    eval_conditions = parse_conditions(args.eval_conditions or str(ckpt_args.get("eval_conditions", DEFAULT_EVAL_CONDITIONS)))
    test_nums = parse_int_list(args.test_nums)
    enroll_per_id = int(args.enroll_per_id if args.enroll_per_id > 0 else ckpt_args.get("enroll_per_id", 3))
    split_seed = int(args.split_seed if args.split_seed >= 0 else ckpt_args.get("split_seed", 0))

    condition_splits = build_close_set_condition_splits(
        root=args.data_root,
        test_ids=test_ids,
        conditions=eval_conditions,
        enroll_per_id=enroll_per_id,
        split_seed=split_seed,
        nums=test_nums,
    )
    loaders = build_condition_loaders(
        condition_splits=condition_splits,
        sensor_size=sensor_size,
        num_frames=num_frames,
        alpha=alpha,
        log_count=log_count,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    metrics = evaluate_conditions(model, loaders, device)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
