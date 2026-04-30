import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    DVSpeakerEvalDataset,
    DVSpeakerTrainDataset,
    DVSpeakerFilters,
    build_close_set_condition_splits,
    build_train_paths,
)
from model import EventSlowFastClassifier
from utils import (
    DEFAULT_ALPHA,
    DEFAULT_EVAL_CONDITIONS,
    DEFAULT_NUM_FRAMES,
    condition_key,
    parse_conditions,
    parse_int_list,
    parse_sensor_size,
    set_seed,
)


def collate(batch: List[Dict]) -> Dict:
    slow = torch.stack([item["video"][0] for item in batch], dim=0)
    fast = torch.stack([item["video"][1] for item in batch], dim=0)
    out = {
        "video": [slow, fast],
        "pid": torch.stack([item["pid"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([item["label"] for item in batch], dim=0)
    return out


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
def close_set_rank1(
    model: EventSlowFastClassifier,
    gallery_loader: DataLoader,
    probe_loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    gallery_emb, gallery_pid, gallery_paths = extract_embeddings(model, gallery_loader, device)
    probe_emb, probe_pid, probe_paths = extract_embeddings(model, probe_loader, device)

    if gallery_emb.shape[0] == 0:
        raise RuntimeError("Gallery is empty.")
    if probe_emb.shape[0] == 0:
        raise RuntimeError("Probe is empty.")

    # L2 on normalized embeddings. Equivalent to cosine-based nearest neighbor ranking.
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


@torch.no_grad()
def evaluate_conditions(
    model: EventSlowFastClassifier,
    condition_loaders: Dict[str, Tuple[DataLoader, DataLoader]],
    device: torch.device,
) -> Dict[str, object]:
    per_condition: Dict[str, Dict[str, object]] = {}
    rank1_list: List[float] = []
    for cond_key, (gallery_loader, probe_loader) in condition_loaders.items():
        result = close_set_rank1(model, gallery_loader, probe_loader, device)
        per_condition[cond_key] = result
        rank1_list.append(float(result["rank1"]))
    avg_rank1 = float(np.mean(rank1_list)) if len(rank1_list) > 0 else float("nan")
    return {
        "average_rank1": avg_rank1,
        "per_condition": per_condition,
    }


def train_one_epoch(
    model: EventSlowFastClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    bar = tqdm(loader, desc=f"Train {epoch}/{epochs}", dynamic_ncols=True)
    for batch in bar:
        slow = batch["video"][0].to(device, non_blocking=True)
        fast = batch["video"][1].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model([slow, fast])
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        bs = label.size(0)
        total_num += bs
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == label).sum().item())

        bar.set_postfix(loss=total_loss / max(1, total_num), acc=total_correct / max(1, total_num))

    return {
        "loss": total_loss / max(1, total_num),
        "acc": total_correct / max(1, total_num),
    }


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
        gallery_loader = DataLoader(
            gallery_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate,
        )
        probe_loader = DataLoader(
            probe_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate,
        )
        out[cond_key] = (gallery_loader, probe_loader)
    return out


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SlowFast on DVSpeaker with close-set identification evaluation.")

    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--sensor-size", default="200,160", type=str)
    parser.add_argument("--num-frames", default=DEFAULT_NUM_FRAMES, type=int)
    parser.add_argument("--alpha", default=DEFAULT_ALPHA, type=int)
    parser.add_argument("--no-log-count", action="store_true")

    parser.add_argument("--train-ids", required=True, type=str, help="Comma-separated IDs used for training classes.")
    parser.add_argument("--test-ids", required=True, type=str, help="Comma-separated IDs used for close-set test gallery/probe.")
    parser.add_argument("--enroll-per-id", default=3, type=int, help="Gallery samples per identity in each test condition.")
    parser.add_argument("--split-seed", default=0, type=int)

    parser.add_argument("--train-lights", default="", type=str)
    parser.add_argument("--train-degrees", default="", type=str)
    parser.add_argument("--train-nums", default="", type=str)
    parser.add_argument("--test-nums", default="", type=str, help="Optional nums filter applied to all four eval conditions.")
    parser.add_argument("--eval-conditions", default=DEFAULT_EVAL_CONDITIONS, type=str, help="Format: light:degree,light:degree,...")

    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--save-root", default="result_dvspeaker_slowfast_close_set", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--slow-base", default=32, type=int)
    parser.add_argument("--fast-base", default=16, type=int)
    parser.add_argument("--fuse-mode", default="add", choices=["add", "concat"], type=str)
    parser.add_argument("--dropout", default=0.0, type=float)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    sensor_size = parse_sensor_size(args.sensor_size)
    train_ids = parse_int_list(args.train_ids)
    test_ids = parse_int_list(args.test_ids)
    train_filters = DVSpeakerFilters(
        lights=parse_int_list(args.train_lights),
        degrees=parse_int_list(args.train_degrees),
        nums=parse_int_list(args.train_nums),
    )
    test_nums = parse_int_list(args.test_nums)
    eval_conditions = parse_conditions(args.eval_conditions)
    log_count = not bool(args.no_log_count)

    overlap = sorted(set(train_ids).intersection(set(test_ids)))
    if overlap:
        raise ValueError(f"train_ids and test_ids must be disjoint for this protocol, but overlap = {overlap}")

    train_paths = build_train_paths(args.data_root, train_ids, train_filters)
    if len(train_paths) == 0:
        raise RuntimeError("No training samples found. Please check --data-root and train filters.")

    condition_splits = build_close_set_condition_splits(
        root=args.data_root,
        test_ids=test_ids,
        conditions=eval_conditions,
        enroll_per_id=args.enroll_per_id,
        split_seed=args.split_seed,
        nums=test_nums,
    )

    print(f"[Data] train_ids={len(train_ids)} test_ids={len(test_ids)} train_samples={len(train_paths)}")
    print(f"[Input] sensor_size={sensor_size} T_fast={args.num_frames} T_slow={max(1, args.num_frames // args.alpha)} alpha={args.alpha} log_count={log_count}")
    print(f"[Model] SlowFast only, CE only, slow_base={args.slow_base}, fast_base={args.fast_base}, fuse_mode={args.fuse_mode}, dropout={args.dropout}")
    print(f"[Eval] close-set identification, enroll_per_id={args.enroll_per_id}, conditions={[condition_key(l, d) for (l, d) in eval_conditions]}")
    for key, split in condition_splits.items():
        print(f"  - {key}: gallery={len(split.gallery_paths)} probe={len(split.probe_paths)}")

    train_ds = DVSpeakerTrainDataset(
        paths=train_paths,
        train_ids=train_ids,
        sensor_size=sensor_size,
        num_frames=args.num_frames,
        alpha=args.alpha,
        log_count=log_count,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    condition_loaders = build_condition_loaders(
        condition_splits=condition_splits,
        sensor_size=sensor_size,
        num_frames=args.num_frames,
        alpha=args.alpha,
        log_count=log_count,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = EventSlowFastClassifier(
        num_classes=len(train_ids),
        slow_base=args.slow_base,
        fast_base=args.fast_base,
        fuse_mode=args.fuse_mode,
        dropout=args.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_root, exist_ok=True)
    history = []
    best_avg_rank1 = -1.0
    best_epoch = -1

    args_to_save = vars(args).copy()
    args_to_save["sensor_size"] = args.sensor_size
    args_to_save["eval_conditions"] = args.eval_conditions

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        eval_stats = evaluate_conditions(model, condition_loaders, device)

        row = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "train_acc": float(train_stats["acc"]),
            "average_rank1": float(eval_stats["average_rank1"]),
            "per_condition_rank1": {
                cond_key: float(cond_val["rank1"])
                for cond_key, cond_val in eval_stats["per_condition"].items()
            },
        }
        history.append(row)

        print(
            f"[Epoch {epoch}] loss={row['train_loss']:.6f} acc={row['train_acc']:.4f} avg_rank1={row['average_rank1']:.4f} "
            + " ".join([f"{k}={v:.4f}" for k, v in row["per_condition_rank1"].items()])
        )

        last_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": args_to_save,
            "train_ids": train_ids,
            "test_ids": test_ids,
            "history": history,
            "metrics": eval_stats,
        }
        torch.save(last_ckpt, os.path.join(args.save_root, "last.pt"))

        if row["average_rank1"] > best_avg_rank1:
            best_avg_rank1 = row["average_rank1"]
            best_epoch = epoch
            best_ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args_to_save,
                "train_ids": train_ids,
                "test_ids": test_ids,
                "history": history,
                "metrics": eval_stats,
                "best_average_rank1": best_avg_rank1,
            }
            torch.save(best_ckpt, os.path.join(args.save_root, "best_avg_rank1.pt"))
            save_json(os.path.join(args.save_root, "best_metrics.json"), eval_stats)

        save_json(os.path.join(args.save_root, "history.json"), {"history": history, "best_epoch": best_epoch, "best_average_rank1": best_avg_rank1})

    print(f"[Done] best_epoch={best_epoch} best_average_rank1={best_avg_rank1:.4f}")


if __name__ == "__main__":
    main()
