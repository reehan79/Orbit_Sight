from __future__ import annotations

"""Frozen tiny foveated neural refiner CV (Training_sets only)."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from orbitsight.evaluation.detection_aggregate import aggregate_detection_metrics
from orbitsight.evaluation.gt_assignment import compatible
from orbitsight.features import (
    extract_candidate_features,
    extract_local_geometry_features,
    rasterize_event_patch,
    refine_c1_centroid,
    refine_c4_median,
)
from orbitsight.io import Detection, read_detection_file
from orbitsight.models import TinyFoveatedRefiner, fit_rankers, parameter_count, score_ranker
from orbitsight.proposals import Candidate, RawGridProposer, infer_sensor_geometry

PRIOR_MS = 80
TOP_K = 20
RANKER = "M2b_extra_trees"
EPOCHS = 12
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
BBOX_LOSS_WEIGHT = 2.0


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sensor_name(sequence: str) -> str:
    u = sequence.upper()
    if u.startswith("DAVIS"):
        return "DAVIS"
    if u.startswith("DVX"):
        return "DVX"
    return "EVK4"


def iou_box(cx: float, cy: float, w: float, h: float, gt: Detection) -> float:
    ax1, ay1 = cx - w / 2.0, cy - h / 2.0
    ax2, ay2 = cx + w / 2.0, cy + h / 2.0
    bx1, by1 = gt.cx - gt.width / 2.0, gt.cy - gt.height / 2.0
    bx2, by2 = gt.cx + gt.width / 2.0, gt.cy + gt.height / 2.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = w * h + gt.width * gt.height - inter
    return inter / union if union > 0 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_table(path: Path) -> dict[str, np.ndarray]:
    from orbitsight.features import FEATURE_NAMES

    sequences, ranks, targets = [], [], []
    features, bbox_targets = [], []
    starts, ends = [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sequences.append(row["sequence"])
            starts.append(int(row["window_start_us"]))
            ends.append(int(row["window_end_us"]))
            ranks.append(int(row["candidate_rank"]))
            targets.append(int(row["target"]))
            features.append([float(row[n]) for n in FEATURE_NAMES])
            if row["bbox_log_w_cells"] == "":
                bbox_targets.append([np.nan, np.nan])
            else:
                bbox_targets.append([float(row["bbox_log_w_cells"]), float(row["bbox_log_h_cells"])])
    return {
        "sequence": np.asarray(sequences, dtype=object),
        "start": np.asarray(starts, dtype=np.int64),
        "end": np.asarray(ends, dtype=np.int64),
        "rank": np.asarray(ranks, dtype=np.int16),
        "target": np.asarray(targets, dtype=np.int8),
        "X": np.asarray(features, dtype=np.float32),
        "bbox_log_wh": np.asarray(bbox_targets, dtype=np.float32),
    }


def load_cache(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


class PatchCacheDataset(Dataset):
    def __init__(self, cache: dict[str, np.ndarray], indices: np.ndarray):
        self.cache = cache
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        patch = torch.from_numpy(self.cache["patches"][idx].astype(np.float32))
        feats = torch.from_numpy(self.cache["features"][idx].astype(np.float32))
        cls = torch.tensor(float(self.cache["cls_target"][idx]), dtype=torch.float32)
        bbox = torch.from_numpy(self.cache["bbox_target"][idx].astype(np.float32))
        return patch, feats, cls, bbox


def fit_size_extratrees(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(
        n_estimators=32,
        max_depth=12,
        min_samples_leaf=24,
        max_features=None,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def build_s2_training(table: dict[str, np.ndarray], train_idx: np.ndarray, split_dir: Path):
    X_rows: list[np.ndarray] = []
    y_rows: list[list[float]] = []
    pos_idx = train_idx[table["target"][train_idx] == 1]
    by_window: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for idx in pos_idx:
        by_window[(str(table["sequence"][idx]), int(table["start"][idx]), int(table["end"][idx]))].append(int(idx))
    for (sequence, start, end), indices in by_window.items():
        npy_path = split_dir / f"{sequence}_labeled_events.npy"
        arr = np.load(npy_path, mmap_mode="r")
        timestamps = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, end, side="left"))
        current = np.asarray(arr[left:right, :4])
        proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
        candidates = proposer.propose(current)
        if not candidates:
            continue
        for idx in indices:
            rank = int(table["rank"][idx])
            if rank < 1 or rank > len(candidates):
                continue
            candidate = candidates[rank - 1]
            feat15 = table["X"][idx]
            rcx, rcy = refine_c1_centroid(current, candidate.cx, candidate.cy, cell)
            local18 = extract_local_geometry_features(current, rcx, rcy, cell, width, height)
            X_rows.append(np.concatenate([feat15, local18]))
            y_rows.append(table["bbox_log_wh"][idx].tolist())
    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def train_refiner(cache: dict[str, np.ndarray], train_sequences: set[str], device: torch.device) -> TinyFoveatedRefiner:
    set_seed(SEED)
    seq = cache["sequence"]
    mask = np.array([str(s) in train_sequences for s in seq], dtype=bool)
    indices = np.flatnonzero(mask)
    cls = cache["cls_target"][indices]
    n_pos = max(int(cls.sum()), 1)
    n_neg = max(int(len(cls) - cls.sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    loader = DataLoader(
        PatchCacheDataset(cache, indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    model = TinyFoveatedRefiner().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    smooth = nn.SmoothL1Loss()

    model.train()
    for _epoch in range(EPOCHS):
        for patch, feats, y_cls, y_bbox in loader:
            patch = patch.to(device)
            feats = feats.to(device)
            y_cls = y_cls.to(device)
            y_bbox = y_bbox.to(device)
            logits, bbox_pred = model(patch, feats)
            cls_loss = bce(logits, y_cls)
            pos = y_cls > 0.5
            if pos.any():
                bbox_loss = smooth(bbox_pred[pos], y_bbox[pos])
            else:
                bbox_loss = torch.zeros((), device=device)
            loss = cls_loss + BBOX_LOSS_WEIGHT * bbox_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
    model.eval()
    return model


def export_onnx(model: TinyFoveatedRefiner, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = TinyFoveatedRefiner()
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()
    patch = torch.zeros(1, 4, 32, 32)
    feats = torch.zeros(1, 15)

    class Wrapper(nn.Module):
        def __init__(self, net: TinyFoveatedRefiner):
            super().__init__()
            self.net = net

        def forward(self, p, f):
            cls, bbox = self.net(p, f)
            return cls, bbox

    torch.onnx.export(
        Wrapper(model_cpu),
        (patch, feats),
        str(path),
        input_names=["patch", "features"],
        output_names=["cls_logit", "bbox"],
        dynamic_axes={"patch": {0: "batch"}, "features": {0: "batch"}, "cls_logit": {0: "batch"}, "bbox": {0: "batch"}},
        opset_version=17,
    )


def onnx_session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def onnx_infer(session, patches: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    outs = session.run(
        None,
        {
            "patch": patches.astype(np.float32),
            "features": features.astype(np.float32),
        },
    )
    return outs[0], outs[1]


def decode_bbox(candidate: Candidate, bbox: np.ndarray, cell: float) -> tuple[float, float, float, float]:
    dx, dy, log_w, log_h = [float(v) for v in bbox]
    return (
        candidate.cx + dx * cell,
        candidate.cy + dy * cell,
        math.exp(log_w) * cell,
        math.exp(log_h) * cell,
    )


def evaluate_configs_for_fold(
    fold_id: int,
    train_sequences: set[str],
    val_sequences: set[str],
    table: dict[str, np.ndarray],
    split_dir: Path,
    cache: dict[str, np.ndarray],
    device: torch.device,
    onnx_dir: Path,
    collect_latency: bool,
) -> tuple[dict[str, list[dict]], list[dict], TinyFoveatedRefiner]:
    train_idx = np.flatnonzero(np.isin(table["sequence"], list(train_sequences)))
    rankers = fit_rankers(
        table["X"][train_idx],
        table["target"][train_idx],
        table["rank"][train_idx],
        model_names=[RANKER],
    )
    bundle = rankers[RANKER]
    size_X, size_y = build_s2_training(table, train_idx, split_dir)
    size_trees = fit_size_extratrees(size_X, size_y)

    model = train_refiner(cache, train_sequences, device)
    onnx_path = onnx_dir / f"fold{fold_id}_refiner.onnx"
    export_onnx(model, onnx_path)
    session = onnx_session(onnx_path)

    details: dict[str, list[dict]] = defaultdict(list)
    latency_rows: list[dict] = []
    times: dict[str, list[float]] = defaultdict(list)
    times_sensor: dict[tuple[str, str], list[float]] = defaultdict(list)

    for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
        sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
        if sequence not in val_sequences:
            continue
        sensor = sensor_name(sequence)
        npy_path = split_dir / f"{sequence}_labeled_events.npy"
        arr = np.load(npy_path, mmap_mode="r")
        timestamps = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
        grouped: dict[tuple[int, int], list[Detection]] = defaultdict(list)
        for gt in read_detection_file(gt_path):
            grouped[(gt.start_us, gt.end_us)].append(gt)

        for (start_us, end_us), gts in sorted(grouped.items()):
            left = int(np.searchsorted(timestamps, start_us, side="left"))
            right = int(np.searchsorted(timestamps, end_us, side="left"))
            prior_left = int(np.searchsorted(timestamps, start_us - PRIOR_MS * 1000, side="left"))
            current = np.asarray(arr[left:right, :4])
            prior = np.asarray(arr[prior_left:left, :4])

            # Shared proposal/feature/ranker outputs (recomputed inside timed paths below).
            candidates = proposer.propose(current)
            if not candidates:
                for label in ("B_CURRENT", "N1_TOP1_BOX", "N2_TOP3_JOINT"):
                    for gt in gts:
                        details[label].append(
                            {
                                "fold": fold_id,
                                "sequence": sequence,
                                "sensor": sensor,
                                "config": label,
                                "proposal_hit": False,
                                "ranker_top1_hit": False,
                                "ranker_top3_hit": False,
                                "neural_selected_hit": False,
                                "centre_error": float("nan"),
                                "iou": 0.0,
                                "iou50": 0.0,
                                "iou75": 0.0,
                            }
                        )
                continue

            features = extract_candidate_features(current, prior, candidates, width, height, cell)
            ranks = np.arange(1, len(candidates) + 1, dtype=np.int16)
            scores = score_ranker(bundle, features, ranks)
            order = np.argsort(-scores)
            top1 = candidates[int(order[0])]
            top3 = [candidates[int(i)] for i in order[: min(3, len(candidates))]]

            # B_CURRENT boxes (C4 centre + S2 size with C1 features)
            b_cx, b_cy = refine_c4_median(current, top1.cx, top1.cy, cell)
            c1_cx, c1_cy = refine_c1_centroid(current, top1.cx, top1.cy, cell)
            local18 = extract_local_geometry_features(current, c1_cx, c1_cy, cell, width, height)
            f15 = features[candidates.index(top1)]
            log_wh = size_trees.predict(np.concatenate([f15, local18]).reshape(1, -1))[0]
            b_w = math.exp(float(log_wh[0])) * cell
            b_h = math.exp(float(log_wh[1])) * cell

            # N1 neural bbox on Top-1
            p1 = rasterize_event_patch(current, prior, top1.cx, top1.cy, float(cell), start_us, end_us)
            f1 = features[candidates.index(top1)].astype(np.float32)
            _, bbox1 = onnx_infer(session, p1[None, ...].astype(np.float32), f1[None, ...])
            n1_cx, n1_cy, n1_w, n1_h = decode_bbox(top1, bbox1[0], float(cell))

            # N2 neural batch Top-3
            patches = [
                rasterize_event_patch(current, prior, c.cx, c.cy, float(cell), start_us, end_us) for c in top3
            ]
            feats3 = [features[candidates.index(c)].astype(np.float32) for c in top3]
            cls_logits, bboxes = onnx_infer(
                session, np.stack(patches).astype(np.float32), np.stack(feats3).astype(np.float32)
            )
            pick = int(np.argmax(cls_logits))
            n2_sel = top3[pick]
            n2_cx, n2_cy, n2_w, n2_h = decode_bbox(n2_sel, bboxes[pick], float(cell))

            if collect_latency:
                # Complete inference paths (exclude training)
                t0 = perf_counter_ns()
                cands = proposer.propose(current)
                feats = extract_candidate_features(current, prior, cands, width, height, cell)
                sc = score_ranker(bundle, feats, np.arange(1, len(cands) + 1, dtype=np.int16))
                sel = cands[int(np.argmax(sc))]
                _cx, _cy = refine_c4_median(current, sel.cx, sel.cy, cell)
                _c1x, _c1y = refine_c1_centroid(current, sel.cx, sel.cy, cell)
                _loc = extract_local_geometry_features(current, _c1x, _c1y, cell, width, height)
                _ = size_trees.predict(
                    np.concatenate([feats[cands.index(sel)], _loc]).reshape(1, -1)
                )
                times["B_CURRENT"].append((perf_counter_ns() - t0) / 1e6)
                times_sensor[("B_CURRENT", sensor)].append(times["B_CURRENT"][-1])

                t0 = perf_counter_ns()
                cands = proposer.propose(current)
                feats = extract_candidate_features(current, prior, cands, width, height, cell)
                sc = score_ranker(bundle, feats, np.arange(1, len(cands) + 1, dtype=np.int16))
                ord_ = np.argsort(-sc)
                sel = cands[int(ord_[0])]
                patch = rasterize_event_patch(current, prior, sel.cx, sel.cy, float(cell), start_us, end_us)
                onnx_infer(
                    session,
                    patch[None, ...].astype(np.float32),
                    feats[cands.index(sel)][None, ...].astype(np.float32),
                )
                times["N1_TOP1_BOX"].append((perf_counter_ns() - t0) / 1e6)
                times_sensor[("N1_TOP1_BOX", sensor)].append(times["N1_TOP1_BOX"][-1])

                t0 = perf_counter_ns()
                cands = proposer.propose(current)
                feats = extract_candidate_features(current, prior, cands, width, height, cell)
                sc = score_ranker(bundle, feats, np.arange(1, len(cands) + 1, dtype=np.int16))
                ord_ = np.argsort(-sc)
                top = [cands[int(i)] for i in ord_[: min(3, len(cands))]]
                ps = [
                    rasterize_event_patch(current, prior, c.cx, c.cy, float(cell), start_us, end_us) for c in top
                ]
                fs = [feats[cands.index(c)].astype(np.float32) for c in top]
                onnx_infer(session, np.stack(ps).astype(np.float32), np.stack(fs).astype(np.float32))
                times["N2_TOP3_JOINT"].append((perf_counter_ns() - t0) / 1e6)
                times_sensor[("N2_TOP3_JOINT", sensor)].append(times["N2_TOP3_JOINT"][-1])

            boxes = {
                "B_CURRENT": (b_cx, b_cy, b_w, b_h),
                "N1_TOP1_BOX": (n1_cx, n1_cy, n1_w, n1_h),
                "N2_TOP3_JOINT": (n2_cx, n2_cy, n2_w, n2_h),
            }
            for gt in gts:
                proposal_hit = any(compatible(c, gt, float(cell)) for c in candidates)
                top1_hit = compatible(top1, gt, float(cell))
                top3_hit = any(compatible(c, gt, float(cell)) for c in top3)
                n2_hit = compatible(n2_sel, gt, float(cell))
                for label, (cx, cy, w, h) in boxes.items():
                    iou = iou_box(cx, cy, w, h, gt)
                    centre_err = math.hypot(cx - gt.cx, cy - gt.cy)
                    details[label].append(
                        {
                            "fold": fold_id,
                            "sequence": sequence,
                            "sensor": sensor,
                            "config": label,
                            "proposal_hit": proposal_hit,
                            "ranker_top1_hit": top1_hit,
                            "ranker_top3_hit": top3_hit,
                            "neural_selected_hit": n2_hit if label == "N2_TOP3_JOINT" else top1_hit,
                            "centre_error": centre_err,
                            "iou": iou,
                            "iou50": float(iou >= 0.5),
                            "iou75": float(iou >= 0.75),
                        }
                    )

    for label, vals in times.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        latency_rows.append(
            {
                "fold": fold_id,
                "config": label,
                "sensor": "ALL",
                "inference_p50_ms": float(np.percentile(arr, 50)),
                "inference_p95_ms": float(np.percentile(arr, 95)),
                "inference_p99_ms": float(np.percentile(arr, 99)),
            }
        )
    for (label, sensor), vals in sorted(times_sensor.items()):
        arr = np.asarray(vals, dtype=np.float64)
        latency_rows.append(
            {
                "fold": fold_id,
                "config": label,
                "sensor": sensor,
                "inference_p50_ms": float(np.percentile(arr, 50)),
                "inference_p95_ms": float(np.percentile(arr, 95)),
                "inference_p99_ms": float(np.percentile(arr, 99)),
            }
        )
    return details, latency_rows, model


def summarize_extra(details: list[dict]) -> dict[str, float]:
    agg = aggregate_detection_metrics(details)
    n = len(details)
    if n == 0:
        return agg
    agg["proposal_contains_gt_pct"] = 100.0 * sum(float(d["proposal_hit"]) for d in details) / n
    agg["extratrees_top1_compatible_pct"] = 100.0 * sum(float(d["ranker_top1_hit"]) for d in details) / n
    agg["extratrees_top3_contains_compatible_pct"] = 100.0 * sum(float(d["ranker_top3_hit"]) for d in details) / n
    agg["neural_selected_compatible_pct"] = 100.0 * sum(float(d["neural_selected_hit"]) for d in details) / n
    return agg


def summarize_by(details: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in details:
        groups[str(row[key])].append(row)
    out = []
    for g, rows in sorted(groups.items()):
        agg = summarize_extra(rows)
        agg[key] = g
        agg["config"] = rows[0]["config"]
        out.append(agg)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny foveated neural refiner CV")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--table", default="artifacts/candidate_table.csv")
    parser.add_argument("--cache", default="artifacts/local_patch_cache.npz")
    parser.add_argument("--folds", default="sequence_folds.json")
    parser.add_argument("--cross-sensor-folds", default="docs/runs/2026-08-30/cross_sensor_folds.json")
    parser.add_argument("--out-dir", default="docs/runs/2026-08-30/tiny_foveated_refiner")
    parser.add_argument("--onnx-dir", default="artifacts/tiny_foveated_onnx")
    args = parser.parse_args()

    set_seed(SEED)
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = Path(args.onnx_dir)
    table = load_table(Path(args.table))
    cache = load_cache(Path(args.cache))
    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} param_count={parameter_count()}")

    all_details: dict[str, list[dict]] = defaultdict(list)
    all_latency: list[dict] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        print(f"Fold {fold_id} training+eval...")
        details, latency_rows, _ = evaluate_configs_for_fold(
            fold_id,
            set(fold["train"]),
            set(fold["validation"]),
            table,
            split_dir,
            cache,
            device,
            onnx_dir,
            collect_latency=True,
        )
        for label, rows in details.items():
            all_details[label].extend(rows)
        all_latency.extend(latency_rows)
        print(f"Fold {fold_id} complete")

    summary_rows = []
    for label in ("B_CURRENT", "N1_TOP1_BOX", "N2_TOP3_JOINT"):
        agg = summarize_extra(all_details[label])
        agg["config"] = label
        lat = [r for r in all_latency if r["config"] == label and r.get("sensor", "ALL") == "ALL"]
        if lat:
            for k in ("inference_p50_ms", "inference_p95_ms", "inference_p99_ms"):
                agg[k] = float(np.mean([float(r[k]) for r in lat]))
        summary_rows.append(agg)
    write_csv(out_dir / "main_summary.csv", summary_rows)

    latency_overall = [r for r in all_latency if r.get("sensor", "ALL") == "ALL"]
    latency_by_sensor = [r for r in all_latency if r.get("sensor", "ALL") != "ALL"]
    write_csv(out_dir / "latency.csv", latency_overall)
    write_csv(out_dir / "latency_by_sensor.csv", latency_by_sensor)

    by_sensor, by_sequence, by_fold = [], [], []
    for label, rows in all_details.items():
        for r in summarize_by(rows, "sensor"):
            by_sensor.append(r)
        for r in summarize_by(rows, "sequence"):
            by_sequence.append(r)
        for r in summarize_by(rows, "fold"):
            by_fold.append(r)
    write_csv(out_dir / "by_sensor.csv", by_sensor)
    write_csv(out_dir / "by_sequence.csv", by_sequence)
    write_csv(out_dir / "by_fold.csv", by_fold)

    # Cross-sensor
    cross_folds = json.loads(Path(args.cross_sensor_folds).read_text(encoding="utf-8"))
    cross_details: dict[str, list[dict]] = defaultdict(list)
    cross_latency: list[dict] = []
    for fold in cross_folds:
        fold_id = int(fold["fold"])
        print(f"Cross-sensor fold {fold_id}...")
        details, latency_rows, _ = evaluate_configs_for_fold(
            fold_id,
            set(fold["train"]),
            set(fold["validation"]),
            table,
            split_dir,
            cache,
            device,
            onnx_dir / "cross",
            collect_latency=True,
        )
        for label in ("N1_TOP1_BOX", "N2_TOP3_JOINT"):
            cross_details[label].extend(details[label])
        cross_latency.extend(
            [r for r in latency_rows if r["config"] in ("N1_TOP1_BOX", "N2_TOP3_JOINT") and r.get("sensor", "ALL") == "ALL"]
        )
    cross_summary = []
    for label in ("N1_TOP1_BOX", "N2_TOP3_JOINT"):
        agg = summarize_extra(cross_details[label])
        agg["config"] = label
        lat = [r for r in cross_latency if r["config"] == label]
        if lat:
            agg["inference_p95_ms"] = float(np.mean([float(r["inference_p95_ms"]) for r in lat]))
        cross_summary.append(agg)
    write_csv(out_dir / "cross_sensor_summary.csv", cross_summary)

    b = next(r for r in summary_rows if r["config"] == "B_CURRENT")
    n1 = next(r for r in summary_rows if r["config"] == "N1_TOP1_BOX")
    n2 = next(r for r in summary_rows if r["config"] == "N2_TOP3_JOINT")
    delta_n1_pool = float(n1["pooled_micro_iou50_pct"]) - float(b["pooled_micro_iou50_pct"])
    delta_n2_pool = float(n2["pooled_micro_iou50_pct"]) - float(b["pooled_micro_iou50_pct"])
    delta_n1_macro = float(n1["sequence_macro_iou50_pct"]) - float(b["sequence_macro_iou50_pct"])
    delta_n2_macro = float(n2["sequence_macro_iou50_pct"]) - float(b["sequence_macro_iou50_pct"])
    crit_a = bool(float(n2["pooled_micro_iou50_pct"]) >= float(b["pooled_micro_iou50_pct"]) + 8.0)
    crit_b = bool(float(n2["sequence_macro_iou50_pct"]) >= float(b["sequence_macro_iou50_pct"]) + 5.0)
    crit_c = bool(float(n2.get("inference_p95_ms", 1e9)) <= 30.0)

    gate_rows = [
        {
            "metric": "delta_N1_vs_B_CURRENT_pooled_iou50_pp",
            "value": delta_n1_pool,
        },
        {
            "metric": "delta_N2_vs_B_CURRENT_pooled_iou50_pp",
            "value": delta_n2_pool,
        },
        {
            "metric": "delta_N1_vs_B_CURRENT_sequence_macro_iou50_pp",
            "value": delta_n1_macro,
        },
        {
            "metric": "delta_N2_vs_B_CURRENT_sequence_macro_iou50_pp",
            "value": delta_n2_macro,
        },
        {"metric": "CRITERION_A", "value": float(crit_a)},
        {"metric": "CRITERION_B", "value": float(crit_b)},
        {"metric": "CRITERION_C", "value": float(crit_c)},
    ]
    write_csv(out_dir / "decision_gate.csv", gate_rows)

    md = [
        "# Tiny foveated neural refiner — 2026-08-30",
        "",
        f"Parameter count: {parameter_count()}",
        f"Device used for training: {device}",
        "",
        "Configs: B_CURRENT = C4_MEDIAN + S2 (C1 size features); N1_TOP1_BOX; N2_TOP3_JOINT.",
        "",
        "## Main comparison",
        "",
        "| config | n_gt | pooled micro IoU50 % | sequence macro % | fold mean % | IoU75 % | mean IoU | median IoU | centre p50 | centre p90 | p95 ms |",
        "|--------|-----:|---------------------:|-----------------:|------------:|--------:|---------:|-----------:|-----------:|-----------:|-------:|",
    ]
    for row in summary_rows:
        md.append(
            f"| {row['config']} | {int(row['n_gt'])} | {row['pooled_micro_iou50_pct']:.3f} | "
            f"{row['sequence_macro_iou50_pct']:.3f} | {row['fold_mean_iou50_pct']:.3f} | "
            f"{row['pooled_micro_iou75_pct']:.3f} | {row['mean_iou']:.4f} | {row['median_iou']:.4f} | "
            f"{row.get('centre_error_median', float('nan')):.3f} | {row.get('centre_error_p90', float('nan')):.3f} | "
            f"{row.get('inference_p95_ms', float('nan')):.3f} |"
        )
    md.extend(
        [
            "",
            "## Decision gate (report only — no action taken)",
            "",
            f"- delta_N1_vs_B_CURRENT pooled IoU50: {delta_n1_pool:+.3f} pp",
            f"- delta_N2_vs_B_CURRENT pooled IoU50: {delta_n2_pool:+.3f} pp",
            f"- delta_N1_vs_B_CURRENT sequence macro: {delta_n1_macro:+.3f} pp",
            f"- delta_N2_vs_B_CURRENT sequence macro: {delta_n2_macro:+.3f} pp",
            f"- CRITERION_A (N2 pooled >= B+8pp): {crit_a}",
            f"- CRITERION_B (N2 seq macro >= B+5pp): {crit_b}",
            f"- CRITERION_C (N2 CPU p95 <= 30ms): {crit_c}",
            "",
            "## Cross-sensor (N1/N2)",
            "",
        ]
    )
    for row in cross_summary:
        md.append(
            f"- {row['config']}: pooled={row['pooled_micro_iou50_pct']:.3f}% "
            f"seq_macro={row['sequence_macro_iou50_pct']:.3f}% "
            f"IoU75={row['pooled_micro_iou75_pct']:.3f}% p95={row.get('inference_p95_ms', float('nan')):.3f}ms"
        )
    md.extend(["", f"CSVs: `{out_dir}/`", ""])
    Path("docs/runs/2026-08-30_tiny_foveated_refiner.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
