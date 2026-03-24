#!/usr/bin/env python3
"""Merge legacy da3_streaming depth maps into DAP3 stream NPZ as *_old keys.

Also computes a parallel per-frame JSON field (default: depth_mask_mean_old)
without overwriting the existing depth_mask_mean values.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from depth_anything_3.utils.camera_trap_masks import load_mask_bool

OLD_FRAME_RE = re.compile(r"^frame_(\d+)\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge old depth maps from results_output/frame_<idx>.npz into a new "
            "<video>_arrays.npz file using keys '<existing_depth_key><suffix>'."
        )
    )
    parser.add_argument(
        "--old-results-dir",
        type=Path,
        required=True,
        help="Legacy results_output directory containing frame_<idx>.npz.",
    )
    parser.add_argument(
        "--new-path",
        type=Path,
        required=True,
        help=(
            "Path to either a run directory containing per-video subfolders, or a single "
            "per-video folder containing one .json and one *_arrays.npz."
        ),
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default=None,
        help="Video name/stem when --new-path points to a multi-video run directory.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Optional output NPZ path. If omitted, the target *_arrays.npz is overwritten.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path. If omitted, updates the selected per-video JSON in place. "
            "The existing depth_mask_mean is preserved."
        ),
    )
    parser.add_argument(
        "--old-key-suffix",
        type=str,
        default="_old",
        help="Suffix appended to each existing depth key (default: _old).",
    )
    parser.add_argument(
        "--old-depth-mask-mean-key",
        type=str,
        default="depth_mask_mean_old",
        help="JSON key name for old-depth mask mean values (default: depth_mask_mean_old).",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing merged keys when they already exist in output NPZ.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup when overwriting in-place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing output NPZ.",
    )
    args = parser.parse_args()
    if not args.old_key_suffix:
        parser.error("--old-key-suffix must be non-empty")
    if not args.old_depth_mask_mean_key:
        parser.error("--old-depth-mask-mean-key must be non-empty")
    return args


def _find_single_json(dir_path: Path) -> Path:
    candidates = sorted(p for p in dir_path.glob("*.json") if p.name != "run_manifest.json")
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one per-video JSON in {dir_path}, found {len(candidates)}.")
    return candidates[0]


def _find_single_arrays(dir_path: Path) -> Path:
    candidates = sorted(dir_path.glob("*_arrays.npz"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one *_arrays.npz in {dir_path}, found {len(candidates)}.")
    return candidates[0]


def resolve_video_paths(new_path: Path, video_name: str | None) -> tuple[Path, Path, Path]:
    if not new_path.is_dir():
        raise FileNotFoundError(f"--new-path is not a directory: {new_path}")

    direct_json = sorted(p for p in new_path.glob("*.json") if p.name != "run_manifest.json")
    direct_npz = sorted(new_path.glob("*_arrays.npz"))
    if len(direct_json) == 1 and len(direct_npz) == 1:
        video_dir = new_path
        return video_dir, direct_json[0], direct_npz[0]

    if video_name:
        video_dir = new_path / video_name
        if not video_dir.is_dir():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
    else:
        video_dirs = sorted(
            d
            for d in new_path.iterdir()
            if d.is_dir() and not (d / "run_manifest.json").exists()
        )
        if len(video_dirs) != 1:
            raise RuntimeError(
                "Multiple video directories found under --new-path. Provide --video-name."
            )
        video_dir = video_dirs[0]

    return video_dir, _find_single_json(video_dir), _find_single_arrays(video_dir)


def load_old_depth_map(old_results_dir: Path) -> dict[int, Path]:
    if not old_results_dir.is_dir():
        raise FileNotFoundError(f"--old-results-dir not found: {old_results_dir}")
    mapping: dict[int, Path] = {}
    for path in sorted(old_results_dir.glob("frame_*.npz")):
        m = OLD_FRAME_RE.match(path.name)
        if not m:
            continue
        mapping[int(m.group(1))] = path
    if not mapping:
        raise RuntimeError(f"No frame_<idx>.npz files found in {old_results_dir}")
    return mapping


def load_new_depth_keys(video_json_path: Path) -> dict[int, str]:
    video_json = json.loads(video_json_path.read_text(encoding="utf-8"))
    mapping: dict[int, str] = {}
    for frame in video_json.get("frames") or []:
        idx = frame.get("frame_index")
        depth_key = ((frame.get("npz_keys") or {}).get("depth"))
        if idx is None or not depth_key:
            continue
        mapping[int(idx)] = str(depth_key)
    if not mapping:
        raise RuntimeError(f"No frame depth keys found in {video_json_path}")
    return mapping


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


def atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.json")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def maybe_backup(path: Path, enabled: bool) -> Path | None:
    if not enabled or not path.exists():
        return None
    backup_path = path.with_name(f"{path.name}.bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
        print(f"Created backup: {backup_path}")
    else:
        print(f"Backup exists, reusing: {backup_path}")
    return backup_path


def _mask_entries_for_frame(frame_row: dict[str, Any]) -> list[dict[str, Any]]:
    npz_keys = frame_row.get("npz_keys") or {}
    mask_entries = npz_keys.get("masks") or []
    if mask_entries:
        return list(mask_entries)
    return list(npz_keys.get("objects") or [])


def build_union_mask(frame_row: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[np.ndarray | None, int]:
    union_mask: np.ndarray | None = None
    missing_keys = 0
    for entry in _mask_entries_for_frame(frame_row):
        key = entry.get("key")
        if not key:
            continue
        try:
            mask_bool = load_mask_bool(npz_data=arrays, entry=entry)
        except KeyError:
            missing_keys += 1
            continue
        if union_mask is None:
            union_mask = mask_bool.copy()
        else:
            if union_mask.shape != mask_bool.shape:
                raise ValueError(
                    f"Mask shape mismatch for frame {frame_row.get('frame_index')}: "
                    f"{union_mask.shape} vs {mask_bool.shape}"
                )
            union_mask |= mask_bool
    return union_mask, missing_keys


def compute_depth_mask_mean(depth: np.ndarray, union_mask: np.ndarray) -> float | None:
    if depth.shape != union_mask.shape:
        depth = cv2.resize(depth, (union_mask.shape[1], union_mask.shape[0]), interpolation=cv2.INTER_CUBIC)
    values = depth[union_mask]
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def main() -> int:
    args = parse_args()
    old_results_dir = args.old_results_dir.resolve()
    new_path = args.new_path.resolve()

    video_dir, video_json_path, arrays_path = resolve_video_paths(new_path=new_path, video_name=args.video_name)
    old_depths = load_old_depth_map(old_results_dir=old_results_dir)
    new_depth_keys = load_new_depth_keys(video_json_path=video_json_path)
    video_json = json.loads(video_json_path.read_text(encoding="utf-8"))
    frames = video_json.get("frames") or []
    frame_by_index: dict[int, dict[str, Any]] = {}
    for frame in frames:
        idx = frame.get("frame_index")
        if idx is None:
            continue
        frame_by_index[int(idx)] = frame

    output_npz = args.output_npz.resolve() if args.output_npz is not None else arrays_path
    output_json = args.output_json.resolve() if args.output_json is not None else video_json_path
    in_place = output_npz == arrays_path

    with np.load(arrays_path, allow_pickle=False) as arrays_in:
        merged_arrays = {k: np.asarray(arrays_in[k]) for k in arrays_in.files}

    matched = 0
    missing_old = 0
    missing_depth_field = 0
    existing_skipped = 0
    merged_written = 0
    json_mean_updated = 0
    json_missing_frame_row = 0
    json_no_mask_entries = 0
    json_missing_mask_keys = 0

    for frame_index, depth_key in sorted(new_depth_keys.items()):
        matched += 1
        old_npz_path = old_depths.get(frame_index)
        if old_npz_path is None:
            missing_old += 1
            continue

        with np.load(old_npz_path, allow_pickle=False) as old_npz:
            if "depth" not in old_npz.files:
                missing_depth_field += 1
                continue
            old_depth = np.asarray(old_npz["depth"], dtype=np.float32)

        frame_row = frame_by_index.get(frame_index)
        if frame_row is None:
            json_missing_frame_row += 1
        else:
            union_mask, missing_keys = build_union_mask(frame_row, merged_arrays)
            json_missing_mask_keys += missing_keys
            if union_mask is None:
                json_no_mask_entries += 1
            else:
                frame_row[args.old_depth_mask_mean_key] = compute_depth_mask_mean(old_depth, union_mask)
                json_mean_updated += 1

        target_key = f"{depth_key}{args.old_key_suffix}"
        if target_key in merged_arrays and not args.overwrite_existing:
            existing_skipped += 1
            continue

        merged_arrays[target_key] = old_depth
        merged_written += 1

    summary = {
        "video_dir": str(video_dir),
        "video_json": str(video_json_path),
        "output_json": str(output_json),
        "input_arrays": str(arrays_path),
        "output_arrays": str(output_npz),
        "old_results_dir": str(old_results_dir),
        "key_suffix": args.old_key_suffix,
        "old_depth_mask_mean_key": args.old_depth_mask_mean_key,
        "counts": {
            "old_depth_frames": len(old_depths),
            "new_depth_frames": len(new_depth_keys),
            "processed_new_frames": matched,
            "merged_written": merged_written,
            "missing_old_frame": missing_old,
            "missing_old_depth_field": missing_depth_field,
            "existing_skipped": existing_skipped,
            "json_old_depth_mask_mean_updated": json_mean_updated,
            "json_missing_frame_row": json_missing_frame_row,
            "json_no_mask_entries": json_no_mask_entries,
            "json_missing_mask_keys": json_missing_mask_keys,
            "final_npz_keys": len(merged_arrays),
        },
        "mode": {
            "in_place": in_place,
            "dry_run": bool(args.dry_run),
            "overwrite_existing": bool(args.overwrite_existing),
            "backup": bool(not args.no_backup),
        },
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    maybe_backup(arrays_path, enabled=in_place and not args.no_backup)
    maybe_backup(output_json, enabled=not args.no_backup)

    atomic_save_npz(output_npz, merged_arrays)
    atomic_save_json(output_json, video_json)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
