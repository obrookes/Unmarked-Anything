from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

NEW_OR_CHANGED_SH = [
    "hpc/jobs/dap3_predict_array.sh",
    "hpc/jobs/read_boards.sh",
    "hpc/jobs/mask_verify_array.sh",
    "hpc/jobs/export_distances.sh",
    "hpc/jobs/ctds_abundance.sh",
    "hpc/scripts/run_full_pipeline.sh",
    "hpc/scripts/submit_array.sh",
    "hpc/scripts/submit_array_with_export.sh",
]


@pytest.mark.parametrize("rel_path", NEW_OR_CHANGED_SH)
def test_bash_syntax_ok(rel_path: str) -> None:
    script = REPO_ROOT / rel_path
    assert script.exists(), f"missing script: {script}"
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash -n failed for {rel_path}:\n{result.stderr}"


def _write_pipeline_env(path: Path, out_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f'REPO_ROOT="{REPO_ROOT}"',
                'VIDEO_DIR="assets/videos"',
                'REF_VIDEO_DIR="assets/reference_videos"',
                'REFERENCE_ROOT="/tmp/wcf-pps-p3"',
                f'OUT_ROOT="{out_root}"',
                'SAM3_CKPT="weights/sam3/sam3-safari-neg-parents.pt"',
                'DA3_MODEL_ID="depth-anything/DA3NESTED-GIANT-LARGE"',
                'DA3_MODE="batch"',
                'DA3_BATCH="48"',
                'TARGET_FPS="6.0"',
                "",
            ]
        )
    )


def test_run_full_pipeline_dry_run(tmp_path: Path) -> None:
    env_file = tmp_path / "pipeline.env"
    out_root = tmp_path / "pss_p3_out"
    _write_pipeline_env(env_file, out_root)

    env = dict(os.environ)
    env["DRY_RUN"] = "1"
    env["PIPELINE_ENV"] = str(env_file)

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "hpc/scripts/run_full_pipeline.sh")],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr

    for marker in ["[A]", "[B]", "[C]", "[D]", "[E]", "[F]"]:
        assert marker in output, f"missing stage marker {marker} in dry-run output"

    assert "dap3_predict_array.sh" in output
    assert "read_boards.sh" in output
    assert "mask_verify_array.sh" in output
    assert "export_distances.sh" in output
    assert "ctds_abundance.sh" in output
    assert "--dependency=afterok:" in output


def test_run_full_pipeline_missing_env_errors() -> None:
    env = dict(os.environ)
    env["DRY_RUN"] = "1"
    env["PIPELINE_ENV"] = "/nonexistent/pipeline.env"
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "hpc/scripts/run_full_pipeline.sh")],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "not found" in (result.stdout + result.stderr)


def test_build_ctds_inputs_from_synthetic_export(tmp_path: Path) -> None:
    """Synthetic export CSV -> build_ctds_inputs.py -> ctds_flatfile/activity_times.

    R (ctds_abundance.R) is skipped by default -- it needs the Distance/dplyr/activity/yaml R
    packages, which aren't assumed to be available locally. Set CTDS_TEST_RSCRIPT to an Rscript
    binary (e.g. `conda run -n r_env Rscript`) to also exercise that stage.
    """
    distances_csv = tmp_path / "distances.csv"
    metadata_csv = tmp_path / "camera_metadata.csv"
    config_yaml = tmp_path / "ctds_config.yaml"
    out_dir = tmp_path / "ctds_out"

    distances_csv.write_text(
        "transect_cam,distance,detection_datetime,video_name,ind_no,confidence\n"
        "1_cam001,4.5,2024-01-01T06:00:00,vid1,1,0.9\n"
        "1_cam001,5.0,2024-01-01T06:20:00,vid1,1,0.9\n"
        "2_cam002,3.2,2024-01-02T18:00:00,vid2,1,0.8\n"
    )
    metadata_csv.write_text(
        "transect_cam,transect_id,ct_model,fov_deg,ct_days\n"
        "1_cam001,1,DS-30MP,38,10\n"
        "2_cam002,2,DS-32MP,50,20\n"
    )
    config_yaml.write_text(
        "\n".join(
            [
                "area_km2: 100.0",
                'region_label: "TEST"',
                "snapshot_interval_s: 2",
                "exclude_transects: []",
                "independence_min: 15",
                "truncation:",
                "  left: 1",
                "  right: 12",
                "activity:",
                "  reps: 10",
                '  sample: "model"',
                "  adj: 1.5",
                'model: "auto"',
            ]
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "apps/camera_trap/abundance/build_ctds_inputs.py"),
            "--distances",
            str(distances_csv),
            "--metadata",
            str(metadata_csv),
            "--config",
            str(config_yaml),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr

    flatfile = out_dir / "ctds_flatfile.csv"
    activity = out_dir / "activity_times.csv"
    summary = out_dir / "inputs_summary.json"
    assert flatfile.exists()
    assert activity.exists()
    assert summary.exists()
    assert "Sample.Label" in flatfile.read_text()

    rscript_bin = os.environ.get("CTDS_TEST_RSCRIPT")
    if not rscript_bin:
        pytest.skip("CTDS_TEST_RSCRIPT not set; skipping ctds_abundance.R stage")
    if shutil.which(rscript_bin.split()[0]) is None:
        pytest.skip(f"Rscript binary not found: {rscript_bin}")

    r_out_dir = tmp_path / "abundance"
    r_cmd = rscript_bin.split() + [
        str(REPO_ROOT / "apps/camera_trap/abundance/ctds_abundance.R"),
        "--flatfile",
        str(flatfile),
        "--activity",
        str(activity),
        "--config",
        str(config_yaml),
        "--out-dir",
        str(r_out_dir),
    ]
    r_result = subprocess.run(r_cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert r_result.returncode == 0, r_result.stderr
