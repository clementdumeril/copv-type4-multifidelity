#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_utils import SIM_RESULTS, ensure_dirs, write_json  # noqa: E402


def find_ccx() -> Path | None:
    candidates = [
        ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx_MT.exe",
        ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a generated CalculiX case if ccx is available.")
    parser.add_argument("--case-id", default="cal_001")
    args = parser.parse_args()

    ensure_dirs()
    case_dir = SIM_RESULTS / "calculix" / args.case_id
    manifest_path = case_dir / "calculix_case_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing generated CalculiX manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inp = Path(manifest["inp"])
    ccx = find_ccx()
    status = {
        "case_id": args.case_id,
        "base_case_id": manifest.get("base_case_id", manifest.get("case_id", args.case_id)),
        "output_id": manifest.get("output_id", args.case_id),
        "ccx_available": bool(ccx),
        "ccx_path": str(ccx) if ccx else "",
        "job_base": str(inp.with_suffix("")),
        "returncode": None,
        "solver_status": "not_run_no_ccx",
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if ccx is None:
        write_json(case_dir / "calculix_run_status.json", status)
        print("CalculiX executable not found")
        return

    cmd = [str(ccx), "-i", inp.stem]
    completed = subprocess.run(
        cmd,
        cwd=str(case_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    status.update(
        {
            "returncode": completed.returncode,
            "solver_status": "ccx_completed" if completed.returncode == 0 else "ccx_failed",
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "frd_exists": (case_dir / f"{inp.stem}.frd").exists(),
            "dat_exists": (case_dir / f"{inp.stem}.dat").exists(),
            "sta_exists": (case_dir / f"{inp.stem}.sta").exists(),
        }
    )
    write_json(case_dir / "calculix_run_status.json", status)
    print(case_dir / "calculix_run_status.json")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
