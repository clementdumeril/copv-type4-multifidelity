#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_utils import SIM_RESULTS, ensure_dirs, flatten_case, load_cases, write_csv, write_json  # noqa: E402


PYTHON = Path(r"C:\Program Files\FreeCAD 1.1\bin\python.exe")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def run_step(script: str, case_id: str, extra: list[str] | None = None, timeout: int = 1200) -> dict:
    args = [str(PYTHON), str(ROOT / "high_fidelity" / script), "--case-id", case_id]
    if extra:
        args.extend(extra)
    completed = subprocess.run(
        args,
        cwd=str(ROOT.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return {
        "script": script,
        "case_id": case_id,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CalculiX composite demonstrator over DOE cases.")
    parser.add_argument("--limit", type=int, default=0, help="Number of DOE cases to run; 0 means all cases.")
    parser.add_argument("--start-case", default="", help="First case_id to run, e.g. cal_065.")
    parser.add_argument("--end-case", default="", help="Last case_id to run, e.g. cal_128.")
    parser.add_argument("--case-id", action="append", default=[], help="Run only the specified case_id; repeatable.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip cases with ccx_completed summaries.")
    parser.add_argument("--render-case", default="cal_001")
    parser.add_argument("--n-meridional", type=int, default=24)
    parser.add_argument("--n-theta", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=1200, help="Per-step timeout in seconds.")
    args = parser.parse_args()

    ensure_dirs()
    all_cases = load_cases()
    cases = all_cases
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in requested]
    if args.start_case:
        cases = [case for case in cases if case["case_id"] >= args.start_case]
    if args.end_case:
        cases = [case for case in cases if case["case_id"] <= args.end_case]
    if args.limit > 0:
        cases = cases[: args.limit]

    results = []
    for case in cases:
        cid = case["case_id"]
        summary_path = SIM_RESULTS / "calculix" / cid / "calculix_summary.json"
        if args.skip_completed and summary_path.exists():
            import json as _json

            summary = _json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("solver_status") == "ccx_completed":
                print(f"[CalculiX DOE] skipping completed {cid}", flush=True)
                results.append({"script": "skip_completed", "case_id": cid, "returncode": 0})
                continue
        print(f"[CalculiX DOE] generating {cid}", flush=True)
        results.append(
            run_step(
                "generate_calculix_composite_case.py",
                cid,
                ["--n-meridional", str(args.n_meridional), "--n-theta", str(args.n_theta)],
                timeout=args.timeout,
            )
        )
        if results[-1]["returncode"] != 0:
            continue
        print(f"[CalculiX DOE] running ccx {cid}", flush=True)
        results.append(run_step("run_calculix_case.py", cid, timeout=args.timeout))
        if results[-1]["returncode"] != 0:
            continue
        if cid == args.render_case:
            print(f"[CalculiX DOE] rendering ParaView {cid}", flush=True)
            results.append(run_step("render_calculix_paraview.py", cid, timeout=args.timeout))
        print(f"[CalculiX DOE] post-processing {cid}", flush=True)
        results.append(run_step("postprocess_calculix_case.py", cid, timeout=args.timeout))
    manifest = {
        "case_count_requested": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "render_case": args.render_case,
        "steps": results,
        "failed_steps": [row for row in results if row["returncode"] != 0],
    }
    out = SIM_RESULTS / "calculix" / "calculix_doe_run_manifest.json"
    write_json(out, manifest)
    print(out)

    # Aggregate CalculiX summaries → external_results.csv so that the
    # calibration pipeline always uses the latest simulation results.
    import json as _json

    ext_rows = []
    for case in all_cases:
        cid = case["case_id"]
        sp = SIM_RESULTS / "calculix" / cid / "calculix_summary.json"
        if not sp.exists():
            continue
        s = _json.loads(sp.read_text(encoding="utf-8"))
        if s.get("solver_status") != "ccx_completed":
            continue
        flat = flatten_case(case)
        ext_rows.append(
            {
                **flat,
                "simulation_status": "calculix_composite_shell_completed",
                "simulation_source": s.get("local_failure_projection", ""),
                "max_radial_displacement_mm": s.get("max_displacement_mm", 0.0),
                "max_hashin": s.get("max_hashin", 0.0),
                "max_tsai_wu": s.get("max_tsai_wu", 0.0),
                "max_puck": s.get("max_puck", 0.0),
                "max_combined": s.get("max_combined", 0.0),
                "sigma_hoop_mpa": s.get("sigma_11_mpa_at_critical", 0.0),
                "sigma_meridional_or_axial_mpa": s.get("sigma_22_mpa_at_critical", 0.0),
                "sigma_radial_mpa": 0.0,
                "critical_zone": s.get("critical_zone", ""),
                "critical_angle_deg": s.get("critical_angle_deg", 0.0),
                "critical_ply": s.get("critical_ply", ""),
                "ratio_cylinder": float("nan"),
                "ratio_dome": float("nan"),
                "ratio_junction": float("nan"),
                "ratio_boss": float("nan"),
                "ratio_combined": float("nan"),
                "critical_zone_simulation": s.get("critical_zone", ""),
                "calculix_max_von_mises_mpa": s.get("max_von_mises_mpa", 0.0),
                "calculix_projection_method": s.get("local_failure_projection", ""),
            }
        )
    if ext_rows:
        write_csv(SIM_RESULTS / "external_results.csv", ext_rows)

    if manifest["failed_steps"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
