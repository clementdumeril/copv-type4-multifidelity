#!/usr/bin/env python3
"""Run a CalculiX campaign (generate -> solve -> post-process) into a fresh output folder.

Used for the runs added for paper/: the hoop-coverage ablation over the 384 DOE cases
and the dome mesh-convergence study. Each case runs in its own process; several cases
run in parallel. A CSV log records return codes and wall-clock time per step.

    python code/paper/run_campaign.py --cases data/doe_11l_hoop0_cases.json \
        --out <new folder> --mesh 24 --workers 4 [--case-id 11l_0122 ...]

Needs CCX (path to ccx) in the environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
STEPS = ("generate_calculix_composite_case", "run_calculix_case", "postprocess_calculix_case")


def run_case(case_id: str, env: dict, mesh: int, max_samples: int, timeout: int) -> dict:
    row = {"case_id": case_id, "mesh": mesh}
    for step in STEPS:
        args = [sys.executable, str(CODE / "high_fidelity" / f"{step}.py"), "--case-id", case_id]
        if step == "generate_calculix_composite_case":
            args += ["--n-meridional", str(mesh), "--n-theta", str(mesh)]
        if step == "postprocess_calculix_case":
            args += ["--max-samples", str(max_samples)]
        t0 = time.time()
        try:
            done = subprocess.run(args, cwd=str(CODE), env=env, capture_output=True, text=True, timeout=timeout)
            rc = done.returncode
        except subprocess.TimeoutExpired:
            rc = -9
        row[f"{step}_rc"] = rc
        row[f"{step}_s"] = round(time.time() - t0, 1)
        if rc != 0:
            break
    summary = Path(env["TYPE4_OUTPUTS_DIR"]) / "simulation_results" / "calculix" / case_id / "calculix_summary.json"
    if summary.exists():
        s = json.loads(summary.read_text(encoding="utf-8"))
        for key in ("solver_status", "max_fiber_stress_ratio_abs", "fiber_stress_ratio_p95_abs",
                    "fiber_stress_ratio_p99_abs", "fiber_proxy_zone", "fiber_proxy_ply_name", "failure_sample_stride"):
            row[key] = s.get(key, "")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mesh", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=10_000_000)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--case-id", action="append", default=[])
    ap.add_argument("--skip-completed", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # ccx_MT is capped at 2 threads by default (shared machine); override through the environment.
    env = dict(os.environ, TYPE4_CASES_FILE=str(Path(args.cases).resolve()), TYPE4_OUTPUTS_DIR=str(out),
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"),
               CCX_NPROC_EQUATION_SOLVER=os.environ.get("CCX_NPROC_EQUATION_SOLVER", "2"))
    case_ids = args.case_id or [c["case_id"] for c in json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]]
    if args.skip_completed:
        def done(cid: str) -> bool:
            p = out / "simulation_results" / "calculix" / cid / "calculix_summary.json"
            return p.exists() and json.loads(p.read_text(encoding="utf-8")).get("solver_status") == "ccx_completed"
        case_ids = [c for c in case_ids if not done(c)]

    log = out / f"campaign_log_mesh{args.mesh}.csv"
    print(f"{len(case_ids)} cases, mesh {args.mesh}, {args.workers} workers -> {out}", flush=True)
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, c, env, args.mesh, args.max_samples, args.timeout): c for c in case_ids}
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            print(f"[{n}/{len(case_ids)}] {row['case_id']} {row.get('solver_status', 'FAILED')} "
                  f"FI={row.get('max_fiber_stress_ratio_abs', '')} elapsed={time.time() - t0:.0f}s", flush=True)
            with log.open("w", newline="", encoding="utf-8") as f:
                keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "case_id", k))
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
    print(f"done in {time.time() - t0:.0f}s, log: {log}", flush=True)


if __name__ == "__main__":
    main()
