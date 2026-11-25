"""Run the Section 6 solvers with fresh output directories on Linux or macOS."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(directory, script, *args):
    argv = [sys.executable, script, *map(str, args)]
    print(directory, "$", " ".join(argv), flush=True)
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run(argv, cwd=directory, env=env, check=True)


def prepare(folder, workspace):
    source = ROOT / folder
    destination = workspace / folder
    destination.mkdir(parents=True, exist_ok=False)
    for script in source.glob("*.py"):
        shutil.copy2(script, destination / script.name)
    if (source / "inputs").exists():
        shutil.copytree(source / "inputs", destination / "inputs")
    return destination


def hopf(work, example=False):
    directory = prepare("hopf", work)
    solver = "run_experiment.py"
    if example:
        run(directory, solver, "--bases", "A", "--targets", "flex_1", "control_1", "--levels", "coarse")
        return
    levels = ["coarse", "medium", "fine", "cusp", "deep", "finest"]
    run(directory, solver, "--levels", *levels)
    run(directory, solver, "--targets", "flex_b1", "flex_b2", "control_b1", "control_b2", "--levels", *levels)
    run(directory, solver, "--targets", "flex_4", "control_4", "--levels", "deep", "finest")
    run(directory, solver, "--targets", "flex_4", "--levels", "quadrature", "--initial-level", "finest")
    run(directory, "finalize_results.py")


def wp(work):
    directory = prepare("wp", work)
    points = [(".23", "1.08", "5", "2"), (".07", ".94", "-4", "3"), (".31", ".83", "-2", "7")]
    settings = [(32, ".001", ".0001", 64, 1.2, 4), (64, ".0005", ".00003", 80, 1.15, 5),
                (96, ".00025", ".00001", 96, 1.12, 6)]
    extra_steps = {32: ".0005", 64: ".001", 96: ".0005"}
    for i, (real, imag, sr, si) in enumerate(points, start=1):
        for n, step, radius, angles, ratio, order in settings:
            steps = [step, extra_steps[n]] if i == 1 else [step]
            for delta in steps:
                run(directory, "second_fundamental.py", "--tau", real, imag, "--s", sr, si,
                    "--N", n, "--rmin", radius, "--ntheta", angles, "--ratio", ratio,
                    "--quad", order, "--common", "--h", delta, "--out", f"results/generic{i}",
                    "--label", f"common-N{n}-h{float(delta):g}", *(["--shift"] if n == 96 else []))
    run(directory, "geodesic_test.py", "--source", "results/generic2/common-N32-h0.001.json",
        "--out", "results/geodesic-generic2", "--h", ".0005", "--times", ".04", ".02", "--steps", 1, 2)
    run(directory, "geodesic_test.py", "--source", "results/generic2/common-N64-h0.0005.json",
        "--out", "results/geodesic-generic2-N64", "--h", ".00025", "--times", ".02", "--steps", 1)
    run(directory, "aggregate.py", "--out", "recomputed")


def isomonodromy(work, canonical):
    original = prepare("isomonodromy/original", work)
    refined = prepare("isomonodromy/refined", work)
    if canonical:
        shutil.copytree(canonical, original / "canonical")
    for base in "AB":
        for control in [False, True]:
            flags = ["--base", base] + (["--control"] if control else [])
            if not canonical:
                run(original, "build_canonical.py", *flags)
            run(original, "run_fixed.py", *flags, "--cusp")
    (refined / "canonical").symlink_to(original / "canonical", target_is_directory=True)
    for base in "AB":
        for stage in ["J1", "J2"]:
            run(refined, "run_pipeline.py", "--base", base, "--stage", stage)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=["example", "hopf", "wp", "isomonodromy"])
    p.add_argument("--work", type=Path, required=True, help="parent for fresh experiment output")
    p.add_argument("--canonical", type=Path, help="use the archived original N64 connections for isomonodromy")
    args = p.parse_args()
    if args.canonical and args.experiment != "isomonodromy":
        p.error("--canonical applies only to isomonodromy")
    workspace = args.work.resolve()
    name = "hopf" if args.experiment == "example" else args.experiment
    if (workspace / name).exists():
        p.error(f"{workspace / name} exists. Choose a fresh --work directory for this experiment.")
    if args.experiment == "example":
        hopf(workspace, example=True)
    elif name == "hopf":
        hopf(workspace)
    elif name == "wp":
        wp(workspace)
    else:
        isomonodromy(workspace, args.canonical)
    if args.experiment != "example":
        run(ROOT, "tables.py", name, "--data", workspace, "--out", workspace / "tables")
