"""Generate the eight Section 6 tables from numerical records (standard library only)."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEVELS = dict(M1="coarse", M2="medium", M3="fine", C1="cusp", C2="deep", M4="finest")
TARGETS = [("Source", "source"), ("$(a,1)$", "flex_1"), ("$(a,2)$", "flex_2"),
           ("$(b,1)$", "flex_b1"), ("$(b,2)$", "flex_b2")]
EPS = r"$\varepsilon_h$"
ETA = r"$\eta_h$"
FIBER = r"$\delta_{\nu,h}$"


def read(path):
    return json.loads(path.read_text())


def sci(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def write_table(directory, name, header, rows):
    lines = [r"\begin{tabular}{@{}" + "l" * len(header) + "@{}}", r"\toprule",
             " & ".join(header) + r"\\", r"\midrule"]
    for row in rows:
        cells = [f"${sci(v)}$" if isinstance(v, float) else str(v) for v in row]
        lines.append(" & ".join(cells) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (directory / (name + ".tex")).write_text("\n".join(lines) + "\n")


def hopf(root, out):
    def record(base, target, level):
        return read(root / "hopf/results" / f"{base}_{target}_{level}.json")
    epsilon = "tail_corrected_pencil_defect"
    eta = "holomorphic_fit_residual"
    rows = []
    for base in "AB":
        for direction, amplitude in [("a", 1), ("a", 2), ("a", 4), ("b", 1), ("b", 2)]:
            suffix = str(amplitude) if direction == "a" else f"b{amplitude}"
            flex = record(base, "flex_" + suffix, "finest")["measurement"]
            control = record(base, "control_" + suffix, "finest")["measurement"]
            rows.append([base, f"${direction}$", amplitude, flex[epsilon], flex[eta],
                         f"${control[epsilon]:.5f}$"])
    write_table(out, "nearby-flex-targets", ["Source", "Direction", r"$\alpha$", EPS, ETA,
                                            "Control " + EPS], rows)
    rows = []
    for stage, level in LEVELS.items():
        a, b = [record(base, "flex_1", level) for base in "AB"]
        ma, mb = a["measurement"], b["measurement"]
        rows.append([stage, a["settings"]["rmin"], ma[epsilon], mb[epsilon], ma[eta], mb[eta]])
    write_table(out, "nearby-flex-refinement", ["Stage", "$R$", r"$\varepsilon_A$",
                r"$\varepsilon_B$", r"$\eta_A$", r"$\eta_B$"], rows)


def wp(root, out):
    summary = read(root / "wp/results/summary.json")
    tensors = []
    coeffs = []
    for item in summary["full_tensors"]:
        base = "$x_" + item["sample"] + "$"
        tensors.append([base, item["N"], item["h"], item["epsilon_II"], item["epsilon_II_real_half"]])
        if item["N"] == 96:
            b = item["B_orthonormal"]
            off_diagonal = (complex(*b[0][1]) + complex(*b[1][0])) / 2
            coeffs.append([base, abs(complex(*b[0][0])), abs(off_diagonal), abs(complex(*b[1][1]))])
    write_table(out, "wp-second-fundamental", ["Source", "$N$", r"$\delta$",
                r"$\varepsilon_{\mathrm{II}}$", r"$\frac12\|\mathrm{II}_{\mathbb R}\|$"], tensors)
    write_table(out, "wp-coefficients", ["Source", "$|B_{11}|$", "$|B_{12}|$", "$|B_{22}|$"], coeffs)
    rows = [[g["N"], f"${g['time']:.2f}$", g["steps"], abs(complex(*g["normal_residual"])),
             abs(complex(*g["control_normal_residual"]))] for g in summary["geodesics"]]
    write_table(out, "wp-geodesic-test", ["Metric $N$", "$t$", "Steps", "$|r(t)|$",
                r"$|\widetilde r(t)|$"], rows)


def isomonodromy(root, out):
    original = root / "isomonodromy/original/cusp_fixed_runs"
    refined = root / "isomonodromy/refined/runs"
    def measurements(directory, expected_hash):
        result = []
        for _, key in TARGETS:
            data = read(directory / f"{key}.json")
            if data["representation_sha256"] != expected_hash:
                raise ValueError(f"Representation changed at {directory}/{key}")
            m = data["measurement"]
            result.append([m["pencil_defect"], m["normalized_fibre_defect"], m["holomorphic_fit_residual"]])
        return result
    comparisons, maxima, improved = [], [], []
    for base in "AB":
        label = f"$x_{base}$"
        initial = original / f"{base}_fibre_F"
        rhash = read(initial / "source.json")["representation_sha256"]
        control = original / f"{base}_control_F"
        chash = read(control / "source.json")["representation_sha256"]
        newer = refined / f"{base}_fibre_J2_F1"
        nhash = read(newer / "source.json")["representation_sha256"]
        if len({rhash, chash, nhash}) != 3:
            raise ValueError(f"Expected three distinct representations at {base}")
        for (target, _), cand, ctl, new in zip(TARGETS, measurements(initial, rhash),
                                               measurements(control, chash), measurements(newer, nhash)):
            comparisons.append([label, target, *cand[:2], *ctl[:2]])
            improved.append([label, target, *new])
        for stage in ["M1", "M2", "C1", "F"]:
            nearby = measurements(original / f"{base}_fibre_{stage}", rhash)[1:]
            maxima.append([label, stage, *map(max, zip(*nearby))])
    write_table(out, "nonfuchsian-controls-F", ["Source", "Target", EPS, FIBER,
                "Control " + EPS, "Control " + FIBER], comparisons)
    write_table(out, "nonfuchsian-candidate-refinement", ["Source", "Stage",
                r"$\max\varepsilon_h$", r"$\max\delta_{\nu,h}$", r"$\max\eta_h$"], maxima)
    write_table(out, "nonfuchsian-refined-F", ["Source", "Target", EPS, FIBER, ETA], improved)


def check_published(directory):
    for name, expected in read(ROOT / "published-rows.json").items():
        source = (directory / (name + ".tex")).read_text()
        rows = source.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
        canonical = re.sub(r"\s|\$", "", rows.replace(r"\midrule", ""))
        if hashlib.sha256(canonical.encode()).hexdigest() != expected:
            raise ValueError(f"Numerical rows differ from the manuscript in {name}")
    print("All eight tables agree with the manuscript at the printed precision.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=["all", "hopf", "wp", "isomonodromy"], default="all", nargs="?")
    p.add_argument("--data", type=Path, default=ROOT, help="archive or fresh run directory")
    p.add_argument("--out", type=Path, default=Path("tables"))
    p.add_argument("--check", action="store_true", help="compare all eight tables with the manuscript")
    args = p.parse_args()
    if args.check and args.experiment != "all":
        p.error("--check compares all three experiments")
    args.out.mkdir(parents=True, exist_ok=True)
    for build in [hopf, wp, isomonodromy]:
        if args.experiment in ["all", build.__name__]:
            build(args.data, args.out)
    if args.check:
        check_published(args.out)
