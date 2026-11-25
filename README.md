# Three numerical experiments on the flex locus

Programs, inputs and numerical records for Section 6 of *Invariant subvarieties, reflection groups, and non-abelian Hodge theory*, by Yeuk Hay Joshua Lam. The release corresponding to the paper is [v1.0.0](https://github.com/joshuayhlam/flex-locus-experiments/releases/tag/v1.0.0).

## Rebuild the tables

From the repository root, run the following with Python 3. No numerical libraries are needed for this step.

```sh
python3 tables.py --check
```

This writes eight LaTeX tabulars in `tables/` (for use with `booktabs` and `amssymb`). It checks the ordered, rounded data rows of each table against a digest taken from the manuscript. See `published-rows.json`. Omit `--check` and pass `hopf`, `wp` or `isomonodromy` to build just one experiment's tables.

| Paper | Output in tables/ | Numerical records |
| --- | --- | --- |
| §6.1, Table 5. Flex targets and controls | nearby-flex-targets.tex | hopf/results/, finest stage |
| §6.1, Table 6. Mesh and cusp refinement | nearby-flex-refinement.tex | hopf/results/, direction a, amplitude 1 |
| §6.2, Table 7. Full second fundamental form | wp-second-fundamental.tex | wp/results/summary.json, full_tensors |
| §6.2, Table 8. Components at the finest mesh | wp-coefficients.tex | Same tensors, symmetrized at N = 96 |
| §6.2, Table 9. Ambient geodesic | wp-geodesic-test.tex | wp/results/geodesic-generic2*/geodesics.json, collected in summary.json |
| §6.3, Table 10. Candidates and controls | nonfuchsian-controls-F.tex | isomonodromy/original/cusp_fixed_runs/, stage F |
| §6.3, Table 11. Maxima over four deformed targets | nonfuchsian-candidate-refinement.tex | Same directory, candidate runs at M1, M2, C1 and F |
| §6.3, Table 12. Refined source representations | nonfuchsian-refined-F.tex | isomonodromy/refined/runs/, candidate J2 at inverse stage F1 |

## Repeat the numerical solves

Use Python 3.12 on Linux or macOS. The following environment was used to test this release. The original run metadata records NumPy 2.5.4 and SciPy 1.18.1. Sparse solves, triangulations and finite differences can vary with the numerical libraries and hardware.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python rerun.py example --work example-run
```

This small example starts from the marked curves at source A and computes the coarse-mesh Hopf differential for the flex target in direction a, amplitude 1, and its transverse control. The two JSON files are under `example-run/hopf/results/`. Compare `measurement.tail_corrected_pencil_defect` with about 0.000234 at the flex target and 0.655 at the control. In a fresh run with the environment above, the flex value differed from the archived one by 0.3% and the control by less than 0.000001 in absolute value. This gives a scale for this particular comparison. The example takes seconds on the machine used for the release checks.

Each of the following commands runs a complete experiment and builds its tables from the new results. They may share the same parent directory. Each experiment starts in a fresh subdirectory so archived JSON files cannot cause the solvers to skip work.

```sh
python rerun.py hopf --work fresh
python rerun.py wp --work fresh
python rerun.py isomonodromy --work fresh
```

The full runs involve many finite-element fits and need substantially more time and disk space than the small example. JSON files record solver settings, timings and residuals. The working directory also retains regenerated meshes or nodal arrays where the solver writes them. Completed experiments can be read again with, for example, `python tables.py wp --data fresh --out fresh/tables`.

### §6.1. The Hitchin image

`hopf/geometry_inputs.py` specifies the basepoints, marked points, targets and controls. `hopf/geometry_manifest.json` records their coordinates. The six manuscript stages M1, M2, M3, C1, C2, M4 are called coarse, medium, fine, cusp, deep, finest in the code. Direction b uses the filenames flex_b1 and flex_b2. The table uses the cusp-tail-corrected pencil defect and the relative holomorphic-fit residual. The quadrature check starts from the finest map and changes the integration order. The finalization step polishes saved solves when needed.

### §6.2. Weil–Petersson geodesicity

`wp/results/generic1/`, `generic2/` and `generic3/` correspond to A, B and C. Their metrics directories contain the ambient metric evaluations at the derivative stencil points. To recompute the tensors and assemble the endpoints from saved data, copy the directory and run the aggregation in the copy. It updates its own summary file.

```sh
cp -R wp wp-check
(cd wp-check && python aggregate.py --out recomputed && python audit.py)
```

The full command also solves the ambient metrics again and integrates the geodesics. The extra step sizes at A supply the paired finite-difference check. In the release check, a fresh computation at B on the coarsest mesh gave 0.0000089243 for the complex tensor defect. Reassembling all nine archived tensors changed that defect by less than 0.000000000005 at every row.

### §6.3. Fixed monodromy and refinement of the source

`isomonodromy/original/inputs/` gives the Appendix A coefficients used at each basepoint. The solver keeps them fixed. Run `seed_coefficients.py` in a separate copy of that folder's parent to recompute them by integration. In the numerical filenames, **fibre** denotes the candidate and **control** the transverse perturbation. There are four nearby target rows per source. The maxima in Table 11 exclude the undeformed source and return. Each target JSON records its representation hash, and the table builder checks that this hash stays fixed across the target rows and inverse mesh stages for each candidate.

The original source mesh has N = 64. The refined construction uses J1 as a starting stage for J2, which has N = 128. The refined candidate in Table 12 is J2. **F1 in the refined directory is the same inverse mesh as F in the original directory.** The two families of generator matrices are recorded in the respective flat_projection.json files. The target solves use the prescribed cusp boundary data and report the measured pencil and fiber defects.

The full command rebuilds the four original connections. To start instead from exactly the original numerical representations, download [original-connections.zip](https://github.com/joshuayhlam/flex-locus-experiments/releases/download/v1.0.0/original-connections.zip) (119 MiB). Its SHA-256 is `0dc1fe0700ea2409f50726c83fd61bab1e7cae927f80c429db1d49999cbd29b0`.

```sh
unzip original-connections.zip -d checkpoints
python rerun.py isomonodromy --work fixed-run --canonical checkpoints/canonical
```

The release check repeated the six inverse solves at B, stage M1, from the saved candidate connection. The fiber defects agreed with the archived values to within 0.000000000001 in absolute value. Fresh numerical agreement measures repeatability of the discretized calculations. The manuscript gives the mesh-refinement evidence and discusses its bearing on the geometric conjectures.
