# CLAUDE.md — Lattice-Distortion

## Project overview
Batch prediction pipeline for RMSAD (lattice distortion), shear modulus (μ),
and yield strength (YS) across full composition grids of refractory HEA systems.
Based on Tandoc et al. 2023, npj Computational Materials.

Repository: github.com/dearangehub/Lattice-Distortion
Owner: dearangehub (fork of tandocca/Lattice-Distortion)

---

## Critical rules — read before doing anything

### GitHub
- NEVER raise a PR targeting `tandocca:main` — this repo is a fork and GitHub
  defaults to the upstream. Always push code directly to `dearangehub:main`
  using the `mcp__github__push_files` or `mcp__github__create_or_update_file`
  MCP tools.
- NEVER push to a branch and open a PR without explicit user approval.
- Do NOT modify: `RMSAD_tool.py`, `train.ipynb`, `training.csv`.

### Environment
- This agent runs in Anthropic's cloud (Linux, `/home/user/`), NOT on the
  user's local Windows machine (`C:\Users\sg3896\`).
- The user's local venv is `dparameter` (Python, Windows).
- IMPORTANT: Files generated in this cloud session are NOT visible in Windows
  Explorer and cannot be run in the user's local venv.
- Workflow: agent writes/pushes code to GitHub → user runs `git pull` and
  executes scripts locally in the `dparameter` venv.
- To run code directly on the user's machine, the user must install
  Claude Code CLI on Windows and run `claude` from their repo directory.
- The user's d-parameter output is at:
  `C:\Users\sg3896\Research\dparameter\data\output\`

---

## Repo layout

```
RMSAD_tool.py          — standalone CLI + get_RMSAD / get_shear_modulus / get_YS
environment.yml        — conda env (name: lattice, conda-forge)
rmsad/
  __init__.py          — version 0.1.0
  grid.py              — generate_grid() for any n-component system
  predict.py           — predict_system(): RMSAD + mu + optional YS via gamma_usf
scripts/
  run_prediction.py    — CLI: python scripts/run_prediction.py SYSTEM --dparameter-dir DIR
  plot_ternary.py      — ternary heatmap of YS_GPa with 900 MPa contour line
data/
  input/               — (empty, gitkeep)
  output/              — grid_*.csv (gitignored), predict_*_RMSAD.csv (tracked)
```

---

## Key workflows

### Run prediction (locally, in dparameter venv)
```bash
# RMSAD + mu only
python scripts/run_prediction.py TiNbV TiNbMo

# RMSAD + mu + YS (requires d-parameter HTP output)
python scripts/run_prediction.py TiNbV TiNbMo \
  --dparameter-dir "C:\Users\sg3896\Research\dparameter\data\output"

# Quaternary at 5% step
python scripts/run_prediction.py TiNbVMo --step 5
```

### d-parameter HTP file convention
- Located at: `C:\Users\sg3896\Research\dparameter\data\output\`
- File name pattern: `predict_{system}_HTP.csv`  e.g. `predict_TiNbV_HTP.csv`
- The `gamma_usf` column name in these files may vary — `_find_gamma_usf_col()`
  in `predict.py` auto-detects it by scanning for "usf", "gamma", "gsfe", "usfe".
  If detection fails, the columns present are printed as a warning.

### YS formula (Eq. 3, Tandoc et al. 2023)
```
YS_GPa = 0.29 [Å/eV] × mu_GPa × gamma_usf [J/m²] × RMSAD [Å]
```

### Plot ternary diagram (locally)
Always plots YS_GPa. Requires YS_GPa column in the predict CSV.
```bash
python scripts/plot_ternary.py TiNbV TiNbMo           # smooth fill, 900 MPa contour
python scripts/plot_ternary.py TiNbV --ys-contour 900 # contour line at 900 MPa (default)
python scripts/plot_ternary.py TiNbV --contour         # contour fill style
```
Output saved to `data/output/ternary_YS_{systems}.png`.

### Supported elements
Ti, Zr, Hf, V, Nb, Ta, Mo, W, Re, Ru

### System name parsing
- Concatenated: `TiNbV`, `ZrHfTaNbMo`
- Dash-separated fallback: `Ti-Nb-V`
- 2-char symbols (Ti,Zr,Hf,Nb,Ta,Mo,Re,Ru) parsed before 1-char (V,W)

---

## Composition grid sizes
| System | Step | Compositions |
|---|---|---|
| Binary | 1% | 99 |
| Ternary | 1% | 4,851 |
| Quaternary | 5% | 969 |
| Quinary | 10% | 126 |
Step must be a divisor of 100.

---

## Output CSV columns
| Column | Description |
|---|---|
| index, sample | row index, composition label (e.g. Ti1Nb1V98) |
| Ti…Ru | element fractions (0–1) |
| RMSAD | lattice distortion (Å) |
| mu_GPa | isotropic shear modulus via VRH + Vegard |
| gamma_usf | unstable stacking fault energy (J/m²) — from HTP merge |
| YS_GPa | yield strength (GPa) — computed if gamma_usf present |
