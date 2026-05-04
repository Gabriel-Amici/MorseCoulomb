# AGENTS.md

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or python_venv.bat / python_venv.ps1 on Windows
pip install -r src/requirements.txt
pip install -e src/
```

## Running Scripts

Scripts live in `scripts/` organized by topic (ionization, poincare, quantum_time_evolution, etc.). Run directly with Python.

## Testing

Tests in `tests/`. Run with `python -m pytest tests/` or directly. Test subdirectories cover bspline, fgh, ionization_stability, qmsolve, and qutip experiments.

## Architecture

- `src/emerald/` - Installable library (`emerald` v1.0.2)
  - `potentials/` - Potential functions: coulomb, soft-Coulomb (sc), Morse-soft-Coulomb (msc), normalized msc
  - `classical/` - Dynamics per potential: unperturbed, driven, poincare, ionization (+ normalized_msc variants)
  - `quantum/` - Quantum implementations: coulomb/sc/msc unperturbed, msc_coupled, coupling_utils
- `scripts/` - Analysis scripts grouped by physics topic
- `notebooks/` - Jupyter notebooks
- `results/` - Output figures/data (gitignored: png, svg, pdf, mp4, json)
- `extra/fortran/` - Legacy Fortran code
- `legacy/scripts/` - Deprecated scripts

## Key Notes

- `numba` is used for JIT compilation — first runs will be slow
- VS Code settings auto-add `./src` to Python analysis extraPaths
- The `quantum/` module has a typo in its init file (`__.init__.py` instead of `__init__.py`)
- Root-level `.gitignore` excludes common output artifacts (images, videos, json, archives)
