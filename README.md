# matstruct-lab

Utilities, examples, and notebooks for setting up atomistic structures, defected structures, and heterostructures using ASE and pymatgen.

## Purpose

This repository is intended for practical structure generation workflows, including:

- bulk and surface structure setup
- supercell construction
- point defects and substitutions
- vacancies and interstitials
- heterostructure setup
- ASE/pymatgen conversion utilities
- tutorial notebooks for reproducible workflows

## Layout

- `notebooks/`: workflow notebooks and tutorials
- `src/matstruct_lab/`: reusable Python functions
- `tests/`: tests for reusable code
- `data/raw/`: original input structures
- `data/processed/`: generated structures
- `examples/structures/`: basic structure examples
- `examples/defects/`: defect structure examples
- `examples/heterostructures/`: heterostructure examples

## Setup

```bash
conda env create -f environment.yml
conda activate matstruct-lab
pip install -e .
jupyter lab
```
