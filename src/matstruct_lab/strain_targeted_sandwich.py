"""Compatibility interface for strain-targeted sandwich generation.

New code should use :mod:`matstruct_lab.sandwich` directly.  This module keeps
small convenience functions for the older Au/Ag/Cu/Pt/Pd/Ni + MoS2 workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import fcc111
from ase.io import read, write

from matstruct_lab.hetero_strain import inplane_cell, prefixed_strain_summary
from matstruct_lab.lattice_match import make_2d_supercell
from matstruct_lab.sandwich import (
    SandwichSplit,
    build_homogeneous_sandwich_from_split,
    find_homogeneous_sandwich_splits,
    split_records,
)


DEFAULT_FCC_A = {
    "Au": 4.0782,
    "Ag": 4.0862,
    "Cu": 3.6149,
    "Pt": 3.9239,
    "Pd": 3.8907,
    "Ni": 3.5240,
}


def safe_token(x: float | None) -> str:
    if x is None:
        return "none"
    return f"{float(x):.3f}".replace("-", "m").replace(".", "p")


def check_2d_cell(atoms: Atoms, name: str, tol: float = 1e-6) -> None:
    cell = atoms.cell.array
    if abs(cell[0, 2]) > tol or abs(cell[1, 2]) > tol:
        raise ValueError(
            f"{name}: first two lattice vectors have nonzero z components. "
            "Reorient so a,b are in-plane and c is out-of-plane."
        )


def find_structure_file(structure_dir: Path, name: str) -> Path | None:
    candidates = [
        structure_dir / f"{name}_111.cif",
        structure_dir / f"{name}_111.vasp",
        structure_dir / f"{name}_111.POSCAR",
        structure_dir / f"{name}111.cif",
        structure_dir / f"{name}111.vasp",
        structure_dir / f"{name}.cif",
        structure_dir / f"{name}.vasp",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_or_build_fcc111(
    metal: str,
    structure_dir: Path,
    fcc_a: dict[str, float] | None = None,
    layers: int = 4,
) -> tuple[Atoms, str]:
    path = find_structure_file(structure_dir, metal)
    if path is not None:
        atoms = read(path)
        atoms.set_pbc([True, True, False])
        check_2d_cell(atoms, metal)
        return atoms, f"file:{path}"

    fcc_a = DEFAULT_FCC_A if fcc_a is None else fcc_a
    if metal not in fcc_a:
        raise ValueError(f"No structure file or fcc lattice constant for {metal}")

    atoms = fcc111(metal, size=(1, 1, layers), a=fcc_a[metal], vacuum=10.0, orthogonal=False)
    atoms.set_pbc([True, True, False])
    check_2d_cell(atoms, metal)
    return atoms, f"generated:fcc111(a={fcc_a[metal]}, layers={layers})"


def write_all_formats(atoms: Atoms, outdir: Path, stem: str) -> dict[str, str]:
    paths = {
        "vasp": outdir / f"{stem}.vasp",
        "cif": outdir / f"{stem}.cif",
        "xyz": outdir / f"{stem}.xyz",
    }
    write(paths["vasp"], atoms, format="vasp", direct=True)
    write(paths["cif"], atoms)
    write(paths["xyz"], atoms, format="extxyz")
    return {k: str(v) for k, v in paths.items()}


def split_metadata(
    metal: str,
    middle_name: str,
    metal_atoms: Atoms,
    middle_atoms: Atoms,
    split: SandwichSplit,
    metal_source: str = "unknown",
    middle_source: str = "unknown",
) -> dict[str, Any]:
    metal_sc = make_2d_supercell(metal_atoms, split.bottom_matrix)
    middle_sc = make_2d_supercell(middle_atoms, split.middle_matrix)
    metal_cell = inplane_cell(metal_sc)
    middle_cell = inplane_cell(middle_sc)
    target = split.target_cell

    return {
        "metal": metal,
        "middle": middle_name,
        "structure_type": f"{metal}/{middle_name}/{metal}",
        "metal_source": metal_source,
        "middle_source": middle_source,
        "metal_input_atoms": len(metal_atoms),
        "middle_input_atoms": len(middle_atoms),
        **split.as_dict(),
        **prefixed_strain_summary(metal_cell, target, "metal"),
        **prefixed_strain_summary(middle_cell, target, middle_name.lower()),
    }


def generate_strain_targeted_metal_mos2_sandwiches(
    structure_dir: str | Path = "../examples/structures",
    mos2_filename: str = "MoS2.cif",
    outdir: str | Path = "strain_targeted_metal_mos2_sandwiches",
    metals: list[str] | None = None,
    distances: list[float] | np.ndarray | None = None,
    max_entry: int = 4,
    max_area: int = 30,
    max_atoms: int = 250,
    weight_step: float = 0.01,
    max_total_abs_strain_percent: float = 6.0,
    limit_per_metal: int = 20,
    vacuum: float = 20.0,
    fcc111_layers: int = 4,
    registry: str = "Metal_on_Mo",
    write_structures: bool = True,
    **_ignored_legacy_kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate homogeneous metal/MoS2/metal sandwiches using the unified split search.

    Returns
    -------
    summary_df
        Candidate strain splits, one row per unique common target cell.
    metadata_df
        One row per written structure and distance.
    skipped_df
        Empty compatibility table. Filtering is handled during split search.
    """
    structure_dir = Path(structure_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metals = ["Au", "Ag", "Cu", "Pt", "Pd", "Ni"] if metals is None else metals
    distances = np.arange(2.5, 4.0001, 0.25) if distances is None else np.asarray(distances, dtype=float)

    mos2_path = structure_dir / mos2_filename
    if not mos2_path.exists():
        raise FileNotFoundError(f"Could not find MoS2 file: {mos2_path}")

    mos2 = read(mos2_path)
    mos2.set_pbc([True, True, False])
    check_2d_cell(mos2, "MoS2")

    summary_records: list[dict[str, Any]] = []
    metadata_records: list[dict[str, Any]] = []

    for metal in metals:
        metal_atoms, metal_source = load_or_build_fcc111(
            metal=metal,
            structure_dir=structure_dir,
            layers=fcc111_layers,
        )

        splits = find_homogeneous_sandwich_splits(
            electrode=metal_atoms,
            middle=mos2,
            max_entry=max_entry,
            max_area=max_area,
            max_atoms=max_atoms,
            weight_step=weight_step,
            max_total_abs_strain_percent=max_total_abs_strain_percent,
            limit=limit_per_metal,
        )

        for split_index, split in enumerate(splits):
            base = split_metadata(
                metal=metal,
                middle_name="MoS2",
                metal_atoms=metal_atoms,
                middle_atoms=mos2,
                split=split,
                metal_source=metal_source,
                middle_source=str(mos2_path),
            )
            base["split_index"] = split_index
            summary_records.append(base)

            for d in distances:
                atoms = build_homogeneous_sandwich_from_split(
                    electrode=metal_atoms,
                    middle=mos2,
                    split=split,
                    electrode_middle_distance=float(d),
                    vacuum=vacuum,
                    middle_registry_material="MoS2",
                    middle_registry=registry,
                )

                stem = (
                    f"{metal}_MoS2_{metal}"
                    f"_split{split_index:03d}"
                    f"_d{safe_token(float(d))}A"
                    f"_{len(atoms)}atoms"
                    f"_M{safe_token(split.bottom_strain_percent)}pct"
                    f"_MoS2{safe_token(split.middle_strain_percent)}pct"
                )

                files = write_all_formats(atoms, outdir, stem) if write_structures else {}
                rec = dict(base)
                rec.update({
                    "interlayer_distance_A": float(d),
                    "actual_atoms": len(atoms),
                    "stem": stem,
                    "vasp": files.get("vasp"),
                    "cif": files.get("cif"),
                    "xyz": files.get("xyz"),
                })
                metadata_records.append(rec)

    summary_df = pd.DataFrame(summary_records)
    metadata_df = pd.DataFrame(metadata_records)
    skipped_df = pd.DataFrame([])

    summary_df.to_csv(outdir / "unique_targets_strain_targeted_summary.csv", index=False)
    metadata_df.to_csv(outdir / "all_generated_structures_metadata.csv", index=False)
    skipped_df.to_csv(outdir / "skipped_targets.csv", index=False)

    summary_df.to_json(outdir / "unique_targets_strain_targeted_summary.json", orient="records", indent=2)
    metadata_df.to_json(outdir / "all_generated_structures_metadata.json", orient="records", indent=2)
    skipped_df.to_json(outdir / "skipped_targets.json", orient="records", indent=2)

    return summary_df, metadata_df, skipped_df


__all__ = [
    "DEFAULT_FCC_A",
    "safe_token",
    "check_2d_cell",
    "find_structure_file",
    "load_or_build_fcc111",
    "write_all_formats",
    "split_metadata",
    "generate_strain_targeted_metal_mos2_sandwiches",
    "find_homogeneous_sandwich_splits",
    "build_homogeneous_sandwich_from_split",
    "split_records",
]
