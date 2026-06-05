"""Canonical 2D strain and layer-lattice utilities.

This module is intentionally ASE-native.  It is the shared source for:
- in-plane cell metrics
- rotation-invariant strain summaries
- matched-supercell strain records
- layer-specific effective lattice extraction after relaxation
- CSV/JSON helpers

Conventions
-----------
ASE row-vector convention is used throughout:
    cell2 = atoms.cell.array[:2, :2]
where row 0 is the in-plane a vector and row 1 is the in-plane b vector.

A strain summary maps reference_cell -> target_cell using:
    reference_cell @ F = target_cell
and reports principal strains from the singular values of F.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write

from matstruct_lab.lattice_match import Match2D, make_2d_supercell
from matstruct_lab.heterostructures import shared_inplane_cell


Array2D = np.ndarray


def inplane_cell(atoms: Atoms) -> np.ndarray:
    """Return the ASE row-vector 2D in-plane cell."""
    return np.asarray(atoms.cell.array, dtype=float)[:2, :2]


def cell_summary(cell: Array2D) -> dict[str, float]:
    """Return a, b, gamma, and area for a 2D row-vector cell."""
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 in-plane cell, got {cell.shape}")

    a = cell[0]
    b = cell[1]
    la = float(np.linalg.norm(a))
    lb = float(np.linalg.norm(b))

    if la == 0.0 or lb == 0.0:
        raise ValueError("Zero-length in-plane lattice vector.")

    cosang = float(np.dot(a, b) / (la * lb))
    cosang = float(np.clip(cosang, -1.0, 1.0))

    return {
        "a_A": la,
        "b_A": lb,
        "gamma_deg": float(np.degrees(np.arccos(cosang))),
        "area_A2": float(abs(np.linalg.det(cell))),
    }


def polar_decomposition_2d(F: Array2D) -> tuple[np.ndarray, np.ndarray]:
    """Return R, U from F = R @ U."""
    F = np.asarray(F, dtype=float)
    C = F.T @ F
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 0.0, None)
    U = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    R = F @ np.linalg.inv(U)
    return R, U


def strain_summary(
    reference_cell: Array2D,
    target_cell: Array2D,
    reference: str = "reference",
    target: str = "target",
) -> dict[str, Any]:
    """Compute rotation-invariant 2D strain required to map reference_cell to target_cell."""
    reference_cell = np.asarray(reference_cell, dtype=float)
    target_cell = np.asarray(target_cell, dtype=float)

    ref = cell_summary(reference_cell)
    tar = cell_summary(target_cell)

    # ASE row-vector convention: reference_cell @ F = target_cell
    F = np.linalg.solve(reference_cell, target_cell)
    singular_values = np.linalg.svd(F, compute_uv=False)
    principal = singular_values - 1.0

    R, U = polar_decomposition_2d(F)
    stretch_strain = U - np.eye(2)

    signed_area_strain = (tar["area_A2"] - ref["area_A2"]) / ref["area_A2"]

    return {
        "reference": reference,
        "target": target,
        "a_ref_A": ref["a_A"],
        "b_ref_A": ref["b_A"],
        "gamma_ref_deg": ref["gamma_deg"],
        "area_ref_A2": ref["area_A2"],
        "a_target_A": tar["a_A"],
        "b_target_A": tar["b_A"],
        "gamma_target_deg": tar["gamma_deg"],
        "area_target_A2": tar["area_A2"],
        "eps_a_percent": 100.0 * (tar["a_A"] - ref["a_A"]) / ref["a_A"],
        "eps_b_percent": 100.0 * (tar["b_A"] - ref["b_A"]) / ref["b_A"],
        "delta_gamma_deg": tar["gamma_deg"] - ref["gamma_deg"],
        "signed_area_strain_percent": 100.0 * signed_area_strain,
        "abs_area_strain_percent": 100.0 * abs(signed_area_strain),
        "stretch_xx_percent": 100.0 * float(stretch_strain[0, 0]),
        "stretch_yy_percent": 100.0 * float(stretch_strain[1, 1]),
        "stretch_xy_percent": 100.0 * float(stretch_strain[0, 1]),
        "principal_strain_1_percent": 100.0 * float(principal[0]),
        "principal_strain_2_percent": 100.0 * float(principal[1]),
        "rms_principal_strain_percent": 100.0 * float(np.sqrt(np.mean(principal**2))),
        "max_abs_principal_strain_percent": 100.0 * float(np.max(np.abs(principal))),
        "deformation_gradient": F.tolist(),
        "rotation_matrix": R.tolist(),
        "stretch_tensor": U.tolist(),
    }


def prefixed_strain_summary(reference_cell: Array2D, target_cell: Array2D, prefix: str) -> dict[str, Any]:
    """Return a compact strain summary with fields prefixed for table joins."""
    rec = strain_summary(reference_cell, target_cell)
    keep = {
        "principal_strain_1_percent",
        "principal_strain_2_percent",
        "rms_principal_strain_percent",
        "max_abs_principal_strain_percent",
        "signed_area_strain_percent",
        "deformation_gradient",
    }
    return {f"{prefix}_{k}": v for k, v in rec.items() if k in keep}


def match_supercell_cells(bottom: Atoms, top: Atoms, match: Match2D) -> tuple[np.ndarray, np.ndarray]:
    """Return bottom/top matched-supercell in-plane cells before common-cell strain."""
    bottom_sc = make_2d_supercell(bottom, match.bottom_matrix)
    top_sc = make_2d_supercell(top, match.top_matrix)
    return inplane_cell(bottom_sc), inplane_cell(top_sc)


def match_strain_records(
    bottom: Atoms,
    top: Atoms,
    match: Match2D,
    match_index: int = 0,
    bottom_name: str = "bottom",
    top_name: str = "top",
) -> list[dict[str, Any]]:
    """Return top->bottom and bottom->top strain records for one selected match."""
    bottom_cell, top_cell = match_supercell_cells(bottom, top, match)

    records = []
    for ref_cell, target_cell, direction, ref_name, target_name in [
        (top_cell, bottom_cell, f"{top_name}_to_{bottom_name}", top_name, bottom_name),
        (bottom_cell, top_cell, f"{bottom_name}_to_{top_name}", bottom_name, top_name),
    ]:
        rec = strain_summary(
            ref_cell,
            target_cell,
            reference=f"{ref_name}_matched_supercell",
            target=f"{target_name}_matched_supercell",
        )
        rec.update({
            "match_index": match_index,
            "strain_direction": direction,
            "bottom_matrix": match.bottom_matrix.tolist(),
            "top_matrix": match.top_matrix.tolist(),
            "bottom_area_multiplier": match.bottom_area_multiplier,
            "top_area_multiplier": match.top_area_multiplier,
            "total_atoms": match.total_atoms,
        })
        records.append(rec)

    return records


def layer_strain_records(
    bottom: Atoms,
    top: Atoms,
    match: Match2D,
    bottom_name: str = "bottom",
    top_name: str = "top",
    shared_top_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Return layer-by-layer strains for fixed-bottom, fixed-top, and shared cases."""
    bottom_cell, top_cell = match_supercell_cells(bottom, top, match)
    shared_cell = shared_inplane_cell(bottom_cell, top_cell, top_weight=shared_top_weight)

    cases = [
        (f"{top_name}_strained_to_{bottom_name}", "top", bottom_name, False, bottom_cell, bottom_cell),
        (f"{top_name}_strained_to_{bottom_name}", "top", top_name, True, top_cell, bottom_cell),
        (f"{bottom_name}_strained_to_{top_name}", "bottom", bottom_name, True, bottom_cell, top_cell),
        (f"{bottom_name}_strained_to_{top_name}", "bottom", top_name, False, top_cell, top_cell),
        (f"{bottom_name}_{top_name}_shared_strain", "shared", bottom_name, True, bottom_cell, shared_cell),
        (f"{bottom_name}_{top_name}_shared_strain", "shared", top_name, True, top_cell, shared_cell),
    ]

    records: list[dict[str, Any]] = []
    for structure, mode, layer, strained, source, target_cell in cases:
        rec = strain_summary(source, target_cell, reference="source_cell", target="target_cell")
        rec.update({
            "structure": structure,
            "strain_mode": mode,
            "layer": layer,
            "strained": strained,
            "bottom_matrix": match.bottom_matrix.tolist(),
            "top_matrix": match.top_matrix.tolist(),
        })
        records.append(rec)

    return records


def selected_indices_by_symbol_and_z(
    atoms: Atoms,
    symbol: str,
    z_min: float | None = None,
    z_max: float | None = None,
) -> list[int]:
    """Return atom indices matching a symbol and optional z window."""
    idx: list[int] = []
    symbols = atoms.get_chemical_symbols()

    for i, sym in enumerate(symbols):
        if sym != symbol:
            continue
        z = float(atoms.positions[i, 2])
        if z_min is not None and z < z_min:
            continue
        if z_max is not None and z > z_max:
            continue
        idx.append(i)

    if not idx:
        raise ValueError(f"No {symbol} atoms found in the requested z window.")

    return idx


def select_z_plane(
    atoms: Atoms,
    symbol: str,
    plane: str = "top",
    z_min: float | None = None,
    z_max: float | None = None,
    z_tol: float = 0.35,
) -> Atoms:
    """Select one approximate z-plane for same-species in-plane distances."""
    idx = selected_indices_by_symbol_and_z(atoms, symbol, z_min=z_min, z_max=z_max)
    zvals = np.array([atoms.positions[i, 2] for i in idx], dtype=float)

    if plane == "top":
        z0 = float(zvals.max())
    elif plane == "bottom":
        z0 = float(zvals.min())
    elif plane == "middle":
        z0 = float(np.median(zvals))
    else:
        raise ValueError("plane must be 'top', 'bottom', or 'middle'.")

    keep = [i for i in idx if abs(float(atoms.positions[i, 2]) - z0) <= z_tol]
    if len(keep) < 2:
        raise ValueError(f"Selected z-plane has fewer than two {symbol} atoms.")

    return atoms[keep]


def same_species_lattice_a_2d(
    atoms: Atoms,
    symbol: str,
    z_min: float | None = None,
    z_max: float | None = None,
    plane: str | None = None,
    z_tol: float = 0.35,
    shell_tol: float = 0.10,
) -> float:
    """Estimate effective 2D lattice parameter from nearest same-species distances."""
    if plane is None:
        idx = selected_indices_by_symbol_and_z(atoms, symbol, z_min=z_min, z_max=z_max)
        sub = atoms[idx]
    else:
        sub = select_z_plane(atoms, symbol, plane=plane, z_min=z_min, z_max=z_max, z_tol=z_tol)

    flat = sub.copy()
    pos = flat.positions.copy()
    pos[:, 2] = 0.0
    flat.positions = pos
    flat.set_pbc([True, True, False])

    d = flat.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)

    vals = d[np.isfinite(d)]
    vals = vals[vals > 1e-6]
    if len(vals) == 0:
        raise ValueError(f"No valid same-species distances found for {symbol}.")

    d0 = float(vals.min())
    shell = vals[np.abs(vals - d0) <= shell_tol * d0]
    return float(np.median(shell)) if len(shell) else d0


def mismatch_percent(reference_a: float, measured_a: float) -> float:
    """Return 100 * (measured-reference)/reference."""
    return 100.0 * (float(measured_a) - float(reference_a)) / float(reference_a)


def layer_lattice_report(
    atoms: Atoms,
    metal_symbol: str,
    middle_material: str = "MoS2",
    bottom_z: tuple[float, float] | None = None,
    middle_z: tuple[float, float] | None = None,
    top_z: tuple[float, float] | None = None,
    reference_metal_a: float | None = None,
    reference_middle_a: float | None = None,
) -> dict[str, float]:
    """Extract effective layer lattice constants from a relaxed metal/layer/metal stack."""
    bz0, bz1 = bottom_z if bottom_z is not None else (None, None)
    mz0, mz1 = middle_z if middle_z is not None else (None, None)
    tz0, tz1 = top_z if top_z is not None else (None, None)

    bottom_a = same_species_lattice_a_2d(atoms, metal_symbol, z_min=bz0, z_max=bz1, plane="top")
    top_a = same_species_lattice_a_2d(atoms, metal_symbol, z_min=tz0, z_max=tz1, plane="bottom")

    if middle_material == "MoS2":
        middle_a = same_species_lattice_a_2d(atoms, "Mo", z_min=mz0, z_max=mz1, plane=None)
    elif middle_material == "hBN":
        b_a = same_species_lattice_a_2d(atoms, "B", z_min=mz0, z_max=mz1, plane=None)
        n_a = same_species_lattice_a_2d(atoms, "N", z_min=mz0, z_max=mz1, plane=None)
        middle_a = 0.5 * (b_a + n_a)
    else:
        raise ValueError("middle_material currently supports 'MoS2' or 'hBN'.")

    out = {
        "bottom_metal_a_eff_A": bottom_a,
        "middle_a_eff_A": middle_a,
        "top_metal_a_eff_A": top_a,
        "bottom_vs_middle_mismatch_percent": mismatch_percent(middle_a, bottom_a),
        "top_vs_middle_mismatch_percent": mismatch_percent(middle_a, top_a),
    }

    if reference_metal_a is not None:
        out["bottom_metal_strain_vs_ref_percent"] = mismatch_percent(reference_metal_a, bottom_a)
        out["top_metal_strain_vs_ref_percent"] = mismatch_percent(reference_metal_a, top_a)

    if reference_middle_a is not None:
        out["middle_strain_vs_ref_percent"] = mismatch_percent(reference_middle_a, middle_a)

    return out


def write_json_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    Path(path).write_text(json.dumps(records, indent=2), encoding="utf-8")


def write_csv_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    if not records:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for rec in records:
        for key in rec:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_structure_formats(atoms: Atoms, outdir: Path, stem: str, formats: list[str]) -> None:
    if "cif" in formats:
        write(outdir / f"{stem}.cif", atoms)
    if "xyz" in formats:
        write(outdir / f"{stem}.xyz", atoms, format="extxyz")
    if "vasp" in formats:
        write(outdir / f"{stem}.vasp", atoms, format="vasp", direct=True)
