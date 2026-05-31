"""ASE-native heterostructure strain analysis.

Run as CLI:
    python -m matstruct_lab.hetero_strain --bottom examples/structures/Au_111.cif --top examples/structures/MoS2.cif --bottom-name Au111 --top-name MoS2

Import in notebooks:
    from matstruct_lab.hetero_strain import layer_strain_records, match_strain_records
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write

from matstruct_lab.lattice_match import (
    Match2D,
    find_2d_matches,
    make_2d_supercell,
    print_matches,
)
from matstruct_lab.heterostructures import build_heterostructure, shared_inplane_cell


def inplane_cell(atoms: Atoms) -> np.ndarray:
    """Return ASE row-vector 2D in-plane cell: rows are a and b."""
    return np.asarray(atoms.cell.array, dtype=float)[:2, :2]


def cell_summary(cell: np.ndarray) -> dict[str, float]:
    """Return a, b, gamma, and area for a 2D row-vector cell."""
    cell = np.asarray(cell, dtype=float)
    a = cell[0]
    b = cell[1]

    la = float(np.linalg.norm(a))
    lb = float(np.linalg.norm(b))
    if la == 0.0 or lb == 0.0:
        raise ValueError("Zero-length in-plane lattice vector.")

    cosang = float(np.dot(a, b) / (la * lb))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    gamma = float(np.degrees(np.arccos(cosang)))
    area = float(abs(np.linalg.det(cell)))

    return {
        "a_A": la,
        "b_A": lb,
        "gamma_deg": gamma,
        "area_A2": area,
    }


def polar_decomposition_2d(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R, U from F = R @ U."""
    C = F.T @ F
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 0.0, None)
    U = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    R = F @ np.linalg.inv(U)
    return R, U


def strain_summary(
    reference_cell: np.ndarray,
    target_cell: np.ndarray,
    reference: str = "reference",
    target: str = "target",
) -> dict[str, Any]:
    """Compute 2D strain required to deform reference_cell into target_cell."""
    reference_cell = np.asarray(reference_cell, dtype=float)
    target_cell = np.asarray(target_cell, dtype=float)

    ref = cell_summary(reference_cell)
    tar = cell_summary(target_cell)

    # Same convention as lattice_match.deformation_strain:
    # reference_cell @ F = target_cell
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


def match_supercell_cells(bottom: Atoms, top: Atoms, match: Match2D) -> tuple[np.ndarray, np.ndarray]:
    """Return bottom/top matched-supercell in-plane cells before heterostructure strain."""
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
    """Return top->bottom and bottom->top strain records for a selected match."""
    bottom_cell, top_cell = match_supercell_cells(bottom, top, match)

    records = []

    top_to_bottom = strain_summary(
        top_cell,
        bottom_cell,
        reference=f"{top_name}_matched_supercell",
        target=f"{bottom_name}_matched_supercell",
    )
    top_to_bottom.update({
        "match_index": match_index,
        "strain_direction": f"{top_name}_to_{bottom_name}",
        "bottom_matrix": match.bottom_matrix.tolist(),
        "top_matrix": match.top_matrix.tolist(),
        "bottom_area_multiplier": match.bottom_area_multiplier,
        "top_area_multiplier": match.top_area_multiplier,
        "total_atoms": match.total_atoms,
    })
    records.append(top_to_bottom)

    bottom_to_top = strain_summary(
        bottom_cell,
        top_cell,
        reference=f"{bottom_name}_matched_supercell",
        target=f"{top_name}_matched_supercell",
    )
    bottom_to_top.update({
        "match_index": match_index,
        "strain_direction": f"{bottom_name}_to_{top_name}",
        "bottom_matrix": match.bottom_matrix.tolist(),
        "top_matrix": match.top_matrix.tolist(),
        "bottom_area_multiplier": match.bottom_area_multiplier,
        "top_area_multiplier": match.top_area_multiplier,
        "total_atoms": match.total_atoms,
    })
    records.append(bottom_to_top)

    return records


def layer_strain_records(
    bottom: Atoms,
    top: Atoms,
    match: Match2D,
    bottom_name: str = "bottom",
    top_name: str = "top",
    shared_top_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Return actual layer-by-layer strains for strain='top', 'bottom', and 'shared'."""
    bottom_cell, top_cell = match_supercell_cells(bottom, top, match)
    shared_cell = shared_inplane_cell(bottom_cell, top_cell, top_weight=shared_top_weight)

    cases = [
        (f"{top_name}_strained_to_{bottom_name}", "top", bottom_name, False, bottom_cell, bottom_cell, "bottom matched supercell", "bottom matched supercell"),
        (f"{top_name}_strained_to_{bottom_name}", "top", top_name, True, top_cell, bottom_cell, "top matched supercell", "bottom matched supercell"),
        (f"{bottom_name}_strained_to_{top_name}", "bottom", bottom_name, True, bottom_cell, top_cell, "bottom matched supercell", "top matched supercell"),
        (f"{bottom_name}_strained_to_{top_name}", "bottom", top_name, False, top_cell, top_cell, "top matched supercell", "top matched supercell"),
        (f"{bottom_name}_{top_name}_shared_strain", "shared", bottom_name, True, bottom_cell, shared_cell, "bottom matched supercell", "shared intermediate cell"),
        (f"{bottom_name}_{top_name}_shared_strain", "shared", top_name, True, top_cell, shared_cell, "top matched supercell", "shared intermediate cell"),
    ]

    records = []
    for structure, mode, layer, strained, source, target, source_label, target_label in cases:
        rec = strain_summary(source, target, reference=source_label, target=target_label)
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


def main() -> None:
    p = argparse.ArgumentParser(description="Search 2D lattice matches, build heterostructures, and calculate strain.")
    p.add_argument("--bottom", required=True)
    p.add_argument("--top", required=True)
    p.add_argument("--bottom-name", default="bottom")
    p.add_argument("--top-name", default="top")
    p.add_argument("--outdir", default="heterostructure_outputs")
    p.add_argument("--match-index", type=int, default=0)
    p.add_argument("--max-entry", type=int, default=2)
    p.add_argument("--max-area", type=int, default=10)
    p.add_argument("--max-strain", type=float, default=0.05, help="Fraction. 0.05 means 5 percent.")
    p.add_argument("--max-atoms", type=int, default=100)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--interlayer-distance", type=float, default=3.3)
    p.add_argument("--vacuum", type=float, default=20.0)
    p.add_argument("--shared-top-weight", type=float, default=0.5)
    p.add_argument("--formats", default="cif,xyz,vasp")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]

    bottom = read(args.bottom)
    top = read(args.top)

    matches = find_2d_matches(
        bottom=bottom,
        top=top,
        max_entry=args.max_entry,
        max_area=args.max_area,
        max_strain=args.max_strain,
        max_atoms=args.max_atoms,
        limit=args.limit,
    )

    if not matches:
        raise SystemExit("No matches found. Increase --max-strain, --max-area, --max-entry, or --max-atoms.")

    print_matches(matches)

    all_match_records = []
    for i, m in enumerate(matches):
        rec = m.as_dict()
        rec.update({
            "match_index": i,
            "bottom_name": args.bottom_name,
            "top_name": args.top_name,
        })
        all_match_records.append(rec)

    write_csv_records(outdir / "all_lattice_matches_summary.csv", all_match_records)
    write_json_records(outdir / "all_lattice_matches_summary.json", all_match_records)

    if args.match_index < 0 or args.match_index >= len(matches):
        raise SystemExit(f"--match-index {args.match_index} is outside available range 0-{len(matches)-1}.")

    match = matches[args.match_index]

    selected_records = match_strain_records(
        bottom=bottom,
        top=top,
        match=match,
        match_index=args.match_index,
        bottom_name=args.bottom_name,
        top_name=args.top_name,
    )
    write_csv_records(outdir / "selected_match_bidirectional_strain.csv", selected_records)
    write_json_records(outdir / "selected_match_bidirectional_strain.json", selected_records)

    top_to_bottom = build_heterostructure(
        bottom=bottom,
        top=top,
        match=match,
        interlayer_distance=args.interlayer_distance,
        vacuum=args.vacuum,
        strain="top",
    )

    bottom_to_top = build_heterostructure(
        bottom=bottom,
        top=top,
        match=match,
        interlayer_distance=args.interlayer_distance,
        vacuum=args.vacuum,
        strain="bottom",
    )

    shared = build_heterostructure(
        bottom=bottom,
        top=top,
        match=match,
        interlayer_distance=args.interlayer_distance,
        vacuum=args.vacuum,
        strain="shared",
        shared_top_weight=args.shared_top_weight,
    )

    structures = {
        f"{args.top_name}_strained_to_{args.bottom_name}": top_to_bottom,
        f"{args.bottom_name}_strained_to_{args.top_name}": bottom_to_top,
        f"{args.bottom_name}_{args.top_name}_shared_strain": shared,
    }

    for stem, atoms in structures.items():
        write_structure_formats(atoms, outdir, stem, formats)

    layer_records = layer_strain_records(
        bottom=bottom,
        top=top,
        match=match,
        bottom_name=args.bottom_name,
        top_name=args.top_name,
        shared_top_weight=args.shared_top_weight,
    )
    write_csv_records(outdir / "selected_match_actual_layer_strains.csv", layer_records)
    write_json_records(outdir / "selected_match_actual_layer_strains.json", layer_records)

    print(f"Wrote outputs to: {outdir.resolve()}")
    print("Key strain files:")
    print(outdir / "all_lattice_matches_summary.csv")
    print(outdir / "selected_match_bidirectional_strain.csv")
    print(outdir / "selected_match_actual_layer_strains.csv")


if __name__ == "__main__":
    main()
