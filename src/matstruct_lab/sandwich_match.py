from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from matstruct_lab.lattice_match import Match2D, find_2d_matches, make_2d_supercell
from matstruct_lab.heterostructures import shared_inplane_cell


@dataclass(frozen=True)
class SandwichMatch:
    """A 2D lattice match with electrode/middle/electrode atom counting."""

    match: Match2D
    match_index: int
    sandwich_atoms: int
    electrode_atoms_total: int
    middle_atoms_total: int

    def as_dict(self) -> dict[str, Any]:
        d = self.match.as_dict()
        d.update(
            {
                "match_index": self.match_index,
                "sandwich_atoms": self.sandwich_atoms,
                "electrode_atoms_total": self.electrode_atoms_total,
                "middle_atoms_total": self.middle_atoms_total,
                "electrode_matrix": self.match.bottom_matrix.tolist(),
                "middle_matrix": self.match.top_matrix.tolist(),
                "electrode_area_multiplier": self.match.bottom_area_multiplier,
                "middle_area_multiplier": self.match.top_area_multiplier,
            }
        )
        return d


def sandwich_atom_count(electrode: Atoms, middle: Atoms, match: Match2D) -> int:
    """Return atom count for electrode/middle/electrode sandwich."""
    return (
        2 * len(electrode) * match.bottom_area_multiplier
        + len(middle) * match.top_area_multiplier
    )


def find_2d_sandwich_matches(
    electrode: Atoms,
    middle: Atoms,
    max_entry: int = 4,
    max_area: int = 20,
    max_strain: float = 0.05,
    max_atoms: int = 200,
    limit: int = 20,
    search_limit: int = 1000,
    sort_by: str = "atoms_then_strain",
) -> list[SandwichMatch]:
    """
    Find 2D matches for electrode/middle/electrode stacks.

    max_atoms is the final sandwich atom count:
        2 * electrode_supercell_atoms + middle_supercell_atoms

    max_strain is a fraction:
        0.03 = 3 percent, 0.05 = 5 percent
    """
    raw_matches = find_2d_matches(
        bottom=electrode,
        top=middle,
        max_entry=max_entry,
        max_area=max_area,
        max_strain=max_strain,
        max_atoms=max_atoms,
        limit=search_limit,
    )

    sandwich_matches: list[SandwichMatch] = []

    for i, match in enumerate(raw_matches):
        electrode_atoms_total = 2 * len(electrode) * match.bottom_area_multiplier
        middle_atoms_total = len(middle) * match.top_area_multiplier
        total = electrode_atoms_total + middle_atoms_total

        if total <= max_atoms:
            sandwich_matches.append(
                SandwichMatch(
                    match=match,
                    match_index=i,
                    sandwich_atoms=total,
                    electrode_atoms_total=electrode_atoms_total,
                    middle_atoms_total=middle_atoms_total,
                )
            )

    if sort_by == "atoms_then_strain":
        key = lambda sm: (
            sm.sandwich_atoms,
            sm.match.max_strain,
            sm.match.rms_strain,
            sm.match.area_strain,
        )
    elif sort_by == "strain_then_atoms":
        key = lambda sm: (
            sm.match.max_strain,
            sm.match.rms_strain,
            sm.sandwich_atoms,
            sm.match.area_strain,
        )
    else:
        raise ValueError("sort_by must be 'atoms_then_strain' or 'strain_then_atoms'")

    sandwich_matches.sort(key=key)
    return sandwich_matches[:limit]


def print_sandwich_matches(matches: list[SandwichMatch]) -> None:
    for i, sm in enumerate(matches):
        m = sm.match
        print(f"[{i}] original match index: {sm.match_index}")
        print(f"  sandwich atoms:          {sm.sandwich_atoms}")
        print(f"  electrode atoms total:   {sm.electrode_atoms_total}")
        print(f"  middle atoms total:      {sm.middle_atoms_total}")
        print(f"  electrode matrix:\n{m.bottom_matrix}")
        print(f"  middle matrix:\n{m.top_matrix}")
        print(f"  electrode area mult:     {m.bottom_area_multiplier}")
        print(f"  middle area mult:        {m.top_area_multiplier}")
        print(f"  max strain:              {100 * m.max_strain:.3f}%")
        print(f"  rms strain:              {100 * m.rms_strain:.3f}%")
        print(f"  area strain:             {100 * m.area_strain:.3f}%")
        print()


def inplane_cell(atoms: Atoms) -> np.ndarray:
    return np.asarray(atoms.cell.array, dtype=float)[:2, :2]


def cell_summary(cell: np.ndarray) -> dict[str, float]:
    cell = np.asarray(cell, dtype=float)
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


def strain_summary(
    reference_cell: np.ndarray,
    target_cell: np.ndarray,
    reference: str,
    target: str,
) -> dict[str, Any]:
    """
    Rotation-invariant 2D strain required to deform reference_cell into target_cell.

    Positive principal strain means stretch.
    Negative principal strain means compression.
    """
    reference_cell = np.asarray(reference_cell, dtype=float)
    target_cell = np.asarray(target_cell, dtype=float)

    ref = cell_summary(reference_cell)
    tar = cell_summary(target_cell)

    F = np.linalg.solve(reference_cell, target_cell)
    singular_values = np.linalg.svd(F, compute_uv=False)
    principal = singular_values - 1.0

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
        "principal_strain_1_percent": 100.0 * float(principal[0]),
        "principal_strain_2_percent": 100.0 * float(principal[1]),
        "rms_principal_strain_percent": 100.0 * float(np.sqrt(np.mean(principal**2))),
        "max_abs_principal_strain_percent": 100.0 * float(np.max(np.abs(principal))),
        "deformation_gradient": F.tolist(),
    }


def sandwich_layer_strain_records(
    electrode: Atoms,
    middle: Atoms,
    sandwich_match: SandwichMatch | Match2D,
    electrode_name: str = "electrode",
    middle_name: str = "middle",
    shared_middle_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Return actual layer strains for strain='middle', 'electrodes', and 'shared'."""
    match = sandwich_match.match if isinstance(sandwich_match, SandwichMatch) else sandwich_match

    electrode_sc = make_2d_supercell(electrode, match.bottom_matrix)
    middle_sc = make_2d_supercell(middle, match.top_matrix)

    electrode_cell = inplane_cell(electrode_sc)
    middle_cell = inplane_cell(middle_sc)
    shared_cell = shared_inplane_cell(
        electrode_cell,
        middle_cell,
        top_weight=shared_middle_weight,
    )

    cases = [
        (
            f"{electrode_name}_{middle_name}_{electrode_name}_middle_strained",
            "middle",
            f"bottom_{electrode_name}",
            False,
            electrode_cell,
            electrode_cell,
            "electrode matched supercell",
            "electrode matched supercell",
        ),
        (
            f"{electrode_name}_{middle_name}_{electrode_name}_middle_strained",
            "middle",
            middle_name,
            True,
            middle_cell,
            electrode_cell,
            "middle matched supercell",
            "electrode matched supercell",
        ),
        (
            f"{electrode_name}_{middle_name}_{electrode_name}_middle_strained",
            "middle",
            f"top_{electrode_name}",
            False,
            electrode_cell,
            electrode_cell,
            "electrode matched supercell",
            "electrode matched supercell",
        ),
        (
            f"{electrode_name}_{middle_name}_{electrode_name}_electrodes_strained",
            "electrodes",
            f"bottom_{electrode_name}",
            True,
            electrode_cell,
            middle_cell,
            "electrode matched supercell",
            "middle matched supercell",
        ),
        (
            f"{electrode_name}_{middle_name}_{electrode_name}_electrodes_strained",
            "electrodes",
            middle_name,
            False,
            middle_cell,
            middle_cell,
            "middle matched supercell",
            "middle matched supercell",
        ),
        (
            f"{electrode_name}_{middle_name}_{electrode_name}_electrodes_strained",
            "electrodes",
            f"top_{electrode_name}",
            True,
            electrode_cell,
            middle_cell,
            "electrode matched supercell",
            "middle matched supercell",
        ),
        (
            f"{electrode_name}_{middle_name}_{electrode_name}_shared_strain",
            "shared",
            f"bottom_{electrode_name}",
            True,
            electrode_cell,
            shared_cell,
            "electrode matched supercell",
            "shared intermediate cell",
        ),
        (
            f"{electrode_name}_{middle_name}_{electrode_name}_shared_strain",
            "shared",
            middle_name,
            True,
            middle_cell,
            shared_cell,
            "middle matched supercell",
            "shared intermediate cell",
        ),
        (
            f"{electrode_name}_{middle_name}_{electrode_name}_shared_strain",
            "shared",
            f"top_{electrode_name}",
            True,
            electrode_cell,
            shared_cell,
            "electrode matched supercell",
            "shared intermediate cell",
        ),
    ]

    records = []
    for structure, mode, layer, strained, ref_cell, target_cell, ref_label, target_label in cases:
        rec = strain_summary(ref_cell, target_cell, ref_label, target_label)
        rec.update(
            {
                "structure": structure,
                "strain_mode": mode,
                "layer": layer,
                "strained": strained,
                "electrode_matrix": match.bottom_matrix.tolist(),
                "middle_matrix": match.top_matrix.tolist(),
                "electrode_area_multiplier": match.bottom_area_multiplier,
                "middle_area_multiplier": match.top_area_multiplier,
            }
        )
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
