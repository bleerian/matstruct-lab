"""Compatibility wrappers for sandwich matching.

Prefer importing from :mod:`matstruct_lab.sandwich` in new code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ase import Atoms

from matstruct_lab.lattice_match import Match2D, find_2d_matches
from matstruct_lab.sandwich import (
    SandwichSplit,
    build_homogeneous_sandwich_from_split,
    build_sandwich,
    build_sandwich_from_split,
    find_heterogeneous_sandwich_splits,
    find_homogeneous_sandwich_splits,
    split_records,
)
from matstruct_lab.hetero_strain import write_csv_records, write_json_records


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
        d.update({
            "match_index": self.match_index,
            "sandwich_atoms": self.sandwich_atoms,
            "electrode_atoms_total": self.electrode_atoms_total,
            "middle_atoms_total": self.middle_atoms_total,
            "electrode_matrix": self.match.bottom_matrix.tolist(),
            "middle_matrix": self.match.top_matrix.tolist(),
            "electrode_area_multiplier": self.match.bottom_area_multiplier,
            "middle_area_multiplier": self.match.top_area_multiplier,
        })
        return d


def sandwich_atom_count(electrode: Atoms, middle: Atoms, match: Match2D) -> int:
    return 2 * len(electrode) * match.bottom_area_multiplier + len(middle) * match.top_area_multiplier


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
    """Old two-body sandwich match search kept for notebook compatibility."""
    raw_matches = find_2d_matches(
        bottom=electrode,
        top=middle,
        max_entry=max_entry,
        max_area=max_area,
        max_strain=max_strain,
        max_atoms=max_atoms,
        limit=search_limit,
    )

    matches: list[SandwichMatch] = []
    for i, match in enumerate(raw_matches):
        electrode_atoms_total = 2 * len(electrode) * match.bottom_area_multiplier
        middle_atoms_total = len(middle) * match.top_area_multiplier
        total = electrode_atoms_total + middle_atoms_total
        if total <= max_atoms:
            matches.append(
                SandwichMatch(
                    match=match,
                    match_index=i,
                    sandwich_atoms=total,
                    electrode_atoms_total=electrode_atoms_total,
                    middle_atoms_total=middle_atoms_total,
                )
            )

    if sort_by == "atoms_then_strain":
        key = lambda sm: (sm.sandwich_atoms, sm.match.max_strain, sm.match.rms_strain, sm.match.area_strain)
    elif sort_by == "strain_then_atoms":
        key = lambda sm: (sm.match.max_strain, sm.match.rms_strain, sm.sandwich_atoms, sm.match.area_strain)
    else:
        raise ValueError("sort_by must be 'atoms_then_strain' or 'strain_then_atoms'")

    matches.sort(key=key)
    return matches[:limit]


def print_sandwich_matches(matches: list[SandwichMatch]) -> None:
    for i, sm in enumerate(matches):
        m = sm.match
        print(f"[{i}] original match index: {sm.match_index}")
        print(f"  sandwich atoms:          {sm.sandwich_atoms}")
        print(f"  electrode atoms total:   {sm.electrode_atoms_total}")
        print(f"  middle atoms total:      {sm.middle_atoms_total}")
        print(f"  electrode matrix:\n{m.bottom_matrix}")
        print(f"  middle matrix:\n{m.top_matrix}")
        print(f"  max strain:              {100 * m.max_strain:.3f}%")
        print(f"  rms strain:              {100 * m.rms_strain:.3f}%")
        print(f"  area strain:             {100 * m.area_strain:.3f}%")
        print()


__all__ = [
    "SandwichMatch",
    "SandwichSplit",
    "sandwich_atom_count",
    "find_2d_sandwich_matches",
    "print_sandwich_matches",
    "find_homogeneous_sandwich_splits",
    "find_heterogeneous_sandwich_splits",
    "build_sandwich",
    "build_sandwich_from_split",
    "build_homogeneous_sandwich_from_split",
    "split_records",
    "write_csv_records",
    "write_json_records",
]
