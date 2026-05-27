from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.build import make_supercell


Array2D = np.ndarray


@dataclass(frozen=True)
class Match2D:
    """A candidate 2D lattice match between two structures."""

    bottom_matrix: np.ndarray
    top_matrix: np.ndarray
    bottom_area_multiplier: int
    top_area_multiplier: int
    max_strain: float
    rms_strain: float
    area_strain: float
    total_atoms: int

    def as_dict(self) -> dict:
        return {
            "bottom_matrix": self.bottom_matrix.tolist(),
            "top_matrix": self.top_matrix.tolist(),
            "bottom_area_multiplier": self.bottom_area_multiplier,
            "top_area_multiplier": self.top_area_multiplier,
            "max_strain_percent": 100 * self.max_strain,
            "rms_strain_percent": 100 * self.rms_strain,
            "area_strain_percent": 100 * self.area_strain,
            "total_atoms": self.total_atoms,
        }


def inplane_cell(atoms: Atoms) -> np.ndarray:
    """Return the 2x2 in-plane cell matrix using the x/y components.

    This assumes the first two lattice vectors lie in the xy plane, which is
    the standard representation for most 2D slabs and monolayers.
    """
    cell = np.asarray(atoms.cell.array, dtype=float)
    return cell[:2, :2]


def det_int(matrix: np.ndarray) -> int:
    """Integer determinant for a 2x2 integer matrix."""
    return int(round(np.linalg.det(np.asarray(matrix, dtype=int))))


def integer_2d_matrices(
    max_entry: int = 4,
    min_det: int = 1,
    max_det: int = 20,
) -> list[np.ndarray]:
    """Generate positive-determinant 2x2 integer supercell matrices."""
    matrices: list[np.ndarray] = []

    for a in range(-max_entry, max_entry + 1):
        for b in range(-max_entry, max_entry + 1):
            for c in range(-max_entry, max_entry + 1):
                for d in range(-max_entry, max_entry + 1):
                    M = np.array([[a, b], [c, d]], dtype=int)
                    det = det_int(M)

                    if min_det <= det <= max_det:
                        matrices.append(M)

    # Remove exact duplicates while preserving order.
    seen = set()
    unique = []
    for M in matrices:
        key = tuple(M.ravel())
        if key not in seen:
            seen.add(key)
            unique.append(M)

    return unique


def make_2d_supercell(atoms: Atoms, matrix_2d: np.ndarray) -> Atoms:
    """Apply a 2D integer supercell matrix to an ASE Atoms object."""
    P = np.eye(3, dtype=int)
    P[:2, :2] = np.asarray(matrix_2d, dtype=int)
    return make_supercell(atoms, P)


def deformation_strain(source_cell: np.ndarray, target_cell: np.ndarray) -> tuple[float, float, float]:
    """Return max, RMS, and area strain needed to deform source_cell into target_cell.

    The comparison is rotation-invariant because it uses the singular values
    of the deformation gradient.
    """
    source_cell = np.asarray(source_cell, dtype=float)
    target_cell = np.asarray(target_cell, dtype=float)

    F = np.linalg.solve(source_cell, target_cell)
    singular_values = np.linalg.svd(F, compute_uv=False)

    principal_strains = singular_values - 1.0
    max_strain = float(np.max(np.abs(principal_strains)))
    rms_strain = float(np.sqrt(np.mean(principal_strains**2)))
    area_strain = float(abs(np.linalg.det(F) - 1.0))

    return max_strain, rms_strain, area_strain


def find_2d_matches(
    bottom: Atoms,
    top: Atoms,
    max_entry: int = 4,
    max_area: int = 20,
    max_strain: float = 0.03,
    max_atoms: int = 500,
    limit: int = 20,
) -> list[Match2D]:
    """Find low-strain 2D supercell matches.

    Parameters
    ----------
    bottom
        Substrate/electrode/reference layer.
    top
        Film/second layer to be matched to the bottom layer.
    max_entry
        Largest absolute integer allowed in the 2x2 supercell matrices.
    max_area
        Maximum determinant of each 2D supercell matrix.
    max_strain
        Maximum allowed principal strain as a fraction. Example: 0.03 = 3%.
    max_atoms
        Maximum total atoms in the combined candidate.
    limit
        Number of best matches to return.
    """
    bottom_cell = inplane_cell(bottom)
    top_cell = inplane_cell(top)

    matrices = integer_2d_matrices(max_entry=max_entry, max_det=max_area)

    matches: list[Match2D] = []

    for Mb in matrices:
        det_b = det_int(Mb)
        if det_b <= 0:
            continue

        bottom_atoms = len(bottom) * det_b
        bottom_supercell = Mb @ bottom_cell

        for Mt in matrices:
            det_t = det_int(Mt)
            if det_t <= 0:
                continue

            total_atoms = bottom_atoms + len(top) * det_t
            if total_atoms > max_atoms:
                continue

            top_supercell = Mt @ top_cell
            max_s, rms_s, area_s = deformation_strain(top_supercell, bottom_supercell)

            if max_s <= max_strain:
                matches.append(
                    Match2D(
                        bottom_matrix=Mb.copy(),
                        top_matrix=Mt.copy(),
                        bottom_area_multiplier=det_b,
                        top_area_multiplier=det_t,
                        max_strain=max_s,
                        rms_strain=rms_s,
                        area_strain=area_s,
                        total_atoms=total_atoms,
                    )
                )

    matches.sort(
        key=lambda m: (
            m.max_strain,
            m.rms_strain,
            m.total_atoms,
            m.bottom_area_multiplier + m.top_area_multiplier,
        )
    )

    return matches[:limit]


def print_matches(matches: Iterable[Match2D]) -> None:
    """Print candidate matches in a readable compact format."""
    for i, match in enumerate(matches):
        print(f"[{i}]")
        print(f"  bottom matrix:\n{match.bottom_matrix}")
        print(f"  top matrix:\n{match.top_matrix}")
        print(f"  bottom area multiplier: {match.bottom_area_multiplier}")
        print(f"  top area multiplier:    {match.top_area_multiplier}")
        print(f"  max strain:             {100 * match.max_strain:.3f}%")
        print(f"  rms strain:             {100 * match.rms_strain:.3f}%")
        print(f"  area strain:            {100 * match.area_strain:.3f}%")
        print(f"  total atoms:            {match.total_atoms}")
        print()


SQRT3_HEX_MATRIX = np.array([[1, 1], [-1, 2]], dtype=int)
"""One common integer matrix that produces a sqrt(3) x sqrt(3) hexagonal cell."""
