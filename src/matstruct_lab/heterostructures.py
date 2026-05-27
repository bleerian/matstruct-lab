from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms

from matstruct_lab.lattice_match import Match2D, make_2d_supercell


StrainMode = Literal["top", "bottom", "shared"]
SandwichStrainMode = Literal["middle", "electrodes", "shared"]


def set_inplane_cell(atoms: Atoms, inplane_cell: np.ndarray, scale_atoms: bool = True) -> Atoms:
    atoms = atoms.copy()

    new_cell = atoms.cell.array.copy()
    new_cell[0, :2] = inplane_cell[0]
    new_cell[1, :2] = inplane_cell[1]
    new_cell[0, 2] = 0.0
    new_cell[1, 2] = 0.0

    atoms.set_cell(new_cell, scale_atoms=scale_atoms)
    return atoms


def shift_z_min_to(atoms: Atoms, z_min: float) -> Atoms:
    atoms = atoms.copy()
    positions = atoms.positions.copy()
    positions[:, 2] += z_min - positions[:, 2].min()
    atoms.positions = positions
    return atoms


def shared_inplane_cell(
    bottom_cell: np.ndarray,
    top_cell: np.ndarray,
    top_weight: float = 0.5,
) -> np.ndarray:
    if not 0.0 <= top_weight <= 1.0:
        raise ValueError("top_weight must be between 0.0 and 1.0")

    return (1.0 - top_weight) * bottom_cell + top_weight * top_cell


def build_heterostructure(
    bottom: Atoms,
    top: Atoms,
    match: Match2D,
    interlayer_distance: float = 3.3,
    vacuum: float = 20.0,
    strain: StrainMode = "top",
    shared_top_weight: float = 0.5,
    lateral_shift: tuple[float, float] = (0.0, 0.0),
) -> Atoms:
    bottom_sc = make_2d_supercell(bottom, match.bottom_matrix)
    top_sc = make_2d_supercell(top, match.top_matrix)

    bottom_cell = bottom_sc.cell.array.copy()
    top_cell = top_sc.cell.array.copy()

    bottom_inplane = bottom_cell[:2, :2]
    top_inplane = top_cell[:2, :2]

    if strain == "top":
        target_inplane = bottom_inplane
        top_sc = set_inplane_cell(top_sc, target_inplane, scale_atoms=True)

    elif strain == "bottom":
        target_inplane = top_inplane
        bottom_sc = set_inplane_cell(bottom_sc, target_inplane, scale_atoms=True)

    elif strain == "shared":
        target_inplane = shared_inplane_cell(
            bottom_inplane,
            top_inplane,
            top_weight=shared_top_weight,
        )
        bottom_sc = set_inplane_cell(bottom_sc, target_inplane, scale_atoms=True)
        top_sc = set_inplane_cell(top_sc, target_inplane, scale_atoms=True)

    else:
        raise ValueError("strain must be 'top', 'bottom', or 'shared'")

    bottom_sc = shift_z_min_to(bottom_sc, vacuum / 2.0)

    top_sc = shift_z_min_to(
        top_sc,
        bottom_sc.positions[:, 2].max() + interlayer_distance,
    )

    top_positions = top_sc.positions.copy()
    top_positions[:, 0] += lateral_shift[0]
    top_positions[:, 1] += lateral_shift[1]
    top_sc.positions = top_positions

    hetero = bottom_sc + top_sc

    z_min = hetero.positions[:, 2].min()
    z_max = hetero.positions[:, 2].max()
    final_z = (z_max - z_min) + vacuum

    final_cell = np.zeros((3, 3), dtype=float)
    final_cell[:2, :2] = target_inplane
    final_cell[2, 2] = final_z

    hetero.set_cell(final_cell, scale_atoms=False)
    hetero.set_pbc([True, True, False])
    hetero.wrap(eps=1e-8)

    return hetero


def build_electrode_sandwich(
    electrode: Atoms,
    middle: Atoms,
    match: Match2D,
    electrode_middle_distance: float = 3.3,
    vacuum: float = 20.0,
    strain: SandwichStrainMode = "middle",
    shared_middle_weight: float = 0.5,
    middle_shift: tuple[float, float] = (0.0, 0.0),
    top_electrode_shift: tuple[float, float] = (0.0, 0.0),
) -> Atoms:
    bottom_electrode = make_2d_supercell(electrode, match.bottom_matrix)
    middle_sc = make_2d_supercell(middle, match.top_matrix)
    top_electrode = make_2d_supercell(electrode, match.bottom_matrix)

    electrode_cell = bottom_electrode.cell.array.copy()
    middle_cell = middle_sc.cell.array.copy()

    electrode_inplane = electrode_cell[:2, :2]
    middle_inplane = middle_cell[:2, :2]

    if strain == "middle":
        target_inplane = electrode_inplane
        middle_sc = set_inplane_cell(middle_sc, target_inplane, scale_atoms=True)

    elif strain == "electrodes":
        target_inplane = middle_inplane
        bottom_electrode = set_inplane_cell(bottom_electrode, target_inplane, scale_atoms=True)
        top_electrode = set_inplane_cell(top_electrode, target_inplane, scale_atoms=True)

    elif strain == "shared":
        target_inplane = shared_inplane_cell(
            electrode_inplane,
            middle_inplane,
            top_weight=shared_middle_weight,
        )
        bottom_electrode = set_inplane_cell(bottom_electrode, target_inplane, scale_atoms=True)
        middle_sc = set_inplane_cell(middle_sc, target_inplane, scale_atoms=True)
        top_electrode = set_inplane_cell(top_electrode, target_inplane, scale_atoms=True)

    else:
        raise ValueError("strain must be 'middle', 'electrodes', or 'shared'")

    bottom_electrode = shift_z_min_to(bottom_electrode, vacuum / 2.0)

    middle_sc = shift_z_min_to(
        middle_sc,
        bottom_electrode.positions[:, 2].max() + electrode_middle_distance,
    )

    middle_positions = middle_sc.positions.copy()
    middle_positions[:, 0] += middle_shift[0]
    middle_positions[:, 1] += middle_shift[1]
    middle_sc.positions = middle_positions

    top_electrode = shift_z_min_to(
        top_electrode,
        middle_sc.positions[:, 2].max() + electrode_middle_distance,
    )

    top_positions = top_electrode.positions.copy()
    top_positions[:, 0] += top_electrode_shift[0]
    top_positions[:, 1] += top_electrode_shift[1]
    top_electrode.positions = top_positions

    stack = bottom_electrode + middle_sc + top_electrode

    z_min = stack.positions[:, 2].min()
    z_max = stack.positions[:, 2].max()
    final_z = (z_max - z_min) + vacuum

    final_cell = np.zeros((3, 3), dtype=float)
    final_cell[:2, :2] = target_inplane
    final_cell[2, 2] = final_z

    stack.set_cell(final_cell, scale_atoms=False)
    stack.set_pbc([True, True, False])
    stack.wrap(eps=1e-8)

    return stack
