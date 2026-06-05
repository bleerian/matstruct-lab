"""Unified sandwich matching, strain-split search, and stack construction.

This module is the preferred interface for:
- homogeneous electrodes: Au/MoS2/Au
- heterogeneous electrodes: Au/MoS2/Ag
- arbitrary strain partitioning with total-strain filtering
- named registry shifts for the middle layer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from ase import Atoms

from matstruct_lab.hetero_strain import inplane_cell, strain_summary
from matstruct_lab.heterostructures import set_inplane_cell, shift_z_min_to
from matstruct_lab.lattice_match import det_int, integer_2d_matrices, make_2d_supercell
from matstruct_lab.registry import apply_fractional_registry_shift, frac_shift_cart


@dataclass(frozen=True)
class SandwichSplit:
    """A common-cell strain split for a metal/layer/metal sandwich."""

    bottom_matrix: np.ndarray
    middle_matrix: np.ndarray
    top_matrix: np.ndarray
    bottom_weight: float
    middle_weight: float
    top_weight: float
    bottom_strain_percent: float
    middle_strain_percent: float
    top_strain_percent: float
    total_abs_strain_percent: float
    max_layer_abs_strain_percent: float
    total_atoms: int
    target_cell: np.ndarray
    homogeneous: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "bottom_matrix": self.bottom_matrix.tolist(),
            "middle_matrix": self.middle_matrix.tolist(),
            "top_matrix": self.top_matrix.tolist(),
            "bottom_weight": self.bottom_weight,
            "middle_weight": self.middle_weight,
            "top_weight": self.top_weight,
            "bottom_strain_percent": self.bottom_strain_percent,
            "middle_strain_percent": self.middle_strain_percent,
            "top_strain_percent": self.top_strain_percent,
            "total_abs_strain_percent": self.total_abs_strain_percent,
            "max_layer_abs_strain_percent": self.max_layer_abs_strain_percent,
            "total_atoms": self.total_atoms,
            "target_cell": self.target_cell.tolist(),
            "homogeneous": self.homogeneous,
            # Compatibility aliases for old notebooks.
            "electrode_matrix": self.bottom_matrix.tolist() if self.homogeneous else None,
            "electrode_strain_percent": self.bottom_strain_percent if self.homogeneous else None,
        }


HomogeneousSandwichSplit = SandwichSplit
HeterogeneousSandwichSplit = SandwichSplit


def weighted_inplane_cell(cells: Iterable[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    """Return a normalized weighted average of several 2D row-vector cells."""
    cells = [np.asarray(c, dtype=float)[:2, :2] for c in cells]
    weights = np.asarray(list(weights), dtype=float)

    if len(cells) != len(weights):
        raise ValueError("Number of cells and weights must match.")
    if np.any(weights < 0):
        raise ValueError("All weights must be non-negative.")
    if float(weights.sum()) == 0.0:
        raise ValueError("At least one weight must be positive.")

    weights = weights / weights.sum()
    out = np.zeros((2, 2), dtype=float)
    for cell, weight in zip(cells, weights):
        out += float(weight) * cell
    return out


def split_target_cell(electrode_cell: np.ndarray, middle_cell: np.ndarray, middle_weight: float) -> np.ndarray:
    """Two-body target cell for homogeneous electrode/middle/electrode stacks."""
    w = float(middle_weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError("middle_weight must be between 0 and 1.")
    return weighted_inplane_cell([electrode_cell, middle_cell], [1.0 - w, w])


def _weight_grid_2(step: float) -> list[float]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("weight_step must be in (0, 1].")
    return [float(np.clip(x, 0.0, 1.0)) for x in np.arange(0.0, 1.0 + 0.5 * step, step)]


def _weight_grid_3(step: float) -> list[tuple[float, float, float]]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("weight_step must be in (0, 1].")

    vals = np.arange(0.0, 1.0 + 0.5 * step, step)
    out: list[tuple[float, float, float]] = []
    for wb in vals:
        for wm in vals:
            wt = 1.0 - wb - wm
            if wt < -1e-10:
                continue
            out.append((float(wb), float(wm), float(max(wt, 0.0))))
    return out


def _matrix_cache(atoms: Atoms, max_entry: int, max_area: int) -> list[tuple[np.ndarray, int, np.ndarray, int]]:
    base = inplane_cell(atoms)
    out: list[tuple[np.ndarray, int, np.ndarray, int]] = []
    for matrix in integer_2d_matrices(max_entry=max_entry, max_det=max_area):
        det = det_int(matrix)
        if det <= 0:
            continue
        out.append((matrix, det, matrix @ base, len(atoms) * det))
    return out


def _strain_abs_percent(ref_cell: np.ndarray, target_cell: np.ndarray) -> float:
    return abs(float(strain_summary(ref_cell, target_cell)["max_abs_principal_strain_percent"]))


def find_homogeneous_sandwich_splits(
    electrode: Atoms,
    middle: Atoms,
    max_entry: int = 4,
    max_area: int = 30,
    max_atoms: int = 250,
    weight_step: float = 0.01,
    max_total_abs_strain_percent: float = 6.0,
    max_layer_abs_strain_percent: float | None = None,
    limit: int = 100,
) -> list[SandwichSplit]:
    """Search metal/layer/same-metal strain splits.

    The reported electrode strain is per electrode, not doubled for top+bottom.
    """
    electrode_cache = _matrix_cache(electrode, max_entry=max_entry, max_area=max_area)
    middle_cache = _matrix_cache(middle, max_entry=max_entry, max_area=max_area)
    weights = _weight_grid_2(weight_step)

    found: list[SandwichSplit] = []

    for Me, _det_e, ecell, e_atoms_one_side in electrode_cache:
        for Mm, _det_m, mcell, m_atoms in middle_cache:
            total_atoms = 2 * e_atoms_one_side + m_atoms
            if total_atoms > max_atoms:
                continue

            for w in weights:
                target = split_target_cell(ecell, mcell, middle_weight=w)
                se = _strain_abs_percent(ecell, target)
                sm = _strain_abs_percent(mcell, target)
                total_s = se + sm
                max_s = max(se, sm)

                if total_s > max_total_abs_strain_percent:
                    continue
                if max_layer_abs_strain_percent is not None and max_s > max_layer_abs_strain_percent:
                    continue

                found.append(
                    SandwichSplit(
                        bottom_matrix=Me.copy(),
                        middle_matrix=Mm.copy(),
                        top_matrix=Me.copy(),
                        bottom_weight=1.0 - w,
                        middle_weight=w,
                        top_weight=1.0 - w,
                        bottom_strain_percent=se,
                        middle_strain_percent=sm,
                        top_strain_percent=se,
                        total_abs_strain_percent=total_s,
                        max_layer_abs_strain_percent=max_s,
                        total_atoms=total_atoms,
                        target_cell=target.copy(),
                        homogeneous=True,
                    )
                )

    found.sort(key=lambda x: (x.total_abs_strain_percent, x.max_layer_abs_strain_percent, x.total_atoms))
    return found[:limit]


def find_heterogeneous_sandwich_splits(
    bottom_electrode: Atoms,
    middle: Atoms,
    top_electrode: Atoms,
    max_entry: int = 3,
    max_area: int = 20,
    max_atoms: int = 250,
    weight_step: float = 0.05,
    max_total_abs_strain_percent: float = 6.0,
    max_layer_abs_strain_percent: float | None = None,
    limit: int = 100,
) -> list[SandwichSplit]:
    """Search bottom-electrode/layer/top-electrode strain splits."""
    bottom_cache = _matrix_cache(bottom_electrode, max_entry=max_entry, max_area=max_area)
    middle_cache = _matrix_cache(middle, max_entry=max_entry, max_area=max_area)
    top_cache = _matrix_cache(top_electrode, max_entry=max_entry, max_area=max_area)
    weights = _weight_grid_3(weight_step)

    found: list[SandwichSplit] = []

    for Mb, _det_b, bcell, b_atoms in bottom_cache:
        for Mm, _det_m, mcell, m_atoms in middle_cache:
            for Mt, _det_t, tcell, t_atoms in top_cache:
                total_atoms = b_atoms + m_atoms + t_atoms
                if total_atoms > max_atoms:
                    continue

                for wb, wm, wt in weights:
                    target = weighted_inplane_cell([bcell, mcell, tcell], [wb, wm, wt])
                    sb = _strain_abs_percent(bcell, target)
                    sm = _strain_abs_percent(mcell, target)
                    st = _strain_abs_percent(tcell, target)
                    total_s = sb + sm + st
                    max_s = max(sb, sm, st)

                    if total_s > max_total_abs_strain_percent:
                        continue
                    if max_layer_abs_strain_percent is not None and max_s > max_layer_abs_strain_percent:
                        continue

                    found.append(
                        SandwichSplit(
                            bottom_matrix=Mb.copy(),
                            middle_matrix=Mm.copy(),
                            top_matrix=Mt.copy(),
                            bottom_weight=wb,
                            middle_weight=wm,
                            top_weight=wt,
                            bottom_strain_percent=sb,
                            middle_strain_percent=sm,
                            top_strain_percent=st,
                            total_abs_strain_percent=total_s,
                            max_layer_abs_strain_percent=max_s,
                            total_atoms=total_atoms,
                            target_cell=target.copy(),
                            homogeneous=False,
                        )
                    )

    found.sort(key=lambda x: (x.total_abs_strain_percent, x.max_layer_abs_strain_percent, x.total_atoms))
    return found[:limit]


def build_sandwich(
    bottom_electrode: Atoms,
    middle: Atoms,
    top_electrode: Atoms,
    bottom_matrix: np.ndarray,
    middle_matrix: np.ndarray,
    top_matrix: np.ndarray,
    target_inplane_cell: np.ndarray | None = None,
    strain_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    electrode_middle_distance: float = 3.3,
    vacuum: float = 20.0,
    middle_registry_material: str | None = None,
    middle_registry: str | None = None,
    middle_registry_anchor_frac: tuple[float, float] = (0.0, 0.0),
    middle_extra_shift_frac: tuple[float, float] = (0.0, 0.0),
    top_electrode_shift_frac: tuple[float, float] = (0.0, 0.0),
    pbc_z: bool = False,
) -> Atoms:
    """Build a bottom/middle/top sandwich. Top and bottom may be identical or different."""
    bottom = make_2d_supercell(bottom_electrode, bottom_matrix)
    mid = make_2d_supercell(middle, middle_matrix)
    top = make_2d_supercell(top_electrode, top_matrix)

    bcell = inplane_cell(bottom)
    mcell = inplane_cell(mid)
    tcell = inplane_cell(top)

    if target_inplane_cell is None:
        target = weighted_inplane_cell([bcell, mcell, tcell], strain_weights)
    else:
        target = np.asarray(target_inplane_cell, dtype=float)[:2, :2]

    bottom = set_inplane_cell(bottom, target, scale_atoms=True)
    mid = set_inplane_cell(mid, target, scale_atoms=True)
    top = set_inplane_cell(top, target, scale_atoms=True)

    if middle_registry_material is not None and middle_registry is not None:
        mid = apply_fractional_registry_shift(
            mid,
            material=middle_registry_material,
            registry=middle_registry,
            anchor_frac=middle_registry_anchor_frac,
            extra_shift_frac=middle_extra_shift_frac,
        )

    d = float(electrode_middle_distance)
    bottom = shift_z_min_to(bottom, vacuum / 2.0)
    mid = shift_z_min_to(mid, float(bottom.positions[:, 2].max()) + d)
    top = shift_z_min_to(top, float(mid.positions[:, 2].max()) + d)

    if top_electrode_shift_frac != (0.0, 0.0):
        top.translate(frac_shift_cart(target, top_electrode_shift_frac))

    stack = bottom + mid + top
    z_min = float(stack.positions[:, 2].min())
    z_max = float(stack.positions[:, 2].max())
    final_z = (z_max - z_min) + float(vacuum)

    final_cell = np.zeros((3, 3), dtype=float)
    final_cell[:2, :2] = target
    final_cell[2, 2] = final_z

    stack.set_cell(final_cell, scale_atoms=False)
    stack.set_pbc([True, True, bool(pbc_z)])
    stack.wrap(eps=1e-8)
    return stack


def build_sandwich_from_split(
    bottom_electrode: Atoms,
    middle: Atoms,
    top_electrode: Atoms,
    split: SandwichSplit | dict[str, Any],
    **kwargs: Any,
) -> Atoms:
    """Build a sandwich from a split returned by a find_*_sandwich_splits function."""
    dct = split.as_dict() if hasattr(split, "as_dict") else dict(split)
    return build_sandwich(
        bottom_electrode=bottom_electrode,
        middle=middle,
        top_electrode=top_electrode,
        bottom_matrix=np.array(dct["bottom_matrix"], dtype=int),
        middle_matrix=np.array(dct["middle_matrix"], dtype=int),
        top_matrix=np.array(dct["top_matrix"], dtype=int),
        target_inplane_cell=np.array(dct["target_cell"], dtype=float),
        **kwargs,
    )


def build_homogeneous_sandwich_from_split(
    electrode: Atoms,
    middle: Atoms,
    split: SandwichSplit | dict[str, Any],
    **kwargs: Any,
) -> Atoms:
    """Build electrode/middle/electrode from a homogeneous split."""
    return build_sandwich_from_split(electrode, middle, electrode, split, **kwargs)


def split_records(splits: Iterable[SandwichSplit]) -> list[dict[str, Any]]:
    return [s.as_dict() for s in splits]


homogeneous_split_records = split_records
heterogeneous_split_records = split_records
build_homogeneous_electrode_sandwich_from_split = build_homogeneous_sandwich_from_split
build_heterogeneous_electrode_sandwich_from_split = build_sandwich_from_split
