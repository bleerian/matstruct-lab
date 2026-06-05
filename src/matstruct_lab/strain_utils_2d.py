"""Compatibility wrappers for older pymatgen-based strain scripts.

New ASE-native code should import from :mod:`matstruct_lab.hetero_strain`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pymatgen.core import Structure

from matstruct_lab.hetero_strain import (
    cell_summary as _cell_summary_dict,
    polar_decomposition_2d,
    strain_summary as _strain_summary_dict,
    write_json_records,
)


@dataclass
class CellSummary:
    a_A: float
    b_A: float
    gamma_deg: float
    area_A2: float


@dataclass
class StrainSummary:
    reference: str
    target: str
    a_ref_A: float
    b_ref_A: float
    gamma_ref_deg: float
    area_ref_A2: float
    a_target_A: float
    b_target_A: float
    gamma_target_deg: float
    area_target_A2: float
    eps_a_percent: float
    eps_b_percent: float
    delta_gamma_deg: float
    area_strain_percent: float
    stretch_xx_percent: float
    stretch_yy_percent: float
    stretch_xy_percent: float
    principal_strain_1_percent: float
    principal_strain_2_percent: float
    rms_principal_strain_percent: float
    max_abs_principal_strain_percent: float
    deformation_gradient: list[list[float]]
    rotation_matrix: list[list[float]]
    stretch_tensor: list[list[float]]


def load_structure(path: str | Path) -> Structure:
    return Structure.from_file(str(path))


def cell2d_from_structure(structure: Structure) -> np.ndarray:
    """Return ASE-style row-vector 2D cell from a pymatgen Structure."""
    mat = np.array(structure.lattice.matrix, dtype=float)
    return mat[:2, :2]


def transform_2d_cell(cell: np.ndarray, transform: Iterable[Iterable[float]]) -> np.ndarray:
    """Apply a 2D supercell transform using ASE row-vector convention."""
    t = np.array(transform, dtype=float)
    if t.shape == (3, 3):
        t = t[:2, :2]
    if t.shape != (2, 2):
        raise ValueError(f"Expected 2x2 or 3x3 transform, got {t.shape}")
    return t @ np.asarray(cell, dtype=float)


def cell_summary(cell: np.ndarray) -> CellSummary:
    d = _cell_summary_dict(cell)
    return CellSummary(**d)


def strain_summary(
    reference_cell: np.ndarray,
    target_cell: np.ndarray,
    reference_name: str = "reference",
    target_name: str = "target",
) -> StrainSummary:
    d = _strain_summary_dict(reference_cell, target_cell, reference=reference_name, target=target_name)
    return StrainSummary(
        reference=d["reference"],
        target=d["target"],
        a_ref_A=d["a_ref_A"],
        b_ref_A=d["b_ref_A"],
        gamma_ref_deg=d["gamma_ref_deg"],
        area_ref_A2=d["area_ref_A2"],
        a_target_A=d["a_target_A"],
        b_target_A=d["b_target_A"],
        gamma_target_deg=d["gamma_target_deg"],
        area_target_A2=d["area_target_A2"],
        eps_a_percent=d["eps_a_percent"],
        eps_b_percent=d["eps_b_percent"],
        delta_gamma_deg=d["delta_gamma_deg"],
        area_strain_percent=d["signed_area_strain_percent"],
        stretch_xx_percent=d["stretch_xx_percent"],
        stretch_yy_percent=d["stretch_yy_percent"],
        stretch_xy_percent=d["stretch_xy_percent"],
        principal_strain_1_percent=d["principal_strain_1_percent"],
        principal_strain_2_percent=d["principal_strain_2_percent"],
        rms_principal_strain_percent=d["rms_principal_strain_percent"],
        max_abs_principal_strain_percent=d["max_abs_principal_strain_percent"],
        deformation_gradient=d["deformation_gradient"],
        rotation_matrix=d["rotation_matrix"],
        stretch_tensor=d["stretch_tensor"],
    )


def to_plain_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return dict(obj)


def write_json(path: str | Path, records: list[dict[str, Any]]) -> None:
    write_json_records(path, records)


def parse_matrix(text: str) -> np.ndarray:
    """Parse 'a b c d' or '[[a,b],[c,d]]' as a 2x2 matrix."""
    s = text.strip()
    if s.startswith("["):
        arr = np.array(json.loads(s), dtype=float)
    else:
        arr = np.array([float(x) for x in s.replace(",", " ").split()], dtype=float)
    if arr.size != 4:
        raise ValueError(f"Expected 4 numbers for a 2x2 matrix, got {arr.size}: {text}")
    return arr.reshape(2, 2)
