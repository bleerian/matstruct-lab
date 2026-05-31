#!/usr/bin/env python3
"""Utilities for 2D lattice mismatch and heterostructure strain analysis.

Conventions
-----------
- pymatgen lattice vectors a,b,c are converted to a 2x2 Cartesian matrix whose
  columns are the in-plane a and b vectors projected onto x-y.
- A strain maps `reference_cell -> target_cell`.
- Positive strain means the reference layer must be stretched to match target.
- For 2D heterostructures, use the polar stretch strain by default because it
  removes rigid in-plane rotation from the deformation gradient.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pymatgen.core import Structure


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
    """Return 2x2 matrix with columns equal to projected in-plane a,b vectors."""
    mat = np.array(structure.lattice.matrix, dtype=float)  # rows: a, b, c
    return np.column_stack([mat[0, :2], mat[1, :2]])


def transform_2d_cell(cell: np.ndarray, transform: Iterable[Iterable[float]]) -> np.ndarray:
    """Apply a pymatgen-style 2D supercell transform to a 2D column cell.

    For pymatgen, new lattice rows = T @ old lattice rows. With column cell
    convention, new_cell = old_cell @ T.T.
    """
    t = np.array(transform, dtype=float)
    if t.shape == (3, 3):
        t = t[:2, :2]
    if t.shape != (2, 2):
        raise ValueError(f"Expected 2x2 or 3x3 transform, got {t.shape}")
    return cell @ t.T


def cell_summary(cell: np.ndarray) -> CellSummary:
    a = cell[:, 0]
    b = cell[:, 1]
    la = float(np.linalg.norm(a))
    lb = float(np.linalg.norm(b))
    cosang = float(np.dot(a, b) / (la * lb))
    cosang = max(-1.0, min(1.0, cosang))
    gamma = float(np.degrees(np.arccos(cosang)))
    area = float(abs(np.linalg.det(cell)))
    return CellSummary(a_A=la, b_A=lb, gamma_deg=gamma, area_A2=area)


def polar_decomposition_2d(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R, U from F = R U, where U is symmetric positive stretch."""
    C = F.T @ F
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 0.0, None)
    U = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    R = F @ np.linalg.inv(U)
    return R, U


def strain_summary(reference_cell: np.ndarray, target_cell: np.ndarray,
                   reference_name: str = "reference", target_name: str = "target") -> StrainSummary:
    """Compute rotation-free 2D strain required to map reference_cell onto target_cell."""
    ref = cell_summary(reference_cell)
    tar = cell_summary(target_cell)

    F = target_cell @ np.linalg.inv(reference_cell)
    R, U = polar_decomposition_2d(F)
    stretch_strain = U - np.eye(2)
    principal = np.linalg.eigvalsh(stretch_strain)

    eps_a = (tar.a_A - ref.a_A) / ref.a_A
    eps_b = (tar.b_A - ref.b_A) / ref.b_A
    area_strain = (tar.area_A2 - ref.area_A2) / ref.area_A2

    return StrainSummary(
        reference=reference_name,
        target=target_name,
        a_ref_A=ref.a_A,
        b_ref_A=ref.b_A,
        gamma_ref_deg=ref.gamma_deg,
        area_ref_A2=ref.area_A2,
        a_target_A=tar.a_A,
        b_target_A=tar.b_A,
        gamma_target_deg=tar.gamma_deg,
        area_target_A2=tar.area_A2,
        eps_a_percent=100 * eps_a,
        eps_b_percent=100 * eps_b,
        delta_gamma_deg=tar.gamma_deg - ref.gamma_deg,
        area_strain_percent=100 * area_strain,
        stretch_xx_percent=100 * stretch_strain[0, 0],
        stretch_yy_percent=100 * stretch_strain[1, 1],
        stretch_xy_percent=100 * stretch_strain[0, 1],
        principal_strain_1_percent=100 * principal[0],
        principal_strain_2_percent=100 * principal[1],
        rms_principal_strain_percent=100 * float(np.sqrt(np.mean(principal**2))),
        max_abs_principal_strain_percent=100 * float(np.max(np.abs(principal))),
        deformation_gradient=F.tolist(),
        rotation_matrix=R.tolist(),
        stretch_tensor=U.tolist(),
    )


def to_plain_dict(obj: Any) -> dict[str, Any]:
    d = asdict(obj)
    # keep JSON/CSV clean
    return d


def write_json(path: str | Path, records: list[dict[str, Any]]) -> None:
    Path(path).write_text(json.dumps(records, indent=2), encoding="utf-8")


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
