from __future__ import annotations

import numpy as np
from ase import Atoms

from matstruct_lab.lattice_match import Match2D, make_2d_supercell, inplane_cell


def cell_lengths_angle_2d(cell_2d: np.ndarray) -> tuple[float, float, float]:
    """Return a, b, and gamma angle in degrees for a 2D cell."""
    cell_2d = np.asarray(cell_2d, dtype=float)

    a_vec = cell_2d[0]
    b_vec = cell_2d[1]

    a = float(np.linalg.norm(a_vec))
    b = float(np.linalg.norm(b_vec))

    cos_gamma = np.dot(a_vec, b_vec) / (a * b)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma = float(np.degrees(np.arccos(cos_gamma)))

    return a, b, gamma


def strain_2d(reference_cell: np.ndarray, final_cell: np.ndarray) -> dict:
    """Calculate 2D strain needed to deform reference_cell into final_cell.

    Parameters
    ----------
    reference_cell
        Original unstrained 2x2 in-plane cell.
    final_cell
        Final strained 2x2 in-plane cell.

    Returns
    -------
    dict
        Principal strains, RMS strain, area strain, and changes in a, b, gamma.
    """
    reference_cell = np.asarray(reference_cell, dtype=float)
    final_cell = np.asarray(final_cell, dtype=float)

    deformation_gradient = np.linalg.solve(reference_cell, final_cell)
    singular_values = np.linalg.svd(deformation_gradient, compute_uv=False)

    principal_strains = singular_values - 1.0

    ref_a, ref_b, ref_gamma = cell_lengths_angle_2d(reference_cell)
    final_a, final_b, final_gamma = cell_lengths_angle_2d(final_cell)

    return {
        "principal_strain_1_percent": 100.0 * float(principal_strains[0]),
        "principal_strain_2_percent": 100.0 * float(principal_strains[1]),
        "max_abs_strain_percent": 100.0 * float(np.max(np.abs(principal_strains))),
        "rms_strain_percent": 100.0 * float(np.sqrt(np.mean(principal_strains**2))),
        "area_strain_percent": 100.0 * float(np.linalg.det(deformation_gradient) - 1.0),
        "a_initial": ref_a,
        "a_final": final_a,
        "a_strain_percent": 100.0 * (final_a / ref_a - 1.0),
        "b_initial": ref_b,
        "b_final": final_b,
        "b_strain_percent": 100.0 * (final_b / ref_b - 1.0),
        "gamma_initial_deg": ref_gamma,
        "gamma_final_deg": final_gamma,
        "gamma_change_deg": final_gamma - ref_gamma,
        "deformation_gradient": deformation_gradient,
    }


def heterostructure_layer_strains(
    bottom: Atoms,
    top: Atoms,
    heterostructure: Atoms,
    match: Match2D,
    bottom_name: str = "bottom",
    top_name: str = "top",
) -> dict:
    """Calculate explicit layer strain after building a two-layer heterostructure.

    The strain is calculated by comparing each matched but unstrained supercell
    against the final common in-plane cell of the constructed heterostructure.
    """
    bottom_unstrained = make_2d_supercell(bottom, match.bottom_matrix)
    top_unstrained = make_2d_supercell(top, match.top_matrix)

    final_cell = inplane_cell(heterostructure)

    return {
        bottom_name: strain_2d(
            reference_cell=inplane_cell(bottom_unstrained),
            final_cell=final_cell,
        ),
        top_name: strain_2d(
            reference_cell=inplane_cell(top_unstrained),
            final_cell=final_cell,
        ),
    }


def electrode_sandwich_layer_strains(
    electrode: Atoms,
    middle: Atoms,
    sandwich: Atoms,
    match: Match2D,
    electrode_name: str = "electrode",
    middle_name: str = "middle",
) -> dict:
    """Calculate explicit strain for electrode / middle / electrode stacks.

    Assumes the top and bottom electrodes use the same material, orientation,
    and matched supercell.
    """
    electrode_unstrained = make_2d_supercell(electrode, match.bottom_matrix)
    middle_unstrained = make_2d_supercell(middle, match.top_matrix)

    final_cell = inplane_cell(sandwich)

    electrode_strain = strain_2d(
        reference_cell=inplane_cell(electrode_unstrained),
        final_cell=final_cell,
    )

    middle_strain = strain_2d(
        reference_cell=inplane_cell(middle_unstrained),
        final_cell=final_cell,
    )

    return {
        f"bottom_{electrode_name}": electrode_strain,
        middle_name: middle_strain,
        f"top_{electrode_name}": electrode_strain,
    }


def print_layer_strains(strains: dict) -> None:
    """Print layer strain results compactly."""
    for name, s in strains.items():
        print(name)
        print(f"  principal strain 1: {s['principal_strain_1_percent']:.4f}%")
        print(f"  principal strain 2: {s['principal_strain_2_percent']:.4f}%")
        print(f"  max abs strain:     {s['max_abs_strain_percent']:.4f}%")
        print(f"  rms strain:         {s['rms_strain_percent']:.4f}%")
        print(f"  area strain:        {s['area_strain_percent']:.4f}%")
        print(f"  a strain:           {s['a_strain_percent']:.4f}%")
        print(f"  b strain:           {s['b_strain_percent']:.4f}%")
        print(f"  gamma change:       {s['gamma_change_deg']:.4f} deg")
        print()
