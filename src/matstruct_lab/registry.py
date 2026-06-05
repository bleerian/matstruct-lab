"""Named in-plane registry shifts for 2D heterostructures."""

from __future__ import annotations

import numpy as np
from ase import Atoms


REGISTRY_FRACTIONAL_SITES: dict[str, dict[str, tuple[float, float]]] = {
    "MoS2": {
        "Metal_on_Mo": (0.0, 0.0),
        "Metal_on_S": (1.0 / 3.0, 2.0 / 3.0),
        "Metal_on_Hollow": (2.0 / 3.0, 1.0 / 3.0),
    },
    "hBN": {
        "Metal_on_B": (0.0, 0.0),
        "Metal_on_N": (1.0 / 3.0, 2.0 / 3.0),
        "Metal_on_Hollow": (2.0 / 3.0, 1.0 / 3.0),
    },
}


def frac_shift_cart(cell2: np.ndarray, uv: tuple[float, float]) -> np.ndarray:
    """Convert a fractional in-plane shift to Cartesian xyz."""
    cell2 = np.asarray(cell2, dtype=float)[:2, :2]
    u, v = uv
    xy = u * cell2[0] + v * cell2[1]
    return np.array([xy[0], xy[1], 0.0], dtype=float)


def apply_fractional_registry_shift(
    atoms: Atoms,
    material: str,
    registry: str,
    anchor_frac: tuple[float, float] = (0.0, 0.0),
    extra_shift_frac: tuple[float, float] = (0.0, 0.0),
) -> Atoms:
    """Shift a layer so a named registry site is placed at anchor_frac.

    Examples
    --------
    MoS2 registries:
        Metal_on_Mo, Metal_on_S, Metal_on_Hollow

    hBN registries:
        Metal_on_B, Metal_on_N, Metal_on_Hollow
    """
    if material not in REGISTRY_FRACTIONAL_SITES:
        raise ValueError(f"Unsupported registry material: {material}")

    if registry not in REGISTRY_FRACTIONAL_SITES[material]:
        allowed = ", ".join(REGISTRY_FRACTIONAL_SITES[material])
        raise ValueError(f"Unknown registry '{registry}'. Allowed: {allowed}")

    out = atoms.copy()
    cell2 = out.cell.array[:2, :2]

    site_frac = REGISTRY_FRACTIONAL_SITES[material][registry]
    shift = (
        frac_shift_cart(cell2, anchor_frac)
        - frac_shift_cart(cell2, site_frac)
        + frac_shift_cart(cell2, extra_shift_frac)
    )

    out.translate(shift)
    out.wrap(eps=1e-8)
    return out
