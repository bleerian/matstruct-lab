#!/usr/bin/env python3
"""Run matstruct_lab.find_2d_matches and compute strain for each match.

This script is intentionally defensive because match object field names can vary
between lattice-matching libraries. Use --debug-attrs if it cannot infer the
film/substrate transforms or matched lattice vectors.

Edit MATCH_KWARGS below to match your search settings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from matstruct_lab.lattice_match import find_2d_matches, print_matches

from strain_utils_2d import (
    cell2d_from_structure,
    load_structure,
    strain_summary,
    to_plain_dict,
    transform_2d_cell,
    write_json,
)

# Put your matstruct_lab search settings here. Examples of likely options:
# MATCH_KWARGS = dict(max_area=250, max_mismatch=0.05, max_angle_diff=2.0)
MATCH_KWARGS: dict[str, Any] = {}


FILM_TRANSFORM_NAMES = [
    "film_transform", "film_transformation", "film_transformation_matrix",
    "film_sl_transform", "film_supercell_transform", "film_matrix",
    "film_supercell_matrix", "film_sc_matrix",
]
SUB_TRANSFORM_NAMES = [
    "substrate_transform", "substrate_transformation", "substrate_transformation_matrix",
    "sub_transform", "sub_transformation", "substrate_sl_transform",
    "substrate_supercell_transform", "substrate_matrix", "sub_matrix",
    "substrate_supercell_matrix", "sub_supercell_matrix", "substrate_sc_matrix",
]
FILM_VECTOR_NAMES = [
    "film_sl_vectors", "film_vectors", "film_supercell_vectors", "film_lattice_vectors",
    "film_supercell_lattice", "film_sl_lattice",
]
SUB_VECTOR_NAMES = [
    "substrate_sl_vectors", "substrate_vectors", "sub_vectors", "substrate_supercell_vectors",
    "substrate_lattice_vectors", "substrate_supercell_lattice", "substrate_sl_lattice",
]


def get_value(obj: Any, names: list[str]) -> Any | None:
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def as_2d_cell_from_vectors(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.array(value, dtype=float)
    # Common cases: [[ax, ay, az], [bx, by, bz]] or [[ax, bx], [ay, by]]
    if arr.shape == (2, 3):
        return np.column_stack([arr[0, :2], arr[1, :2]])
    if arr.shape == (3, 2):
        return arr[:2, :]
    if arr.shape == (2, 2):
        # Ambiguous. Treat rows as vectors if that resembles pymatgen/lattice output.
        return np.column_stack([arr[0, :], arr[1, :]])
    if arr.shape == (3, 3):
        return np.column_stack([arr[0, :2], arr[1, :2]])
    return None


def as_transform(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.array(value, dtype=float)
    if arr.shape == (3, 3):
        arr = arr[:2, :2]
    if arr.shape == (2, 2):
        return arr
    return None


def cell_from_match(match: Any, primitive: Structure, transform_names: list[str], vector_names: list[str]) -> np.ndarray | None:
    vectors = as_2d_cell_from_vectors(get_value(match, vector_names))
    if vectors is not None:
        return vectors
    transform = as_transform(get_value(match, transform_names))
    if transform is not None:
        return transform_2d_cell(cell2d_from_structure(primitive), transform)
    return None


def debug_attrs(match: Any) -> None:
    print("\nCould not infer lattice fields. Inspecting first match object.\n")
    if isinstance(match, dict):
        print("dict keys:", sorted(match.keys()))
    else:
        attrs = [a for a in dir(match) if not a.startswith("_")]
        print("attributes:", attrs)
        for a in attrs:
            try:
                v = getattr(match, a)
                if isinstance(v, (str, int, float, list, tuple, np.ndarray)):
                    print(f"  {a} = {v}")
            except Exception:
                pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--film", required=True, help="Film/2D layer structure file")
    p.add_argument("--substrate", required=True, help="Substrate/second layer structure file")
    p.add_argument("--csv", default="lattice_match_strain.csv")
    p.add_argument("--json", default="lattice_match_strain.json")
    p.add_argument("--debug-attrs", action="store_true")
    args = p.parse_args()

    film = load_structure(args.film)
    substrate = load_structure(args.substrate)

    matches = list(find_2d_matches(film, substrate, **MATCH_KWARGS))
    print_matches(matches)

    if not matches:
        raise SystemExit("No matches returned by find_2d_matches.")

    records = []
    for i, match in enumerate(matches):
        film_cell = cell_from_match(match, film, FILM_TRANSFORM_NAMES, FILM_VECTOR_NAMES)
        sub_cell = cell_from_match(match, substrate, SUB_TRANSFORM_NAMES, SUB_VECTOR_NAMES)

        if film_cell is None or sub_cell is None:
            if args.debug_attrs:
                debug_attrs(match)
            raise SystemExit(
                "Could not infer film/substrate supercell vectors from match object. "
                "Run with --debug-attrs and add the correct field names to *_NAMES lists."
            )

        # Two useful conventions. If substrate is fixed, film_to_substrate is the film strain.
        r1 = to_plain_dict(strain_summary(film_cell, sub_cell, "film_supercell", "substrate_supercell"))
        r2 = to_plain_dict(strain_summary(sub_cell, film_cell, "substrate_supercell", "film_supercell"))
        for r in (r1, r2):
            r["match_index"] = i
            # Copy any common scalar metadata from the match object if present.
            for key in ["area", "match_area", "mismatch", "strain", "angle", "angle_diff", "rotation"]:
                val = get_value(match, [key])
                if isinstance(val, (int, float, str)):
                    r[f"match_{key}"] = val
        records.extend([r1, r2])

    df = pd.DataFrame(records)
    front = [
        "match_index", "reference", "target", "eps_a_percent", "eps_b_percent", "delta_gamma_deg",
        "area_strain_percent", "principal_strain_1_percent", "principal_strain_2_percent",
        "rms_principal_strain_percent", "max_abs_principal_strain_percent",
    ]
    other = [c for c in df.columns if c not in front]
    df = df[[c for c in front if c in df.columns] + other]
    df.to_csv(args.csv, index=False)
    write_json(args.json, records)

    print("\nComputed strain table:")
    print(df[[c for c in front if c in df.columns]].to_string(index=False))
    print(f"\nWrote: {Path(args.csv).resolve()}")
    print(f"Wrote: {Path(args.json).resolve()}")


if __name__ == "__main__":
    main()
